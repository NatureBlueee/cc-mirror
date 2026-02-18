"""
db.py — CC Mirror SQLite schema 初始化 + 连接管理

职责：
  - 定义完整的 DDL（所有表 + FTS5 虚拟表）
  - init_db()：创建并初始化数据库
  - get_or_create_db()：增量友好的入口，已存在则复用

全部使用 CREATE TABLE IF NOT EXISTS，确保幂等。
"""

# Python 3.9 兼容：支持 X | Y 类型注解语法
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
-- WAL 模式：提升并发读写性能，解析期间避免锁竞争
PRAGMA journal_mode=WAL;

-- 强制外键约束
PRAGMA foreign_keys=ON;

-- -----------------------------------------------------------------------
-- sessions 表
-- 每个 JSONL 文件对应一个 session
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id                      TEXT PRIMARY KEY,
    project                 TEXT NOT NULL,
    start_time              TEXT NOT NULL,         -- ISO8601
    end_time                TEXT,                  -- ISO8601，最后一条消息时间
    message_count           INTEGER DEFAULT 0,
    user_message_count      INTEGER DEFAULT 0,
    assistant_message_count INTEGER DEFAULT 0,
    total_input_tokens      INTEGER DEFAULT 0,
    total_output_tokens     INTEGER DEFAULT 0,
    cost_usd                REAL    DEFAULT 0,
    primary_model           TEXT,                  -- 出现次数最多的模型
    git_branch              TEXT,
    has_compact             BOOLEAN DEFAULT FALSE,  -- 是否含 compact 事件
    compact_count           INTEGER DEFAULT 0,
    has_subagents           BOOLEAN DEFAULT FALSE,  -- 是否触发过子 agent
    jsonl_path              TEXT    NOT NULL        -- 源文件绝对路径
);

-- -----------------------------------------------------------------------
-- messages 表
-- 存元数据 + 关键文本（不存原始 JSON，控制体积）
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    uuid                    TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL REFERENCES sessions(id),
    parent_uuid             TEXT,
    type                    TEXT NOT NULL,          -- "user" | "assistant" | "system" …
    timestamp               TEXT NOT NULL,           -- ISO8601
    has_user_text           BOOLEAN DEFAULT FALSE,
    user_text_length        INTEGER DEFAULT 0,
    user_text               TEXT,
    assistant_text          TEXT,
    has_tool_use            BOOLEAN DEFAULT FALSE,
    tool_names              TEXT,                   -- JSON array，如 ["Read","Bash"]
    tool_use_summary        TEXT,                   -- 简短摘要，供 L2 快速过滤
    has_tool_result         BOOLEAN DEFAULT FALSE,
    has_error               BOOLEAN DEFAULT FALSE,
    has_thinking            BOOLEAN DEFAULT FALSE,
    thinking_text           TEXT,
    token_input             INTEGER DEFAULT 0,
    token_output            INTEGER DEFAULT 0,
    is_candidate_correction BOOLEAN DEFAULT FALSE,  -- L1 粗筛：可能是纠正行为
    sequence_num            INTEGER                 -- 在 session 内的顺序
);

-- -----------------------------------------------------------------------
-- tool_calls 表
-- 每个工具调用单独一行，方便 workflow_clusters 分析
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tool_calls (
    id           TEXT    PRIMARY KEY,               -- "{message_uuid}_{idx}"
    session_id   TEXT    NOT NULL REFERENCES sessions(id),
    message_uuid TEXT    NOT NULL REFERENCES messages(uuid),
    tool_name    TEXT    NOT NULL,
    input_summary TEXT,                             -- input dict 截断至 100 字符
    is_error     BOOLEAN DEFAULT FALSE,
    timestamp    TEXT    NOT NULL,
    sequence_num INTEGER                            -- 在 session 内的全局顺序
);

-- -----------------------------------------------------------------------
-- corrections 表
-- L2 精筛后填充；L1 只写 messages.is_candidate_correction
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corrections (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT    NOT NULL REFERENCES sessions(id),
    project                 TEXT    NOT NULL,
    user_message_uuid       TEXT    NOT NULL REFERENCES messages(uuid),
    assistant_message_uuid  TEXT    NOT NULL,       -- 被纠正的 assistant 消息
    cc_did                  TEXT    NOT NULL,       -- CC 做了什么（L2 提取）
    user_wanted             TEXT    NOT NULL,       -- 用户真正想要的（L2 提取）
    correction_type         TEXT    NOT NULL,       -- "style"|"scope"|"approach"|"factual"|…
    is_generalizable        BOOLEAN,               -- 能否泛化为规则（L2 判断）
    confidence              REAL,                  -- L2 置信度 0-1
    raw_user_text           TEXT,
    timestamp               TEXT    NOT NULL
);

-- -----------------------------------------------------------------------
-- decisions 表
-- 对话中的架构/方案决策（L2 提取）
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL REFERENCES sessions(id),
    project          TEXT    NOT NULL,
    decision         TEXT    NOT NULL,             -- 决策内容
    context          TEXT,                         -- 背景
    alternatives     TEXT,                         -- JSON array，被否决的选项
    reasoning        TEXT,                         -- 用户给的理由
    implicit_criteria TEXT,                        -- JSON array，隐含评判标准
    timestamp        TEXT    NOT NULL,
    tags             TEXT                          -- JSON array，如 ["architecture","naming"]
);

-- -----------------------------------------------------------------------
-- repeated_prompts 表
-- L1 粗筛：相同用户文本在 ≥3 个不同 session 中出现
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS repeated_prompts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_text TEXT    NOT NULL,
    occurrences    INTEGER NOT NULL,
    session_ids    TEXT    NOT NULL,               -- JSON array
    project_ids    TEXT    NOT NULL,               -- JSON array
    first_seen     TEXT    NOT NULL,               -- ISO8601
    last_seen      TEXT    NOT NULL                -- ISO8601
);

-- -----------------------------------------------------------------------
-- workflow_clusters 表
-- L2/L3 聚合：相似工具调用序列模式
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workflow_clusters (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_sequence_pattern TEXT   NOT NULL,         -- JSON array，如 ["Read","Bash","Edit"]
    session_ids          TEXT    NOT NULL,          -- JSON array
    session_count        INTEGER NOT NULL,
    similarity_threshold REAL    NOT NULL,          -- 聚合时用的阈值
    description          TEXT,                     -- L3 生成的自然语言描述
    skill_suggestion     TEXT                      -- JSON object，Skill 建议草稿
);

-- -----------------------------------------------------------------------
-- llm_calls 表
-- 记录 L2/L3/L4 阶段所有 LLM API 调用，用于成本核算
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    stage        TEXT    NOT NULL,                 -- "L2"|"L3"|"L4"
    model        TEXT    NOT NULL,
    prompt_hash  TEXT    NOT NULL,                 -- sha256 前 16 字符，去重用
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd     REAL,
    timestamp    TEXT    NOT NULL,
    duration_ms  INTEGER
);

-- -----------------------------------------------------------------------
-- FTS5 全文搜索虚拟表
-- content= 指向 messages，rowid= 对应 messages.rowid
-- 支持对 user_text 全文检索
-- -----------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    uuid,
    session_id,
    user_text,
    content=messages,
    content_rowid=rowid
);
"""

# FTS5 触发器：保持 messages_fts 与 messages 同步
# 注意：content= 外部内容表模式下，FTS 不自动同步，需要触发器
_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS messages_fts_ai
    AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, uuid, session_id, user_text)
            VALUES (new.rowid, new.uuid, new.session_id, new.user_text);
    END;

CREATE TRIGGER IF NOT EXISTS messages_fts_ad
    AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, uuid, session_id, user_text)
            VALUES ('delete', old.rowid, old.uuid, old.session_id, old.user_text);
    END;

CREATE TRIGGER IF NOT EXISTS messages_fts_au
    AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, uuid, session_id, user_text)
            VALUES ('delete', old.rowid, old.uuid, old.session_id, old.user_text);
        INSERT INTO messages_fts(rowid, uuid, session_id, user_text)
            VALUES (new.rowid, new.uuid, new.session_id, new.user_text);
    END;
"""


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    """
    创建（或重新初始化）指定路径的 SQLite 数据库。

    - 执行完整 DDL（幂等，IF NOT EXISTS）
    - 开启 WAL 模式 + foreign_keys
    - 创建 FTS5 同步触发器
    - 返回已打开的连接（调用方负责 close）

    Args:
        db_path: SQLite 文件路径，父目录必须存在。

    Returns:
        sqlite3.Connection，row_factory 已设置为 sqlite3.Row。
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"[db] 无法打开数据库 {db_path}: {e}", file=sys.stderr)
        raise

    try:
        # executescript 不支持参数化，但这里全是 DDL，安全
        conn.executescript(_DDL)
        conn.executescript(_FTS_TRIGGERS)
        conn.commit()
    except Exception as e:
        print(f"[db] schema 初始化失败: {e}", file=sys.stderr)
        conn.close()
        raise

    return conn


def get_or_create_db(db_path: Path) -> sqlite3.Connection:
    """
    增量友好的数据库入口：
    - 如果文件已存在且包含 sessions 表，直接打开（跳过重建）
    - 否则调用 init_db() 创建

    这样在增量解析时不会因为 DDL 执行而产生不必要的 I/O。

    Args:
        db_path: SQLite 文件路径。

    Returns:
        sqlite3.Connection，row_factory 已设置为 sqlite3.Row。
    """
    db_path = Path(db_path)

    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            # 验证 schema 完整性：检查 sessions 表是否存在
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            if cur.fetchone() is not None:
                # 已初始化：只确保 PRAGMA 生效（每次连接都要重设）
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                return conn
            # 表不存在（空文件）：走完整初始化
            conn.close()
        except Exception as e:
            print(f"[db] 打开已有数据库失败，将重新初始化: {e}", file=sys.stderr)

    return init_db(db_path)
