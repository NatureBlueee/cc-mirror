"""
l1_parser.py — CC Mirror L1 解析层（无 LLM）

职责：
  - 遍历 ~/.claude/projects/**/*.jsonl
  - 跳过子 agent 文件（agent-*.jsonl）和 isSidechain 消息
  - 解析每条消息，提取元数据 + 文本 + 工具调用
  - 写入 SQLite（sessions / messages / tool_calls）
  - L1 粗筛标记：is_candidate_correction（纠正候选）
  - 增量更新：session 已存在则跳过整个文件
  - 重复提示检测：写入 repeated_prompts 表

L2 层会在此基础上做 LLM 精筛（corrections / decisions / workflow_clusters）。
"""

# from __future__ import annotations 让 Python 3.9 支持 X | Y 联合类型注解语法
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 确认性用语（不算纠正候选）
_CONFIRMATIONS: frozenset[str] = frozenset({
    "好", "好的", "ok", "继续", "可以", "是", "对", "嗯", "行",
    "没问题", "go", "yes", "sure", "proceed", "y", "好吧", "done", "next",
    "好的好的", "没问题的", "收到", "明白", "了解", "ok的", "okok",
    "继续吧", "你继续", "go on", "alright", "yep", "yup", "yeah",
})

# 否定信号：包含这些词的短文本不算纯确认
_NEGATION_SIGNALS: tuple[str, ...] = (
    "不", "no", "别", "wrong", "错", "换", "改", "重", "取消",
    "停", "cancel", "undo", "rollback", "回滚",
)

# 候选纠正最小长度（字符数）
_CORRECTION_MIN_LEN = 15

# 重复提示检测：user_text 最大长度
_REPEATED_PROMPT_MAX_LEN = 200

# 重复提示检测：最少出现 session 数
_REPEATED_PROMPT_MIN_SESSIONS = 3


# ---------------------------------------------------------------------------
# 纯函数工具
# ---------------------------------------------------------------------------

def is_confirmation_only(text: str) -> bool:
    """
    判断用户消息是否只是简单确认（不是纠正）。

    规则（按顺序）：
    1. 精确匹配已知确认词集合 → True
    2. 文本极短（< 10 字符）且不含否定信号 → True
    3. 否则 → False
    """
    cleaned = text.strip().lower()
    if cleaned in _CONFIRMATIONS:
        return True
    if len(cleaned) < 10 and not any(neg in cleaned for neg in _NEGATION_SIGNALS):
        return True
    return False


def _extract_content(content: str | list) -> dict:
    """
    解析 message.content 字段。

    content 可以是：
    - str（旧版 CC）
    - list（新版 CC，含 text / tool_use / tool_result / thinking 块）

    返回 dict：
    {
        'user_text': str | None,
        'assistant_text': str | None,
        'thinking_text': str | None,
        'has_thinking': bool,
        'tool_uses': list[dict],   # [{'name': str, 'input_summary': str}]
        'has_tool_result': bool,
        'has_error': bool,
    }
    """
    result = {
        "user_text": None,
        "assistant_text": None,
        "thinking_text": None,
        "has_thinking": False,
        "tool_uses": [],
        "has_tool_result": False,
        "has_error": False,
    }

    if isinstance(content, str):
        # 旧版：content 直接是纯文本，角色由调用方传入后设置
        result["_raw_str"] = content
        return result

    if not isinstance(content, list):
        return result

    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "text":
            text_parts.append(item.get("text", ""))

        elif item_type == "thinking":
            thinking_parts.append(item.get("thinking", ""))
            result["has_thinking"] = True

        elif item_type == "tool_use":
            # 工具调用（assistant 发出）
            name = item.get("name", "unknown")
            raw_input = item.get("input", {})
            # 截断 input 摘要，避免大型内容（如完整文件）膨胀数据库
            input_summary = str(raw_input)[:200]
            result["tool_uses"].append({
                "id": item.get("id", ""),
                "name": name,
                "input_summary": input_summary,
            })

        elif item_type == "tool_result":
            result["has_tool_result"] = True
            # is_error 可能是布尔或字符串
            is_error = item.get("is_error", False)
            if is_error:
                result["has_error"] = True

    if text_parts:
        result["_text_joined"] = "\n".join(text_parts)
    if thinking_parts:
        result["thinking_text"] = "\n".join(thinking_parts)

    return result


def _summarize_tool_uses(tool_uses: list[dict]) -> tuple[str, str]:
    """
    从 tool_uses 列表提取：
    - tool_names_json：JSON 数组字符串，如 '["Read","Bash"]'
    - tool_use_summary：自然语言摘要，如 "Read×2, Bash×1"
    """
    if not tool_uses:
        return "[]", ""
    names = [t["name"] for t in tool_uses]
    tool_names_json = json.dumps(names, ensure_ascii=False)
    counter = Counter(names)
    summary = ", ".join(f"{name}×{cnt}" for name, cnt in counter.most_common())
    return tool_names_json, summary


# ---------------------------------------------------------------------------
# 核心解析逻辑
# ---------------------------------------------------------------------------

def parse_session(jsonl_path: Path, project: str, db: sqlite3.Connection) -> dict:
    """
    解析单个 JSONL 文件，写入 sessions / messages / tool_calls 三张表。

    容错原则：
    - 每行独立 try/except，解析失败的行记录错误但不中断
    - 写入失败时回滚该 session 的所有变更

    Args:
        jsonl_path: JSONL 文件绝对路径
        project:    项目名称（目录名）
        db:         SQLite 连接（调用方管理事务）

    Returns:
        {
            'session_id': str,
            'messages': int,      # 成功解析的消息数
            'parse_errors': int,  # 解析失败的行数
        }
    """
    stats = {"session_id": "", "messages": 0, "parse_errors": 0}
    raw_lines: list[str] = []

    try:
        raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"[l1] 读取文件失败 {jsonl_path}: {e}", file=sys.stderr)
        stats["parse_errors"] += 1
        return stats

    # ---- 第一遍：提取 session_id 和基础元数据 ----
    session_id: str = ""
    first_timestamp: str = ""
    last_timestamp: str = ""
    message_rows: list[dict] = []
    tool_call_rows: list[dict] = []
    global_seq = 0  # session 内全局顺序计数器

    # 聚合统计
    total_input_tokens = 0
    total_output_tokens = 0
    user_msg_count = 0
    assistant_msg_count = 0
    model_counter: Counter = Counter()
    git_branch: str | None = None
    has_compact = False
    compact_count = 0
    has_subagents = False
    prev_was_assistant = False  # 用于候选纠正判断

    for line_num, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[l1] JSON 解析失败 {jsonl_path}:{line_num}: {e}", file=sys.stderr)
            stats["parse_errors"] += 1
            continue

        # ---- 跳过子 agent 消息（isSidechain） ----
        if msg.get("isSidechain", False):
            continue

        # ---- 提取 session_id（取第一条消息的） ----
        msg_session_id = msg.get("sessionId", "")
        if not session_id and msg_session_id:
            session_id = msg_session_id

        # ---- 时间戳 ----
        ts = msg.get("timestamp", "")
        if ts:
            if not first_timestamp:
                first_timestamp = ts
            last_timestamp = ts

        # ---- compact 事件检测 ----
        msg_type = msg.get("type", "")
        if msg_type in ("compact", "summary"):
            has_compact = True
            compact_count += 1
            continue  # compact 行不写进 messages

        # ---- git branch（从 system prompt 或 metadata 提取） ----
        if git_branch is None:
            git_branch = msg.get("metadata", {}).get("git_branch") or \
                         msg.get("gitBranch")

        # ---- 解析 message 内容 ----
        inner = msg.get("message", {})
        if not isinstance(inner, dict):
            continue

        role = inner.get("role", msg_type)  # 部分行 role 在外层 type 里
        uuid = msg.get("uuid", "")
        parent_uuid = msg.get("parentUuid", "")

        # content 解析
        content = inner.get("content", [])
        parsed = _extract_content(content)

        # 根据 role 分配文本
        user_text: str | None = None
        assistant_text: str | None = None

        raw_str = parsed.get("_raw_str")  # 旧版 str content
        text_joined = parsed.get("_text_joined")  # 新版 list 合并文本

        if role == "user":
            user_text = raw_str or text_joined
            user_msg_count += 1
        elif role == "assistant":
            assistant_text = raw_str or text_joined
            assistant_msg_count += 1
        # system / tool 等其他 role 暂不单独处理，保留在 type 字段

        # ---- 工具调用 ----
        tool_uses = parsed["tool_uses"]
        tool_names_json, tool_use_summary = _summarize_tool_uses(tool_uses)

        # ---- 候选纠正标记（L1 粗筛） ----
        # 条件：user 消息 + 有文本 + > 15 字符 + 不是纯确认 + 前一条是 assistant
        is_candidate = False
        if (role == "user"
                and user_text
                and len(user_text.strip()) > _CORRECTION_MIN_LEN
                and not is_confirmation_only(user_text)
                and prev_was_assistant):
            is_candidate = True

        # ---- token 统计 ----
        usage = inner.get("usage", {})
        tok_in = usage.get("input_tokens", 0) or 0
        tok_out = usage.get("output_tokens", 0) or 0
        total_input_tokens += tok_in
        total_output_tokens += tok_out

        # ---- 模型统计 ----
        model = inner.get("model", "")
        if model:
            model_counter[model] += 1

        # ---- 子 agent 检测（tool_use 中含 agent_ 前缀或 Dispatch 工具名） ----
        agent_tool_names = {"agent", "dispatch", "spawn_agent", "computer_use"}
        if any(t["name"].lower() in agent_tool_names for t in tool_uses):
            has_subagents = True

        # ---- 构造 message row ----
        seq = global_seq
        global_seq += 1
        message_rows.append({
            "uuid": uuid,
            "session_id": session_id,
            "parent_uuid": parent_uuid or None,
            "type": role,
            "timestamp": ts,
            "has_user_text": bool(user_text),
            "user_text_length": len(user_text.strip()) if user_text else 0,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "has_tool_use": bool(tool_uses),
            "tool_names": tool_names_json if tool_uses else "[]",
            "tool_use_summary": tool_use_summary or None,
            "has_tool_result": parsed["has_tool_result"],
            "has_error": parsed["has_error"],
            "has_thinking": parsed["has_thinking"],
            "thinking_text": parsed.get("thinking_text"),
            "token_input": tok_in,
            "token_output": tok_out,
            "is_candidate_correction": is_candidate,
            "sequence_num": seq,
        })

        # ---- 构造 tool_call rows ----
        for idx, tool in enumerate(tool_uses):
            tool_call_rows.append({
                "id": f"{uuid}_{idx}",
                "session_id": session_id,
                "message_uuid": uuid,
                "tool_name": tool["name"],
                "input_summary": tool["input_summary"],
                "is_error": parsed["has_error"],
                "timestamp": ts,
                "sequence_num": seq,  # 与 message 同序号
            })

        # 更新前一条 role（用于下一条的候选纠正判断）
        prev_was_assistant = (role == "assistant")

    # ---- 会话无效检查 ----
    if not session_id:
        print(f"[l1] 无有效 session_id，跳过: {jsonl_path}", file=sys.stderr)
        return stats

    stats["session_id"] = session_id

    # ---- 主要模型 ----
    primary_model = model_counter.most_common(1)[0][0] if model_counter else None

    # ---- 写入数据库（单事务） ----
    try:
        cur = db.cursor()

        # sessions
        cur.execute(
            """
            INSERT OR IGNORE INTO sessions (
                id, project, start_time, end_time,
                message_count, user_message_count, assistant_message_count,
                total_input_tokens, total_output_tokens, cost_usd,
                primary_model, git_branch,
                has_compact, compact_count, has_subagents,
                jsonl_path
            ) VALUES (
                :id, :project, :start_time, :end_time,
                :message_count, :user_message_count, :assistant_message_count,
                :total_input_tokens, :total_output_tokens, :cost_usd,
                :primary_model, :git_branch,
                :has_compact, :compact_count, :has_subagents,
                :jsonl_path
            )
            """,
            {
                "id": session_id,
                "project": project,
                "start_time": first_timestamp,
                "end_time": last_timestamp,
                "message_count": len(message_rows),
                "user_message_count": user_msg_count,
                "assistant_message_count": assistant_msg_count,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "cost_usd": 0,  # L2 根据模型定价计算
                "primary_model": primary_model,
                "git_branch": git_branch,
                "has_compact": has_compact,
                "compact_count": compact_count,
                "has_subagents": has_subagents,
                "jsonl_path": str(jsonl_path),
            },
        )

        # messages（批量插入）
        if message_rows:
            cur.executemany(
                """
                INSERT OR IGNORE INTO messages (
                    uuid, session_id, parent_uuid, type, timestamp,
                    has_user_text, user_text_length, user_text,
                    assistant_text,
                    has_tool_use, tool_names, tool_use_summary,
                    has_tool_result, has_error,
                    has_thinking, thinking_text,
                    token_input, token_output,
                    is_candidate_correction, sequence_num
                ) VALUES (
                    :uuid, :session_id, :parent_uuid, :type, :timestamp,
                    :has_user_text, :user_text_length, :user_text,
                    :assistant_text,
                    :has_tool_use, :tool_names, :tool_use_summary,
                    :has_tool_result, :has_error,
                    :has_thinking, :thinking_text,
                    :token_input, :token_output,
                    :is_candidate_correction, :sequence_num
                )
                """,
                message_rows,
            )

        # tool_calls（批量插入）
        if tool_call_rows:
            cur.executemany(
                """
                INSERT OR IGNORE INTO tool_calls (
                    id, session_id, message_uuid, tool_name,
                    input_summary, is_error, timestamp, sequence_num
                ) VALUES (
                    :id, :session_id, :message_uuid, :tool_name,
                    :input_summary, :is_error, :timestamp, :sequence_num
                )
                """,
                tool_call_rows,
            )

        db.commit()
        stats["messages"] = len(message_rows)

    except Exception as e:
        print(f"[l1] 写入数据库失败 session={session_id}: {e}", file=sys.stderr)
        db.rollback()
        stats["parse_errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# 重复提示检测
# ---------------------------------------------------------------------------

def _detect_repeated_prompts(db: sqlite3.Connection) -> int:
    """
    全库扫描 messages 表，找出出现在 ≥3 个不同 session 中的相同 user_text（精确匹配）。
    写入 repeated_prompts 表。

    只处理 user_text 长度 < 200 字符的短消息（长消息不太可能是"习惯性提示"）。

    Returns:
        写入的记录数
    """
    # 拉取所有短 user 消息（避免把完整文件内容拉进内存）
    cur = db.execute(
        """
        SELECT user_text, session_id, timestamp
        FROM messages
        WHERE has_user_text = 1
          AND user_text_length > 0
          AND user_text_length < :max_len
        ORDER BY timestamp ASC
        """,
        {"max_len": _REPEATED_PROMPT_MAX_LEN},
    )
    rows = cur.fetchall()

    # 按 user_text 分组：text → [(session_id, timestamp), ...]
    text_to_sessions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        text = (row["user_text"] or "").strip()
        if not text:
            continue
        text_to_sessions[text].append((row["session_id"], row["timestamp"]))

    # 筛选：出现在 ≥3 个不同 session 中
    written = 0
    try:
        cur = db.cursor()
        for canonical_text, occurrences in text_to_sessions.items():
            # 统计不同 session 数
            unique_sessions = list({sid for sid, _ in occurrences})
            if len(unique_sessions) < _REPEATED_PROMPT_MIN_SESSIONS:
                continue

            timestamps = [ts for _, ts in occurrences]
            first_seen = min(timestamps)
            last_seen = max(timestamps)

            # 获取对应的 project_ids
            placeholders = ",".join("?" * len(unique_sessions))
            proj_cur = db.execute(
                f"SELECT DISTINCT project FROM sessions WHERE id IN ({placeholders})",
                unique_sessions,
            )
            project_ids = [r[0] for r in proj_cur.fetchall()]

            cur.execute(
                """
                INSERT INTO repeated_prompts
                    (canonical_text, occurrences, session_ids, project_ids, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_text,
                    len(occurrences),
                    json.dumps(unique_sessions, ensure_ascii=False),
                    json.dumps(project_ids, ensure_ascii=False),
                    first_seen,
                    last_seen,
                ),
            )
            written += 1

        db.commit()
    except Exception as e:
        print(f"[l1] repeated_prompts 写入失败: {e}", file=sys.stderr)
        db.rollback()

    return written


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def parse_all_sessions(
    claude_dir: Path,
    db: sqlite3.Connection,
    project_filter: str | None = None,
) -> dict:
    """
    遍历 claude_dir（通常是 ~/.claude/projects/）下所有 JSONL 文件，
    增量解析并写入 SQLite。

    目录结构假设：
        claude_dir/
            {project_name}/           # URL 编码的项目路径
                *.jsonl               # session 文件
                agent-*.jsonl         # 子 agent 文件（跳过）

    增量策略：
        session_id 已在 sessions 表中 → 跳过整个文件（不重复解析）

    Args:
        claude_dir:     ~/.claude/projects/ 路径
        db:             已初始化的 SQLite 连接
        project_filter: 只处理指定项目名（None 表示全部）

    Returns:
        {
            'sessions': int,             # 新解析的 session 数
            'messages': int,             # 新解析的消息数
            'candidate_corrections': int,# is_candidate_correction=True 的消息数
            'repeated_prompts': int,     # 写入 repeated_prompts 的记录数
            'projects': list[str],       # 处理的项目名列表
            'parse_errors': int,         # 解析失败的行/文件数
            'skipped_sessions': int,     # 已在 DB 中跳过的 session 数
        }
    """
    claude_dir = Path(claude_dir)
    if not claude_dir.exists():
        print(f"[l1] claude_dir 不存在: {claude_dir}", file=sys.stderr)
        return {
            "sessions": 0,
            "messages": 0,
            "candidate_corrections": 0,
            "repeated_prompts": 0,
            "projects": [],
            "parse_errors": 0,
            "skipped_sessions": 0,
        }

    # 如果传入的是 ~/.claude 根目录，自动切换到 projects/ 子目录
    projects_subdir = claude_dir / "projects"
    if projects_subdir.exists():
        claude_dir = projects_subdir

    # 预加载已存在的 session_id 集合（增量更新：避免重复解析）
    existing_sessions: set[str] = set()
    try:
        cur = db.execute("SELECT id FROM sessions")
        existing_sessions = {row[0] for row in cur.fetchall()}
    except Exception as e:
        print(f"[l1] 读取已有 session 失败: {e}", file=sys.stderr)

    total_sessions = 0
    total_messages = 0
    total_errors = 0
    skipped_sessions = 0
    processed_projects: list[str] = []

    # 遍历项目目录
    project_dirs = sorted(
        p for p in claude_dir.iterdir() if p.is_dir()
    )

    for project_dir in project_dirs:
        project = project_dir.name

        # 项目过滤
        if project_filter and project != project_filter:
            continue

        # 收集该项目下所有 JSONL 文件（排除子 agent 文件）
        jsonl_files = sorted(
            f for f in project_dir.glob("*.jsonl")
            if not f.name.startswith("agent-")
        )

        if not jsonl_files:
            continue

        processed_projects.append(project)
        print(f"[l1] 处理项目: {project} ({len(jsonl_files)} 个文件)", file=sys.stderr)

        for jsonl_path in jsonl_files:
            # ---- 增量检查：从文件中快速读取 session_id ----
            # 只读前几行提取 sessionId，避免全量加载
            session_id_from_file = _peek_session_id(jsonl_path)

            if session_id_from_file and session_id_from_file in existing_sessions:
                skipped_sessions += 1
                continue  # 已解析过，跳过

            # ---- 解析并写入 ----
            result = parse_session(jsonl_path, project, db)

            if result["session_id"]:
                total_sessions += 1
                existing_sessions.add(result["session_id"])  # 本次新增

            total_messages += result["messages"]
            total_errors += result["parse_errors"]

    # ---- 候选纠正统计 + 工具调用统计 ----
    candidate_corrections = 0
    total_tool_calls = 0
    user_text_messages = 0
    try:
        cur = db.execute(
            "SELECT COUNT(*) FROM messages WHERE is_candidate_correction = 1"
        )
        row = cur.fetchone()
        candidate_corrections = row[0] if row else 0

        cur = db.execute("SELECT COUNT(*) FROM tool_calls")
        row = cur.fetchone()
        total_tool_calls = row[0] if row else 0

        cur = db.execute(
            "SELECT COUNT(*) FROM messages WHERE user_text IS NOT NULL AND user_text != ''"
        )
        row = cur.fetchone()
        user_text_messages = row[0] if row else 0
    except Exception as e:
        print(f"[l1] 统计候选纠正失败: {e}", file=sys.stderr)

    # ---- 重复提示检测 ----
    repeated_prompts = 0
    if total_sessions > 0:
        # 只在有新数据时运行（全量检测，包含历史数据）
        try:
            # 清空旧检测结果，重新计算（保证准确性）
            db.execute("DELETE FROM repeated_prompts")
            db.commit()
        except Exception as e:
            print(f"[l1] 清空 repeated_prompts 失败: {e}", file=sys.stderr)

        repeated_prompts = _detect_repeated_prompts(db)
        print(f"[l1] 重复提示检测完成，写入 {repeated_prompts} 条", file=sys.stderr)

    return {
        "sessions": total_sessions,
        "messages": total_messages,
        "tool_calls": total_tool_calls,
        "user_text_messages": user_text_messages,
        "candidate_corrections": candidate_corrections,
        "repeated_prompts": repeated_prompts,
        "projects": processed_projects,
        "parse_errors": total_errors,
        "skipped_sessions": skipped_sessions,
    }


def _peek_session_id(jsonl_path: Path, max_lines: int = 10) -> str | None:
    """
    只读取文件前 max_lines 行，快速提取 sessionId。
    避免为了增量检查而全量加载大型 JSONL。

    Returns:
        session_id 字符串，或 None（未找到）
    """
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    sid = msg.get("sessionId", "")
                    if sid:
                        return sid
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[l1] peek_session_id 失败 {jsonl_path}: {e}", file=sys.stderr)
    return None
