"""
l2_correction.py — CC Mirror L2 纠正检测层

职责：
  - 从 messages 表读取 is_candidate_correction=1 的候选消息
  - 为每条候选构造 prompt（含前 1-3 条 assistant 消息上下文）
  - asyncio 并发调用 Sonnet，解析 JSON 响应
  - 写入 corrections 表
  - 通过 BudgetController 记录到 llm_calls 表
  - 支持幂等重跑（检查 corrections 表是否已处理）

L1 粗筛 → L2 精筛：L1 负责候选标记，L2 负责语义判断。
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from typing import Any

from .budget import BudgetController


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 使用的模型
_MODEL = "claude-sonnet-4-6"

# 上下文：最多取前几条 assistant 消息
_MAX_CONTEXT_ASSISTANT_MSGS = 3

# 每条 assistant 消息的最大字符数（截断控制 token）
_ASSISTANT_TEXT_MAX_CHARS = 500

# max_tokens：给 JSON 响应留足空间，同时避免截断
_MAX_TOKENS = 512

# Sonnet 4.6 定价（每 1M token，美元）
_PRICE_INPUT_PER_M  = 3.0
_PRICE_OUTPUT_PER_M = 15.0


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def build_prompt(
    user_text: str,
    context_assistant_msgs: list[str],
) -> str:
    """
    构造发送给 Sonnet 的 prompt。

    Args:
        user_text:              候选纠正的用户消息文本
        context_assistant_msgs: 前几条 assistant 消息文本（已截断）

    Returns:
        完整 prompt 字符串
    """
    # 构造上下文块
    if context_assistant_msgs:
        context_lines = []
        for i, msg in enumerate(context_assistant_msgs, start=1):
            # 截断超长 assistant 文本
            truncated = msg[:_ASSISTANT_TEXT_MAX_CHARS]
            if len(msg) > _ASSISTANT_TEXT_MAX_CHARS:
                truncated += "…（已截断）"
            context_lines.append(f"[Assistant 回复 {i}]\n{truncated}")
        context_block = "\n\n".join(context_lines)
    else:
        context_block = "（无 assistant 上下文）"

    prompt = f"""你是代码审查助手，分析一段 Claude Code 对话，判断用户是否在纠正 AI 的行为。

--- 上下文（前几条 assistant 回复）---
{context_block}

--- 待判断的用户消息 ---
{user_text}

---
请回答以下 JSON（不要有 markdown 代码块）：
{{
  "is_correction": true/false,
  "cc_did": "...",
  "user_wanted": "...",
  "correction_type": "style|scope|approach|factual|other",
  "is_generalizable": true/false,
  "confidence": 0.0-1.0
}}

字段说明：
- is_correction: 这是不是真正的纠正行为（用户在纠正 CC 做错的事，而不是发出新指令）
- cc_did: 如果 is_correction=true，CC 做了什么（1 句话，否则留空字符串 ""）
- user_wanted: 如果 is_correction=true，用户真正想要的（1 句话，否则留空字符串 ""）
- correction_type: 纠正类型（style=风格/格式, scope=范围/边界, approach=方法/路径, factual=事实错误, other=其他）
- is_generalizable: 这个纠正能否泛化为规则（适用于未来项目中 CC 的行为）
- confidence: 你对 is_correction 判断的置信度（0.0-1.0）"""

    return prompt


def _parse_llm_response(raw_text: str) -> dict[str, Any] | None:
    """
    解析 Sonnet 返回的 JSON 文本。

    处理两种格式：
    1. 裸 JSON
    2. markdown 代码块包裹的 JSON（```json ... ```）

    Returns:
        解析后的 dict，或 None（解析失败）
    """
    text = raw_text.strip()

    # 剥离 markdown 代码块
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉第一行（```json 或 ```）和最后一行（```）
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError as e:
        print(f"[l2] JSON 解析失败: {e} | 原始文本: {raw_text[:200]}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 数据库工具
# ---------------------------------------------------------------------------

def _load_candidates(db: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    从 messages 表读取所有候选纠正消息（is_candidate_correction=1）。

    Returns:
        Row 列表，含 uuid, session_id, user_text, sequence_num 等字段
    """
    cur = db.execute(
        """
        SELECT
            m.uuid,
            m.session_id,
            m.user_text,
            m.timestamp,
            m.sequence_num,
            s.project
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE m.is_candidate_correction = 1
          AND m.user_text IS NOT NULL
        ORDER BY m.session_id, m.sequence_num ASC
        """
    )
    return cur.fetchall()


def _get_already_processed(db: sqlite3.Connection) -> set[str]:
    """
    从 corrections 表读取已处理的 user_message_uuid 集合（幂等支持）。
    """
    cur = db.execute("SELECT user_message_uuid FROM corrections")
    return {row[0] for row in cur.fetchall()}


def _get_context_assistant_msgs(
    db: sqlite3.Connection,
    session_id: str,
    before_sequence_num: int,
) -> list[dict]:
    """
    取候选消息之前最多 _MAX_CONTEXT_ASSISTANT_MSGS 条 assistant 消息。

    返回 list[dict]，每个 dict 含：
        uuid, assistant_text
    顺序：从旧到新（升序）
    """
    cur = db.execute(
        """
        SELECT uuid, assistant_text
        FROM messages
        WHERE session_id = ?
          AND type = 'assistant'
          AND sequence_num < ?
          AND assistant_text IS NOT NULL
          AND assistant_text != ''
        ORDER BY sequence_num DESC
        LIMIT ?
        """,
        (session_id, before_sequence_num, _MAX_CONTEXT_ASSISTANT_MSGS),
    )
    rows = cur.fetchall()
    # DESC 取出后翻转为升序（旧→新）
    return [{"uuid": r["uuid"], "assistant_text": r["assistant_text"]}
            for r in reversed(rows)]


def _write_correction(
    db: sqlite3.Connection,
    session_id: str,
    project: str,
    user_message_uuid: str,
    assistant_message_uuid: str,
    result: dict[str, Any],
    raw_user_text: str,
    timestamp: str,
) -> None:
    """
    将单条纠正结果写入 corrections 表。
    """
    db.execute(
        """
        INSERT INTO corrections (
            session_id, project, user_message_uuid, assistant_message_uuid,
            cc_did, user_wanted, correction_type, is_generalizable,
            confidence, raw_user_text, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            project,
            user_message_uuid,
            assistant_message_uuid,
            result.get("cc_did", ""),
            result.get("user_wanted", ""),
            result.get("correction_type", "other"),
            bool(result.get("is_generalizable", False)),
            float(result.get("confidence", 0.0)),
            raw_user_text,
            timestamp,
        ),
    )
    db.commit()


# ---------------------------------------------------------------------------
# 成本计算
# ---------------------------------------------------------------------------

def _calc_cost(input_tokens: int, output_tokens: int) -> float:
    """根据 Sonnet 4.6 定价计算成本（美元）。"""
    return (
        input_tokens  * _PRICE_INPUT_PER_M  / 1_000_000
        + output_tokens * _PRICE_OUTPUT_PER_M / 1_000_000
    )


# ---------------------------------------------------------------------------
# 单条任务处理（async）
# ---------------------------------------------------------------------------

async def _process_one(
    candidate: sqlite3.Row,
    db: sqlite3.Connection,
    budget: BudgetController,
    semaphore: asyncio.Semaphore,
    client,  # anthropic.AsyncAnthropic
) -> dict[str, Any]:
    """
    处理单条候选纠正消息。

    Returns:
        {
            'status': 'correction' | 'not_correction' | 'error',
            'uuid': str,
        }
    """
    uuid = candidate["uuid"]
    session_id = candidate["session_id"]
    user_text = candidate["user_text"]
    timestamp = candidate["timestamp"]
    sequence_num = candidate["sequence_num"]
    project = candidate["project"]

    async with semaphore:
        try:
            # 取 assistant 上下文
            context_msgs = _get_context_assistant_msgs(
                db, session_id, sequence_num
            )
            context_texts = [m["assistant_text"] for m in context_msgs]
            assistant_uuid = context_msgs[0]["uuid"] if context_msgs else ""

            # 构造 prompt
            prompt = build_prompt(user_text, context_texts)

            # 调用 LLM（带 rate limit 重试）
            t_start = time.monotonic()
            for _attempt in range(4):
                try:
                    response = await client.messages.create(
                        model=_MODEL,
                        max_tokens=_MAX_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    break
                except Exception as _e:
                    if "429" in str(_e) and _attempt < 3:
                        wait = 15 * (2 ** _attempt)
                        await asyncio.sleep(wait)
                        continue
                    raise
            duration_ms = int((time.monotonic() - t_start) * 1000)

            # 提取 token 数和成本
            usage = response.usage
            input_tokens  = usage.input_tokens
            output_tokens = usage.output_tokens
            cost = _calc_cost(input_tokens, output_tokens)

            # 记录到 llm_calls 表
            budget.record_call(
                stage="L2",
                model=_MODEL,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_ms=duration_ms,
            )

            # 解析响应
            raw_text = response.content[0].text if response.content else ""
            parsed = _parse_llm_response(raw_text)

            if parsed is None:
                print(f"[l2] 解析失败，跳过 uuid={uuid}", file=sys.stderr)
                return {"status": "error", "uuid": uuid}

            # 只写入真正的纠正（is_correction=True）
            if parsed.get("is_correction", False):
                _write_correction(
                    db=db,
                    session_id=session_id,
                    project=project,
                    user_message_uuid=uuid,
                    assistant_message_uuid=assistant_uuid,
                    result=parsed,
                    raw_user_text=user_text,
                    timestamp=timestamp,
                )
                return {"status": "correction", "uuid": uuid}
            else:
                # 非纠正：写一条占位记录确保幂等（confidence=0, cc_did=""）
                # 改为：不写入 corrections 表，但用一个内存集合跟踪已处理
                # 这样幂等性通过"重跑时再判断"实现，而非写入空记录
                # 注意：这意味着非纠正每次重跑都会重新调用 LLM
                # 权衡：准确性 > 成本（候选数量有限，通常 < 100 条）
                return {"status": "not_correction", "uuid": uuid}

        except Exception as e:
            print(f"[l2] 处理失败 uuid={uuid}: {e}", file=sys.stderr)
            return {"status": "error", "uuid": uuid}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def run_l2_corrections(
    db: sqlite3.Connection,
    budget: BudgetController,
    parallelism: int = 20,
) -> dict:
    """
    L2 纠正检测主流程。

    Args:
        db:          已初始化的 SQLite 连接
        budget:      预算控制器
        parallelism: 最大并发 LLM 调用数（默认 20）

    Returns:
        {
            'processed': int,          # 实际调用 LLM 的条数
            'corrections_found': int,  # 判断为纠正的条数
            'skipped': int,            # 跳过（已处理）的条数
            'errors': int,             # 处理失败的条数
        }
    """
    stats = {"processed": 0, "corrections_found": 0, "skipped": 0, "errors": 0}

    # 检查预算
    strategy = budget.get_strategy()
    if strategy == "stop":
        print("[l2] 预算已超 80%，停止运行", file=sys.stderr)
        return stats

    # 读取候选列表
    candidates = _load_candidates(db)
    if not candidates:
        print("[l2] 没有候选纠正消息", file=sys.stderr)
        return stats

    print(f"[l2] 共 {len(candidates)} 条候选，parallelism={parallelism}", file=sys.stderr)

    # 已处理集合（幂等：跳过 corrections 表中已有的）
    already_processed = _get_already_processed(db)

    # 过滤掉已处理
    to_process = [c for c in candidates if c["uuid"] not in already_processed]
    stats["skipped"] = len(candidates) - len(to_process)

    if not to_process:
        print("[l2] 全部已处理，跳过", file=sys.stderr)
        return stats

    # 获取 API key
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("TOWOW_ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未找到 ANTHROPIC_API_KEY 或 TOWOW_ANTHROPIC_API_KEY 环境变量"
        )

    # 导入 anthropic（延迟导入，避免 budget.py 等无 LLM 模块也必须安装）
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("请安装 anthropic 包: pip install anthropic") from e

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(parallelism)

    # 并发处理
    tasks = [
        _process_one(candidate, db, budget, semaphore, client)
        for candidate in to_process
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # 汇总结果
    for r in results:
        if isinstance(r, dict):
            status = r.get("status", "error")
            if status == "correction":
                stats["corrections_found"] += 1
                stats["processed"] += 1
            elif status == "not_correction":
                stats["processed"] += 1
            else:
                stats["errors"] += 1
        else:
            stats["errors"] += 1

    print(
        f"[l2] 完成: processed={stats['processed']}, "
        f"corrections={stats['corrections_found']}, "
        f"skipped={stats['skipped']}, "
        f"errors={stats['errors']}",
        file=sys.stderr,
    )

    return stats
