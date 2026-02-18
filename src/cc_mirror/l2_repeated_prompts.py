"""
l2_repeated_prompts.py — CC Mirror L2 重复提示语义分析

职责：
  - 读取 L1 已写入的 repeated_prompts 表（精确重复检测结果）
  - 取出现次数 >= 3 的 top 20 条
  - 批量调用 Sonnet 分析每条重复提示的意图 + 改进建议
  - 结果写回 repeated_prompts.analysis_json 字段

L2 阶段：一次 Sonnet 批量调用（所有 repeated_prompts 合并到一个 prompt）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cc_mirror.budget import BudgetController

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 分析用的 Sonnet 模型
_MODEL = "claude-sonnet-4-6"

# Sonnet 定价（每百万 token，美元）
_PRICE_INPUT_PER_M = 3.0
_PRICE_OUTPUT_PER_M = 15.0

# 批量分析：最多取多少条 repeated_prompts
_MAX_PROMPTS_TO_ANALYZE = 20

# 最少出现次数（L1 已过滤 >= 3，这里再做一次保险）
_MIN_OCCURRENCES = 3


# ---------------------------------------------------------------------------
# DB 工具
# ---------------------------------------------------------------------------

def _ensure_analysis_column(db: sqlite3.Connection) -> None:
    """
    确保 repeated_prompts 表有 analysis_json 字段。
    如果字段已存在，忽略 OperationalError。
    """
    try:
        db.execute("ALTER TABLE repeated_prompts ADD COLUMN analysis_json TEXT")
        db.commit()
    except sqlite3.OperationalError:
        # 列已存在，正常情况
        pass


def _fetch_top_repeated_prompts(db: sqlite3.Connection) -> list[dict]:
    """
    从 repeated_prompts 表读取出现次数最多的前 N 条。

    Returns:
        list of dict，每个 dict 包含 id, canonical_text, occurrences
    """
    cur = db.execute(
        """
        SELECT id, canonical_text, occurrences
        FROM repeated_prompts
        WHERE occurrences >= ?
        ORDER BY occurrences DESC
        LIMIT ?
        """,
        (_MIN_OCCURRENCES, _MAX_PROMPTS_TO_ANALYZE),
    )
    rows = cur.fetchall()
    return [{"id": r["id"], "text": r["canonical_text"], "occurrences": r["occurrences"]} for r in rows]


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """从环境变量读取 API key（优先 ANTHROPIC_API_KEY，fallback TOWOW_ANTHROPIC_API_KEY）。"""
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("TOWOW_ANTHROPIC_API_KEY")


def _analyze_repeated_prompts_batch(
    prompts: list[dict],
    budget: "BudgetController",
) -> list[dict] | None:
    """
    批量调用 Sonnet 分析所有重复提示（一次 API 调用）。

    Args:
        prompts: list of {"id": int, "text": str, "occurrences": int}
        budget:  预算控制器

    Returns:
        Sonnet 返回的分析 list，或 None（预算超限 / API 失败）
    """
    strategy = budget.get_strategy()
    if strategy == "stop":
        print("[l2_repeated_prompts] 预算耗尽，跳过分析", file=sys.stderr)
        return None

    api_key = _get_api_key()
    if not api_key:
        print(
            "[l2_repeated_prompts] 未找到 ANTHROPIC_API_KEY / TOWOW_ANTHROPIC_API_KEY",
            file=sys.stderr,
        )
        return None

    # 构造编号列表
    numbered_lines = []
    for i, p in enumerate(prompts, 1):
        # 截断过长文本，避免 token 浪费
        text_preview = p["text"][:150]
        numbered_lines.append(f'{i}. （出现{p["occurrences"]}次）"{text_preview}"')
    numbered_list = "\n".join(numbered_lines)

    prompt = f"""以下是用户在 Claude Code 中反复输入的提示词（按出现次数降序）：

{numbered_list}

请分析每条重复提示，回答：
1. 用户为什么会反复输入这个？（habit/missing-context/workflow/unclear-default）
2. 这个提示可以写到哪里来避免重复？（CLAUDE.md/skill/alias/none）

回答 JSON array（无 markdown 代码块），数组长度必须与上面条数相同：
[
  {{
    "text": "...",
    "reason_type": "habit|missing-context|workflow|unclear-default",
    "suggestion_location": "CLAUDE.md|skill|alias|none",
    "suggestion_content": "具体建议写什么"
  }},
  ...
]"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        t0 = time.time()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        duration_ms = int((time.time() - t0) * 1000)

        # token 统计 + 记录
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cost_usd = (
            input_tokens * _PRICE_INPUT_PER_M / 1_000_000
            + output_tokens * _PRICE_OUTPUT_PER_M / 1_000_000
        )

        budget.record_call(
            stage="L2",
            model=_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

        # 解析响应
        raw_text = response.content[0].text.strip()
        # 清理可能的 markdown 代码块标记
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:-1]) if len(lines) > 2 else raw_text

        results = json.loads(raw_text)
        if not isinstance(results, list):
            print(
                f"[l2_repeated_prompts] Sonnet 返回格式非 list: {type(results)}",
                file=sys.stderr,
            )
            return None

        return results

    except json.JSONDecodeError as e:
        print(f"[l2_repeated_prompts] Sonnet 响应 JSON 解析失败: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[l2_repeated_prompts] Sonnet API 调用失败: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_l2_repeated_prompts(
    db: sqlite3.Connection,
    budget: "BudgetController",
) -> dict:
    """
    对 L1 检测到的重复提示做语义分析，将结果写回 DB。

    流程：
      1. 确保 repeated_prompts.analysis_json 字段存在
      2. 读取 top 20 条出现次数 >= 3 的重复提示
      3. 一次 Sonnet 批量调用分析全部
      4. 按顺序将分析结果写回对应行的 analysis_json 字段

    Args:
        db:     SQLite 连接
        budget: 预算控制器

    Returns:
        {"analyzed": int, "suggestions_generated": int}
    """
    stats = {"analyzed": 0, "suggestions_generated": 0}

    # ---- 1. 确保字段存在 ----
    _ensure_analysis_column(db)

    # ---- 2. 读取数据 ----
    try:
        prompts = _fetch_top_repeated_prompts(db)
    except Exception as e:
        print(f"[l2_repeated_prompts] 读取 repeated_prompts 失败: {e}", file=sys.stderr)
        return stats

    if not prompts:
        print("[l2_repeated_prompts] 没有符合条件的重复提示，跳过", file=sys.stderr)
        return stats

    print(
        f"[l2_repeated_prompts] 分析 {len(prompts)} 条重复提示",
        file=sys.stderr,
    )

    # ---- 3. 批量 Sonnet 分析 ----
    analyses = _analyze_repeated_prompts_batch(prompts, budget)

    if analyses is None:
        print("[l2_repeated_prompts] 分析失败或预算耗尽，跳过写入", file=sys.stderr)
        return stats

    # ---- 4. 写回 DB ----
    suggestions_generated = 0
    analyzed = 0

    # 按顺序配对（Sonnet 应按顺序返回，数量可能不等，取 min）
    paired_count = min(len(prompts), len(analyses))

    for i in range(paired_count):
        prompt_row = prompts[i]
        analysis_item = analyses[i]

        analysis_json_str = json.dumps(analysis_item, ensure_ascii=False)

        try:
            db.execute(
                """
                UPDATE repeated_prompts
                SET analysis_json = ?
                WHERE id = ?
                """,
                (analysis_json_str, prompt_row["id"]),
            )
            analyzed += 1

            # 统计有实质建议的条数（location != none）
            location = analysis_item.get("suggestion_location", "none")
            if location and location != "none":
                suggestions_generated += 1

        except Exception as e:
            print(
                f"[l2_repeated_prompts] 写入 id={prompt_row['id']} 失败: {e}",
                file=sys.stderr,
            )

    try:
        db.commit()
    except Exception as e:
        print(f"[l2_repeated_prompts] commit 失败: {e}", file=sys.stderr)
        db.rollback()

    stats["analyzed"] = analyzed
    stats["suggestions_generated"] = suggestions_generated

    print(
        f"[l2_repeated_prompts] 完成：{analyzed} 条已分析，{suggestions_generated} 条有建议",
        file=sys.stderr,
    )
    return stats
