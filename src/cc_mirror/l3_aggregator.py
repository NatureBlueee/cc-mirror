"""
l3_aggregator.py — CC Mirror L3 聚合层

职责：
  - 从 DB 读取 L2 分析结果（corrections / workflow_clusters / repeated_prompts）
  - 用 Opus 生成 CLAUDE.md 规则建议
  - 用 Sonnet 生成 Skill 建议文档
  - 生成纯数字概览摘要（无 LLM）
  - 将三个文件写入 output_dir/

产出文件：
  suggested-rules.md   — CLAUDE.md 规则建议
  suggested-skills.md  — Skill 建议
  analysis-summary.md  — 数字仪表盘 + 重复提示建议
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cc_mirror.budget import BudgetController

# 模块级导入 anthropic（延迟不可 mock），不安装时为 None
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# L3a 用 Opus：生成规则是高价值任务，值得用最强模型
_MODEL_OPUS = "claude-opus-4-5-20251001"

# L3b/L3c 用 Sonnet：批量 Skill 建议，节省预算
_MODEL_SONNET = "claude-sonnet-4-6"

# Opus 定价（每百万 token，美元）
_OPUS_PRICE_INPUT_PER_M  = 15.0
_OPUS_PRICE_OUTPUT_PER_M = 75.0

# Sonnet 定价（每百万 token，美元）
_SONNET_PRICE_INPUT_PER_M  = 3.0
_SONNET_PRICE_OUTPUT_PER_M = 15.0

# corrections 置信度阈值（只处理 >= 此值的记录）
_MIN_CONFIDENCE = 0.7

# Opus 输出 token 上限（规则文档不需要太长）
_OPUS_MAX_TOKENS = 2048

# Sonnet 输出 token 上限（Skill 建议批量）
_SONNET_MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """从环境变量读取 API key（优先 ANTHROPIC_API_KEY，fallback TOWOW_ANTHROPIC_KEY）。"""
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("TOWOW_ANTHROPIC_KEY")


def _calc_opus_cost(input_tokens: int, output_tokens: int) -> float:
    """计算 Opus 调用成本（美元）。"""
    return (
        input_tokens  * _OPUS_PRICE_INPUT_PER_M  / 1_000_000
        + output_tokens * _OPUS_PRICE_OUTPUT_PER_M / 1_000_000
    )


def _calc_sonnet_cost(input_tokens: int, output_tokens: int) -> float:
    """计算 Sonnet 调用成本（美元）。"""
    return (
        input_tokens  * _SONNET_PRICE_INPUT_PER_M  / 1_000_000
        + output_tokens * _SONNET_PRICE_OUTPUT_PER_M / 1_000_000
    )


def _strip_code_fence(text: str) -> str:
    """去除 Markdown 代码块包裹（```...```），返回纯内容。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉第一行（```markdown 或 ```）和最后一行（```）
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return text


# ---------------------------------------------------------------------------
# L3a: generate_rules — 从 corrections 生成 CLAUDE.md 规则
# ---------------------------------------------------------------------------

def generate_rules(db: sqlite3.Connection, budget: "BudgetController") -> str:
    """
    从 corrections 表生成 CLAUDE.md 规则建议。

    - 只处理 confidence >= 0.7 AND is_generalizable = TRUE 的记录
    - 如果少于 3 条，仍然生成（用所有可用记录）
    - 如果 0 条，直接返回空字符串（不调用 LLM）
    - 按 correction_type 分组，一次 Opus 调用

    Returns:
        Markdown 字符串（空字符串表示无规则可生成）
    """
    # 读取符合条件的纠正记录
    try:
        cur = db.execute(
            """
            SELECT correction_type, cc_did, user_wanted, confidence, project
            FROM corrections
            WHERE confidence >= ?
              AND is_generalizable = TRUE
            ORDER BY correction_type, confidence DESC
            """,
            (_MIN_CONFIDENCE,),
        )
        rows = cur.fetchall()
    except Exception as e:
        print(f"[l3a] 读取 corrections 失败: {e}", file=sys.stderr)
        return ""

    if not rows:
        print("[l3a] 没有符合条件的 corrections（confidence >= 0.7 且 is_generalizable），跳过规则生成", file=sys.stderr)
        return ""

    print(f"[l3a] 共 {len(rows)} 条可泛化纠正记录，准备调用 Opus 生成规则", file=sys.stderr)

    # 按 correction_type 分组，构造 prompt 输入文本
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        ctype = row["correction_type"] or "other"
        if ctype not in by_type:
            by_type[ctype] = []
        by_type[ctype].append({
            "cc_did": row["cc_did"],
            "user_wanted": row["user_wanted"],
            "confidence": row["confidence"],
        })

    # 构造分组文本
    group_lines = []
    for ctype, items in by_type.items():
        group_lines.append(f"### 类型：{ctype}")
        for item in items:
            conf = item["confidence"]
            cc_did = item["cc_did"] or "（未记录）"
            user_wanted = item["user_wanted"] or "（未记录）"
            group_lines.append(f"- CC 做了：{cc_did}")
            group_lines.append(f"  用户想要：{user_wanted}（置信度 {conf:.2f}）")
        group_lines.append("")

    corrections_by_type = "\n".join(group_lines).strip()

    prompt = f"""你是一个 Claude Code 配置专家。以下是从用户与 Claude Code 的对话历史中提取的纠正行为，
请生成对应的 CLAUDE.md 规则（可以直接粘贴到 CLAUDE.md 中使用）。

纠正记录（按类型分组）：
{corrections_by_type}

输出格式（纯 Markdown，不要代码块包裹，不要解释）：
## 代码风格规则
- [规则1]
- [规则2]

## 工作流规则  
- [规则1]"""

    # 检查 anthropic 是否可用
    if anthropic is None:
        print("[l3a] anthropic 包未安装，跳过规则生成", file=sys.stderr)
        return ""

    # 检查 API key
    api_key = _get_api_key()
    if not api_key:
        print("[l3a] 未找到 ANTHROPIC_API_KEY / TOWOW_ANTHROPIC_KEY，跳过规则生成", file=sys.stderr)
        return ""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        t0 = time.time()
        response = client.messages.create(
            model=_MODEL_OPUS,
            max_tokens=_OPUS_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        duration_ms = int((time.time() - t0) * 1000)

        usage = response.usage
        input_tokens  = usage.input_tokens
        output_tokens = usage.output_tokens
        cost_usd = _calc_opus_cost(input_tokens, output_tokens)

        # 记录调用
        budget.record_call(
            stage="L3",
            model=_MODEL_OPUS,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

        raw_text = response.content[0].text if response.content else ""
        result = _strip_code_fence(raw_text)

        print(f"[l3a] Opus 调用完成（cost=${cost_usd:.4f}，{input_tokens}→{output_tokens} tokens）", file=sys.stderr)
        return result

    except Exception as e:
        print(f"[l3a] Opus API 调用失败: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# L3b: generate_skill_suggestions — 从 workflow_clusters 生成 Skill 建议
# ---------------------------------------------------------------------------

def generate_skill_suggestions(db: sqlite3.Connection, budget: "BudgetController") -> str:
    """
    从 workflow_clusters 生成 Skill 建议文档。

    - 只处理 JSON_EXTRACT(skill_suggestion, '$.is_skill_candidate') = 'true' 的聚类
    - 每个聚类生成一个 Skill 草稿（名称、触发场景、步骤）
    - 如果没有聚类，返回空字符串
    - 用 Sonnet 批量处理（不用 Opus，节省预算）

    Returns:
        Markdown 字符串（空字符串表示无 Skill 建议）
    """
    # 读取 is_skill_candidate=true 的聚类
    try:
        cur = db.execute(
            """
            SELECT
                tool_sequence_pattern,
                session_count,
                description,
                skill_suggestion
            FROM workflow_clusters
            WHERE skill_suggestion IS NOT NULL
              AND (JSON_EXTRACT(skill_suggestion, '$.is_skill_candidate') = 1 OR JSON_EXTRACT(skill_suggestion, '$.is_skill_candidate') = 'true')
            ORDER BY session_count DESC
            """
        )
        rows = cur.fetchall()
    except Exception as e:
        print(f"[l3b] 读取 workflow_clusters 失败: {e}", file=sys.stderr)
        return ""

    if not rows:
        print("[l3b] 没有 Skill 候选聚类，跳过 Skill 建议生成", file=sys.stderr)
        return ""

    print(f"[l3b] 共 {len(rows)} 个 Skill 候选聚类，准备调用 Sonnet", file=sys.stderr)

    # 构造聚类信息文本
    cluster_lines = []
    for i, row in enumerate(rows, 1):
        try:
            tool_seq = json.loads(row["tool_sequence_pattern"]) if row["tool_sequence_pattern"] else []
            skill_info = json.loads(row["skill_suggestion"]) if row["skill_suggestion"] else {}
        except json.JSONDecodeError:
            tool_seq = []
            skill_info = {}

        skill_name = skill_info.get("skill_name") or f"workflow-{i}"
        trigger = skill_info.get("skill_trigger_scenario") or "（未指定）"
        description = row["description"] or "（无描述）"
        seq_str = " → ".join(tool_seq) if tool_seq else "（无工具序列）"
        session_count = row["session_count"]

        cluster_lines.append(f"### 工作流 {i}：{skill_name}")
        cluster_lines.append(f"- 出现次数：{session_count} 个 session")
        cluster_lines.append(f"- 工具序列：{seq_str}")
        cluster_lines.append(f"- 描述：{description}")
        cluster_lines.append(f"- 建议触发场景：{trigger}")
        cluster_lines.append("")

    workflow_clusters_info = "\n".join(cluster_lines).strip()

    prompt = f"""以下是从 Claude Code 会话中识别出的工作流模式，请为每个模式生成 Skill 建议。

{workflow_clusters_info}

对每个工作流，输出（Markdown 格式，每个 Skill 之间用 --- 分隔）：
## /[skill_name]
**触发场景**：[什么时候用这个 Skill]
**工具序列**：[工具1] → [工具2] → ...
**建议步骤**：
1. ...
2. ..."""

    # 检查 anthropic 是否可用
    if anthropic is None:
        print("[l3b] anthropic 包未安装，跳过 Skill 建议生成", file=sys.stderr)
        return ""

    # 检查 API key
    api_key = _get_api_key()
    if not api_key:
        print("[l3b] 未找到 ANTHROPIC_API_KEY / TOWOW_ANTHROPIC_KEY，跳过 Skill 建议生成", file=sys.stderr)
        return ""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        t0 = time.time()
        response = client.messages.create(
            model=_MODEL_SONNET,
            max_tokens=_SONNET_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        duration_ms = int((time.time() - t0) * 1000)

        usage = response.usage
        input_tokens  = usage.input_tokens
        output_tokens = usage.output_tokens
        cost_usd = _calc_sonnet_cost(input_tokens, output_tokens)

        # 记录调用
        budget.record_call(
            stage="L3",
            model=_MODEL_SONNET,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

        raw_text = response.content[0].text if response.content else ""
        result = _strip_code_fence(raw_text)

        print(f"[l3b] Sonnet 调用完成（cost=${cost_usd:.4f}，{input_tokens}→{output_tokens} tokens）", file=sys.stderr)
        return result

    except Exception as e:
        print(f"[l3b] Sonnet API 调用失败: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# L3c: generate_summary — 纯数字概览报告（不调用 LLM）
# ---------------------------------------------------------------------------

def generate_summary(db: sqlite3.Connection, budget: "BudgetController") -> str:
    """
    生成 analysis-summary.md（概览报告）。

    - 不调用 LLM，纯数字聚合
    - 包含：sessions/messages/tool_calls 数字
    - corrections: 总候选 / LLM确认 / 纠正类型分布
    - repeated prompts: 列表（只显示 canonical_text + occurrences）
    - workflow_clusters: 聚类数 + skill_candidate 数

    Returns:
        Markdown 字符串（始终返回非空内容）
    """
    lines: list[str] = []

    # ---- 1. 基础统计 ----
    lines.append("# CC Mirror 分析摘要")
    lines.append("")

    # sessions
    try:
        cur = db.execute("SELECT COUNT(*) as cnt FROM sessions")
        row = cur.fetchone()
        session_count = row[0] if row else 0
    except Exception:
        session_count = 0

    # messages
    try:
        cur = db.execute("SELECT COUNT(*) as cnt FROM messages")
        row = cur.fetchone()
        message_count = row[0] if row else 0
    except Exception:
        message_count = 0

    # tool_calls
    try:
        cur = db.execute("SELECT COUNT(*) as cnt FROM tool_calls")
        row = cur.fetchone()
        tool_call_count = row[0] if row else 0
    except Exception:
        tool_call_count = 0

    lines.append("## 数据概览")
    lines.append("")
    lines.append("| 指标 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| Sessions | {session_count:,} |")
    lines.append(f"| Messages | {message_count:,} |")
    lines.append(f"| Tool Calls | {tool_call_count:,} |")
    lines.append("")

    # ---- 2. Corrections 统计 ----
    try:
        cur = db.execute("SELECT COUNT(*) FROM messages WHERE is_candidate_correction = 1")
        row = cur.fetchone()
        candidate_count = row[0] if row else 0
    except Exception:
        candidate_count = 0

    try:
        cur = db.execute("SELECT COUNT(*) FROM corrections")
        row = cur.fetchone()
        confirmed_count = row[0] if row else 0
    except Exception:
        confirmed_count = 0

    # 纠正类型分布
    try:
        cur = db.execute(
            """
            SELECT correction_type, COUNT(*) as cnt
            FROM corrections
            GROUP BY correction_type
            ORDER BY cnt DESC
            """
        )
        type_rows = cur.fetchall()
    except Exception:
        type_rows = []

    lines.append("## 纠正行为检测")
    lines.append("")
    lines.append(f"- 候选总数：{candidate_count:,} 条（L1 粗筛）")
    lines.append(f"- LLM 确认：{confirmed_count:,} 条（L2 精筛）")
    lines.append("")

    if type_rows:
        lines.append("**纠正类型分布：**")
        lines.append("")
        for trow in type_rows:
            ctype = trow[0] if trow[0] else "other"
            cnt = trow[1]
            lines.append(f"- {ctype}: {cnt} 条")
        lines.append("")

    # ---- 3. Repeated Prompts 列表 ----
    try:
        cur = db.execute(
            """
            SELECT canonical_text, occurrences
            FROM repeated_prompts
            ORDER BY occurrences DESC
            """
        )
        rp_rows = cur.fetchall()
    except Exception:
        rp_rows = []

    lines.append("## 重复提示词")
    lines.append("")

    if rp_rows:
        lines.append(f"共检测到 {len(rp_rows)} 个重复模式：")
        lines.append("")
        for rp in rp_rows:
            text = rp[0] or ""
            count = rp[1]
            # 截断过长文本
            display_text = text[:80] + "…" if len(text) > 80 else text
            lines.append(f"- （出现 {count} 次）`{display_text}`")
        lines.append("")
    else:
        lines.append("无重复提示词检测结果。")
        lines.append("")

    # ---- 4. Workflow Clusters 统计 ----
    try:
        cur = db.execute("SELECT COUNT(*) FROM workflow_clusters")
        row = cur.fetchone()
        cluster_count = row[0] if row else 0
    except Exception:
        cluster_count = 0

    try:
        cur = db.execute(
            """
            SELECT COUNT(*) FROM workflow_clusters
            WHERE skill_suggestion IS NOT NULL
              AND (JSON_EXTRACT(skill_suggestion, '$.is_skill_candidate') = 1 OR JSON_EXTRACT(skill_suggestion, '$.is_skill_candidate') = 'true')
            """
        )
        row = cur.fetchone()
        skill_candidate_count = row[0] if row else 0
    except Exception:
        skill_candidate_count = 0

    lines.append("## 工作流聚类")
    lines.append("")
    lines.append(f"- 聚类总数：{cluster_count} 个")
    lines.append(f"- Skill 候选：{skill_candidate_count} 个")
    lines.append("")

    # ---- 5. LLM 调用成本 ----
    try:
        cur = db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM llm_calls"
        )
        row = cur.fetchone()
        total_cost = float(row[0]) if row else 0.0
    except Exception:
        total_cost = 0.0

    try:
        cur = db.execute("SELECT COUNT(*) FROM llm_calls")
        row = cur.fetchone()
        llm_call_count = row[0] if row else 0
    except Exception:
        llm_call_count = 0

    lines.append("## 分析成本")
    lines.append("")
    lines.append(f"- LLM 调用次数：{llm_call_count} 次")
    lines.append(f"- 总成本：${total_cost:.4f} USD")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口: run_l3
# ---------------------------------------------------------------------------

def run_l3(
    db: sqlite3.Connection,
    budget: "BudgetController",
    output_dir: Path,
) -> dict:
    """
    L3 聚合层：从 DB 读取分析结果，生成建议文档。

    产出文件（写到 output_dir/）：
    - suggested-rules.md   — CLAUDE.md 规则建议
    - suggested-skills.md  — Skill 建议
    - analysis-summary.md  — 概览（数字仪表盘 + 重复提示建议）

    Args:
        db:         已初始化的 SQLite 连接
        budget:     预算控制器
        output_dir: 输出目录（不存在则自动创建）

    Returns:
        {
            "rules_generated": int,            # suggested-rules.md 里的规则条数
            "skills_suggested": int,           # suggested-skills.md 里的 Skill 数
            "repeated_prompts_addressed": int,  # repeated_prompts 条数
            "output_files": list[str],          # 生成的文件绝对路径
            "llm_calls": int,
        }
    """
    # 确保输出目录存在（不存在则自动创建）
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "rules_generated": 0,
        "skills_suggested": 0,
        "repeated_prompts_addressed": 0,
        "output_files": [],
        "llm_calls": 0,
    }

    # 检查预算策略
    strategy = budget.get_strategy()
    if strategy == "stop":
        print("[l3] 预算已超 80%，跳过 L3a/L3b，只生成 summary", file=sys.stderr)

    # ---- L3a: 规则建议（预算允许时才调用 LLM）----
    rules_md = ""
    if strategy != "stop":
        rules_md = generate_rules(db, budget)

    # 计算规则条数：统计以 "- " 开头的行
    if rules_md:
        rule_lines = [line for line in rules_md.splitlines() if line.strip().startswith("- ")]
        stats["rules_generated"] = len(rule_lines)

    # 写入文件（即使为空也写，方便 idempotent 重跑）
    rules_path = output_dir / "suggested-rules.md"
    header = "# 建议的 CLAUDE.md 规则\n\n_由 CC Mirror L3 自动生成_\n\n"
    if not rules_md:
        rules_content = header + "_暂无可泛化规则（需要更多纠正数据）。_\n"
    else:
        rules_content = header + rules_md + "\n"

    try:
        rules_path.write_text(rules_content, encoding="utf-8")
        stats["output_files"].append(str(rules_path.absolute()))
        print(f"[l3] 规则文件已写入：{rules_path}（{stats['rules_generated']} 条规则）", file=sys.stderr)
    except Exception as e:
        print(f"[l3] 写入 {rules_path} 失败: {e}", file=sys.stderr)

    # ---- L3b: Skill 建议（预算允许时才调用 LLM）----
    skills_md = ""
    if strategy != "stop":
        skills_md = generate_skill_suggestions(db, budget)

    # 计算 Skill 数：统计以 "## /" 开头的行
    if skills_md:
        skill_sections = [line for line in skills_md.splitlines() if line.strip().startswith("## /")]
        stats["skills_suggested"] = len(skill_sections)

    # 写入文件
    skills_path = output_dir / "suggested-skills.md"
    skill_header = "# 建议的 Claude Code Skills\n\n_由 CC Mirror L3 自动生成_\n\n"
    if not skills_md:
        skills_content = skill_header + "_暂无 Skill 建议（需要更多工作流聚类数据）。_\n"
    else:
        skills_content = skill_header + skills_md + "\n"

    try:
        skills_path.write_text(skills_content, encoding="utf-8")
        stats["output_files"].append(str(skills_path.absolute()))
        print(f"[l3] Skill 文件已写入：{skills_path}（{stats['skills_suggested']} 个 Skill）", file=sys.stderr)
    except Exception as e:
        print(f"[l3] 写入 {skills_path} 失败: {e}", file=sys.stderr)

    # ---- L3c: 摘要报告（无 LLM，始终生成）----
    summary_md = generate_summary(db, budget)

    summary_path = output_dir / "analysis-summary.md"
    try:
        summary_path.write_text(summary_md, encoding="utf-8")
        stats["output_files"].append(str(summary_path.absolute()))
        print(f"[l3] 摘要文件已写入：{summary_path}", file=sys.stderr)
    except Exception as e:
        print(f"[l3] 写入 {summary_path} 失败: {e}", file=sys.stderr)

    # ---- 统计 repeated_prompts ----
    try:
        cur = db.execute("SELECT COUNT(*) FROM repeated_prompts")
        row = cur.fetchone()
        stats["repeated_prompts_addressed"] = row[0] if row else 0
    except Exception:
        pass

    # ---- 统计本次 L3 阶段 LLM 调用次数 ----
    try:
        cur = db.execute("SELECT COUNT(*) FROM llm_calls WHERE stage = 'L3'")
        row = cur.fetchone()
        stats["llm_calls"] = row[0] if row else 0
    except Exception:
        pass

    print(
        f"[l3] 完成：rules={stats['rules_generated']}, "
        f"skills={stats['skills_suggested']}, "
        f"llm_calls={stats['llm_calls']}, "
        f"files={len(stats['output_files'])}",
        file=sys.stderr,
    )

    return stats
