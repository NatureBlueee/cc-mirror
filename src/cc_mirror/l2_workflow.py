"""
l2_workflow.py — CC Mirror L2 工作流聚类分析

职责：
  - 从 tool_calls 表按 session 提取工具名序列
  - 计算序列间 LCS（最长公共子序列）相似度矩阵
  - 层次聚类（scipy 可用时用 scipy，否则 fallback 贪心聚类）
  - 每个聚类调用 Sonnet 分析任务类型 + Skill 建议
  - 写入 workflow_clusters 表

L2 阶段：有 LLM 调用（记录到 llm_calls 表）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cc_mirror.budget import BudgetController

from cc_mirror.llm import call_llm

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 分析用的模型（传给 call_llm 的 model 参数）
_MODEL = "sonnet"

# 每个聚类最多取几个 session 的序列展示给 Sonnet
_MAX_SEQUENCES_PER_CLUSTER = 5

# 序列最短长度（太短没分析价值）
_MIN_SEQUENCE_LEN = 3

# 聚类最少 session 数（单 session 聚类不写入）
_MIN_CLUSTER_SESSIONS = 2


# ---------------------------------------------------------------------------
# LCS 相似度（纯 Python，无外部依赖）
# ---------------------------------------------------------------------------

def lcs_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """
    计算两个工具名序列之间的 LCS（最长公共子序列）相似度。

    相似度公式：2 * lcs_len / (len_a + len_b)
    范围：[0.0, 1.0]
    空序列 → 0.0

    Args:
        seq_a: 工具名列表，如 ["Read", "Bash", "Edit"]
        seq_b: 工具名列表，如 ["Read", "Edit", "Bash"]

    Returns:
        相似度浮点数，范围 [0.0, 1.0]
    """
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0 or len_b == 0:
        return 0.0

    # 标准 LCS 动态规划
    # dp[i][j] = LCS length of seq_a[:i] and seq_b[:j]
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[len_a][len_b]
    return 2.0 * lcs_len / (len_a + len_b)


# ---------------------------------------------------------------------------
# 聚类逻辑
# ---------------------------------------------------------------------------

def _build_similarity_matrix(sequences: list[list[str]]) -> list[list[float]]:
    """
    计算 n×n 相似度矩阵。

    Args:
        sequences: 工具名序列列表

    Returns:
        n×n 相似度矩阵（列表的列表）
    """
    n = len(sequences)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = lcs_similarity(sequences[i], sequences[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def _cluster_scipy(
    sequences: list[list[str]],
    threshold: float,
) -> list[list[int]]:
    """
    用 scipy 层次聚类分组 session 索引。

    Args:
        sequences:  工具名序列列表（与 session_ids 一一对应）
        threshold:  相似度阈值（0-1），高于此值才归入同一类

    Returns:
        聚类列表，每个元素是 session 索引组成的列表
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    sim_matrix = _build_similarity_matrix(sequences)
    # 转距离矩阵（距离 = 1 - 相似度）
    n = len(sequences)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = 1.0 - sim_matrix[i][j]

    dist_array = np.array(dist_matrix)
    # squareform 将方形矩阵转为压缩向量形式
    condensed = squareform(dist_array, checks=False)

    # 层次聚类（average linkage）
    Z = linkage(condensed, method="average")

    # 用距离阈值切割：1 - threshold 即最大允许距离
    labels = fcluster(Z, t=(1.0 - threshold), criterion="distance")

    # 按 label 分组
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    return list(groups.values())


def _cluster_greedy(
    sequences: list[list[str]],
    threshold: float,
) -> list[list[int]]:
    """
    贪心聚类（fallback，不依赖 scipy）。

    遍历序列，如果与某个已有聚类的代表序列相似度 > threshold，
    则加入该聚类；否则新建聚类。

    代表序列 = 聚类中第一条（固定，不更新）。

    Args:
        sequences:  工具名序列列表
        threshold:  相似度阈值

    Returns:
        聚类列表，每个元素是 session 索引组成的列表
    """
    # clusters: [(代表索引, [成员索引列表])]
    clusters: list[tuple[int, list[int]]] = []

    for idx, seq in enumerate(sequences):
        assigned = False
        for rep_idx, members in clusters:
            sim = lcs_similarity(sequences[rep_idx], seq)
            if sim >= threshold:
                members.append(idx)
                assigned = True
                break
        if not assigned:
            clusters.append((idx, [idx]))

    return [members for _, members in clusters]


def _do_cluster(
    sequences: list[list[str]],
    threshold: float,
) -> list[list[int]]:
    """
    聚类入口：优先 scipy，不可用时 fallback 贪心。

    Returns:
        聚类列表，每个元素是 session 索引组成的列表
    """
    try:
        return _cluster_scipy(sequences, threshold)
    except ImportError:
        print("[l2_workflow] scipy 不可用，使用贪心聚类", file=sys.stderr)
        return _cluster_greedy(sequences, threshold)
    except Exception as e:
        print(f"[l2_workflow] scipy 聚类失败 ({e})，fallback 贪心聚类", file=sys.stderr)
        return _cluster_greedy(sequences, threshold)


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def _analyze_cluster_with_sonnet(
    cluster_sequences: list[list[str]],
    budget: "BudgetController",
) -> dict | None:
    """
    调用 LLM 分析一个工作流聚类的任务类型和 Skill 建议。

    Args:
        cluster_sequences: 聚类内若干 session 的工具序列（最多 5 个）
        budget:            预算控制器（记录调用 + 检查策略）

    Returns:
        分析结果 dict，或 None（预算超限 / API 失败）
    """
    strategy = budget.get_strategy()
    if strategy == "stop":
        print("[l2_workflow] 预算耗尽，跳过 Sonnet 分析", file=sys.stderr)
        return None

    # 构造 sequences 展示文本
    seq_lines = []
    for i, seq in enumerate(cluster_sequences[:_MAX_SEQUENCES_PER_CLUSTER], 1):
        seq_lines.append(f"  会话{i}: {' → '.join(seq)}")
    sequences_text = "\n".join(seq_lines)

    prompt = f"""以下是多个 Claude Code 会话中重复出现的工具调用序列模式：

{sequences_text}

请分析：
1. 这个工作流在做什么任务？（1-2 句话）
2. 这个工作流适合沉淀为 Claude Code Skill 吗？（是/否 + 理由）
3. 如果适合，建议 Skill 名称和触发场景？

回答 JSON（无 markdown 代码块）：
{{
  "description": "...",
  "is_skill_candidate": true/false,
  "skill_name": "...",
  "skill_trigger_scenario": "..."
}}"""

    try:
        t0 = time.time()
        raw_text, cost_usd = asyncio.run(call_llm(prompt, model=_MODEL))
        duration_ms = int((time.time() - t0) * 1000)

        # 记录调用（token 数由 CLI 路径省略，填 0 作为占位）
        budget.record_call(
            stage="L2",
            model=_MODEL,
            input_tokens=0,
            output_tokens=0,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

        # 解析响应
        raw_text = raw_text.strip()
        # 清理可能的 markdown 代码块标记
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:-1]) if len(lines) > 2 else raw_text

        result = json.loads(raw_text)
        return result

    except json.JSONDecodeError as e:
        print(f"[l2_workflow] LLM 响应 JSON 解析失败: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[l2_workflow] LLM 调用失败: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_l2_workflow(
    db: sqlite3.Connection,
    budget: "BudgetController",
    similarity_threshold: float = 0.6,
) -> dict:
    """
    分析工具调用序列，发现用户的工作流模式，写入 workflow_clusters 表。

    流程：
      1. 从 tool_calls 表按 session 提取工具名序列
      2. 过滤掉长度 < _MIN_SEQUENCE_LEN 的序列
      3. LCS 相似度矩阵 + 层次聚类
      4. 每个 >= 2 session 的聚类调用 Sonnet 分析
      5. 写入 workflow_clusters 表

    Args:
        db:                  SQLite 连接
        budget:              预算控制器
        similarity_threshold: 聚类相似度阈值，默认 0.6

    Returns:
        {"clusters_found": int, "sessions_analyzed": int, "llm_calls": int}
    """
    stats = {"clusters_found": 0, "sessions_analyzed": 0, "llm_calls": 0}

    # ---- 1. 从 DB 提取每个 session 的工具名序列 ----
    try:
        cur = db.execute(
            """
            SELECT session_id, tool_name
            FROM tool_calls
            ORDER BY session_id, sequence_num ASC
            """
        )
        rows = cur.fetchall()
    except Exception as e:
        print(f"[l2_workflow] 读取 tool_calls 失败: {e}", file=sys.stderr)
        return stats

    # 按 session_id 分组，保留顺序
    from collections import defaultdict
    session_tools: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        session_tools[row["session_id"]].append(row["tool_name"])

    # ---- 2. 过滤短序列 ----
    valid_sessions = {
        sid: tools
        for sid, tools in session_tools.items()
        if len(tools) >= _MIN_SEQUENCE_LEN
    }

    if not valid_sessions:
        print("[l2_workflow] 没有有效工具序列（全部过短），跳过", file=sys.stderr)
        return stats

    session_ids = list(valid_sessions.keys())
    sequences = [valid_sessions[sid] for sid in session_ids]
    stats["sessions_analyzed"] = len(session_ids)

    print(
        f"[l2_workflow] 分析 {len(session_ids)} 个 session 的工具序列",
        file=sys.stderr,
    )

    # ---- 3. 聚类 ----
    if len(sequences) == 1:
        # 只有 1 个 session，聚类无意义
        print("[l2_workflow] 只有 1 个 session，跳过聚类", file=sys.stderr)
        return stats

    cluster_groups = _do_cluster(sequences, similarity_threshold)

    print(
        f"[l2_workflow] 聚类完成，得到 {len(cluster_groups)} 个聚类",
        file=sys.stderr,
    )

    # ---- 4. 分析 + 写入 ----
    written = 0
    llm_calls = 0

    for group_indices in cluster_groups:
        # 过滤单 session 聚类
        if len(group_indices) < _MIN_CLUSTER_SESSIONS:
            continue

        group_session_ids = [session_ids[i] for i in group_indices]
        group_sequences = [sequences[i] for i in group_indices]

        # 代表序列：取最长的那个
        representative_seq = max(group_sequences, key=len)

        # 调用 Sonnet 分析
        analysis = _analyze_cluster_with_sonnet(group_sequences, budget)
        if analysis is not None:
            llm_calls += 1

        description = analysis.get("description") if analysis else None
        skill_suggestion = (
            {
                "is_skill_candidate": analysis.get("is_skill_candidate"),
                "skill_name": analysis.get("skill_name"),
                "skill_trigger_scenario": analysis.get("skill_trigger_scenario"),
            }
            if analysis
            else None
        )

        # 写入 workflow_clusters
        try:
            db.execute(
                """
                INSERT INTO workflow_clusters (
                    tool_sequence_pattern,
                    session_ids,
                    session_count,
                    similarity_threshold,
                    description,
                    skill_suggestion
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    json.dumps(representative_seq, ensure_ascii=False),
                    json.dumps(group_session_ids, ensure_ascii=False),
                    len(group_session_ids),
                    similarity_threshold,
                    description,
                    json.dumps(skill_suggestion, ensure_ascii=False) if skill_suggestion else None,
                ),
            )
            db.commit()
            written += 1
        except Exception as e:
            print(f"[l2_workflow] 写入 workflow_clusters 失败: {e}", file=sys.stderr)
            db.rollback()

    stats["clusters_found"] = written
    stats["llm_calls"] = llm_calls

    print(
        f"[l2_workflow] 完成：{written} 个聚类写入，{llm_calls} 次 LLM 调用",
        file=sys.stderr,
    )
    return stats
