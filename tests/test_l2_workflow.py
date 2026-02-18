"""
test_l2_workflow.py — L2 工作流聚类 + 重复提示分析的最小测试

测试策略：
  - lcs_similarity：确定性 case，不需要 mock
  - run_l2_workflow / run_l2_repeated_prompts：smoke test，mock Anthropic API
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保 src 在路径中（不依赖 pip install）
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cc_mirror.db import init_db
from cc_mirror.l2_workflow import lcs_similarity, run_l2_workflow
from cc_mirror.l2_repeated_prompts import run_l2_repeated_prompts


# ---------------------------------------------------------------------------
# 测试辅助：内存数据库 + BudgetController
# ---------------------------------------------------------------------------

def _make_in_memory_db() -> sqlite3.Connection:
    """创建内存 SQLite 数据库，执行完整 schema。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = init_db(db_path)
    return db


def _make_budget(db: sqlite3.Connection, budget_usd: float = 10.0):
    """创建 BudgetController 实例。"""
    from cc_mirror.budget import BudgetController
    return BudgetController(budget_usd=budget_usd, db=db)


def _insert_session(db: sqlite3.Connection, session_id: str) -> None:
    """向 sessions 表插入最小 session 记录。"""
    db.execute(
        """
        INSERT OR IGNORE INTO sessions
            (id, project, start_time, message_count, jsonl_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, "test-project", "2024-01-01T00:00:00Z", 0, "/tmp/test.jsonl"),
    )
    db.commit()


def _insert_tool_calls(
    db: sqlite3.Connection,
    session_id: str,
    tools: list[str],
) -> None:
    """向 tool_calls 表插入一组工具调用（按顺序）。"""
    # 需要先有对应 session
    _insert_session(db, session_id)
    # 需要一个 dummy message
    msg_uuid = f"msg-{session_id}"
    db.execute(
        """
        INSERT OR IGNORE INTO messages
            (uuid, session_id, type, timestamp, sequence_num)
        VALUES (?, ?, ?, ?, ?)
        """,
        (msg_uuid, session_id, "assistant", "2024-01-01T00:00:00Z", 0),
    )
    for i, tool_name in enumerate(tools):
        db.execute(
            """
            INSERT INTO tool_calls
                (id, session_id, message_uuid, tool_name, timestamp, sequence_num)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (f"{session_id}_tc_{i}", session_id, msg_uuid, tool_name,
             "2024-01-01T00:00:00Z", i),
        )
    db.commit()


# ---------------------------------------------------------------------------
# lcs_similarity 测试
# ---------------------------------------------------------------------------

class TestLcsSimilarity(unittest.TestCase):
    """验证 LCS 相似度函数的边界 case 和确定性结果。"""

    def test_identical_sequences(self):
        """完全相同的序列，相似度 = 1.0。"""
        seq = ["Read", "Bash", "Edit", "Bash"]
        self.assertAlmostEqual(lcs_similarity(seq, seq), 1.0)

    def test_empty_sequence(self):
        """含空序列，返回 0.0。"""
        self.assertAlmostEqual(lcs_similarity([], ["Read"]), 0.0)
        self.assertAlmostEqual(lcs_similarity(["Read"], []), 0.0)
        self.assertAlmostEqual(lcs_similarity([], []), 0.0)

    def test_no_common_elements(self):
        """完全不同的序列，相似度 = 0.0。"""
        seq_a = ["Read", "Read"]
        seq_b = ["Bash", "Bash"]
        self.assertAlmostEqual(lcs_similarity(seq_a, seq_b), 0.0)

    def test_partial_overlap(self):
        """部分重叠。
        seq_a = ["Read", "Bash", "Edit"]  (len=3)
        seq_b = ["Read", "Edit"]           (len=2)
        LCS = ["Read", "Edit"]，长度 2
        相似度 = 2*2/(3+2) = 4/5 = 0.8
        """
        seq_a = ["Read", "Bash", "Edit"]
        seq_b = ["Read", "Edit"]
        expected = 2 * 2 / (3 + 2)
        self.assertAlmostEqual(lcs_similarity(seq_a, seq_b), expected, places=6)

    def test_interleaved_sequence(self):
        """子序列（不是子串）测试。
        seq_a = ["A", "B", "C", "D"]
        seq_b = ["A", "C"]
        LCS = ["A", "C"]，长度 2
        相似度 = 2*2/(4+2) = 4/6 ≈ 0.6667
        """
        seq_a = ["A", "B", "C", "D"]
        seq_b = ["A", "C"]
        expected = 2 * 2 / (4 + 2)
        self.assertAlmostEqual(lcs_similarity(seq_a, seq_b), expected, places=6)

    def test_single_element_match(self):
        """单元素相同。
        seq_a = ["Read"] (len=1), seq_b = ["Read"] (len=1)
        LCS = 1，相似度 = 2*1/(1+1) = 1.0
        """
        self.assertAlmostEqual(lcs_similarity(["Read"], ["Read"]), 1.0)

    def test_single_element_no_match(self):
        """单元素不同。相似度 = 0.0。"""
        self.assertAlmostEqual(lcs_similarity(["Read"], ["Bash"]), 0.0)

    def test_result_in_range(self):
        """相似度始终在 [0, 1] 内。"""
        cases = [
            (["Read", "Bash"], ["Edit", "Bash", "Write"]),
            (["A", "B", "C"], ["C", "B", "A"]),
            (["X"] * 10, ["X"] * 5 + ["Y"] * 5),
        ]
        for seq_a, seq_b in cases:
            sim = lcs_similarity(seq_a, seq_b)
            self.assertGreaterEqual(sim, 0.0, f"seq_a={seq_a}, seq_b={seq_b}")
            self.assertLessEqual(sim, 1.0, f"seq_a={seq_a}, seq_b={seq_b}")


# ---------------------------------------------------------------------------
# run_l2_workflow smoke test（mock Anthropic API）
# ---------------------------------------------------------------------------

class TestRunL2WorkflowSmoke(unittest.TestCase):
    """smoke test：不实际调用 Anthropic API，验证函数不崩溃且写入正确。"""

    def _make_mock_anthropic_response(self, analysis_text: str):
        """构造一个 mock Anthropic Messages response 对象。"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=analysis_text)]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        return mock_response

    def test_no_tool_calls(self):
        """空数据库：没有 tool_calls，应该返回全零统计。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        stats = run_l2_workflow(db, budget)

        self.assertEqual(stats["clusters_found"], 0)
        self.assertEqual(stats["sessions_analyzed"], 0)
        self.assertEqual(stats["llm_calls"], 0)
        db.close()

    def test_single_session_skipped(self):
        """只有 1 个 session：序列有效但无法聚类，不写入。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        _insert_tool_calls(db, "session-001", ["Read", "Bash", "Edit", "Bash"])

        stats = run_l2_workflow(db, budget)

        # 1 个 session 无法形成 >= 2 session 的聚类
        self.assertEqual(stats["clusters_found"], 0)
        db.close()

    def test_two_similar_sessions_form_cluster(self):
        """两个相似序列应该被聚成 1 个聚类，并调用 Sonnet。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        # 两个几乎相同的工具序列
        _insert_tool_calls(db, "session-A", ["Read", "Bash", "Edit", "Bash", "Write"])
        _insert_tool_calls(db, "session-B", ["Read", "Bash", "Edit", "Write"])

        # mock Anthropic API
        analysis_json = json.dumps({
            "description": "读取文件、执行命令、编辑文件的典型工作流",
            "is_skill_candidate": True,
            "skill_name": "read-edit-workflow",
            "skill_trigger_scenario": "修改代码后验证结果",
        })
        mock_response = self._make_mock_anthropic_response(analysis_json)

        with patch("anthropic.Anthropic") as MockAnthropicClass:
            mock_client = MagicMock()
            MockAnthropicClass.return_value = mock_client
            mock_client.messages.create.return_value = mock_response

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
                stats = run_l2_workflow(db, budget, similarity_threshold=0.5)

        # 应该找到 1 个聚类
        self.assertEqual(stats["clusters_found"], 1)
        self.assertEqual(stats["llm_calls"], 1)

        # 验证写入了 workflow_clusters
        cur = db.execute("SELECT * FROM workflow_clusters")
        clusters = cur.fetchall()
        self.assertEqual(len(clusters), 1)
        self.assertIn("session-A", clusters[0]["session_ids"])
        self.assertIn("session-B", clusters[0]["session_ids"])
        self.assertIsNotNone(clusters[0]["description"])

        db.close()

    def test_dissimilar_sessions_no_cluster(self):
        """两个完全不同的序列，相似度低，不应形成聚类（单 session 聚类被过滤）。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        # 完全不同的工具序列
        _insert_tool_calls(db, "session-X", ["Read", "Read", "Read", "Read"])
        _insert_tool_calls(db, "session-Y", ["Bash", "Bash", "Bash", "Bash"])

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            stats = run_l2_workflow(db, budget, similarity_threshold=0.8)

        # 相似度 = 0，阈值 0.8，两个 session 进不了同一聚类
        # 每个聚类只有 1 个 session → 全部被过滤
        self.assertEqual(stats["clusters_found"], 0)
        db.close()

    def test_short_sequences_skipped(self):
        """长度 < 3 的序列不应被分析。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        # 只有 2 个工具调用，低于最小长度 3
        _insert_tool_calls(db, "session-short-1", ["Read", "Bash"])
        _insert_tool_calls(db, "session-short-2", ["Read", "Bash"])

        stats = run_l2_workflow(db, budget)

        self.assertEqual(stats["sessions_analyzed"], 0)
        self.assertEqual(stats["clusters_found"], 0)
        db.close()

    def test_stop_budget_skips_llm(self):
        """预算耗尽时，聚类逻辑还是运行，但 LLM 分析被跳过（description=None）。"""
        db = _make_in_memory_db()
        # 极小预算，模拟 stop 状态
        budget = _make_budget(db, budget_usd=0.001)

        # 人工把 spent 推高：插入假的 llm_calls 记录
        db.execute(
            """
            INSERT INTO llm_calls (stage, model, prompt_hash, input_tokens, output_tokens, cost_usd, timestamp)
            VALUES ('L2', 'test', 'abc123', 100, 50, 0.001, '2024-01-01T00:00:00Z')
            """
        )
        db.commit()

        _insert_tool_calls(db, "session-P", ["Read", "Edit", "Bash", "Write", "Bash"])
        _insert_tool_calls(db, "session-Q", ["Read", "Edit", "Write", "Bash"])

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            stats = run_l2_workflow(db, budget, similarity_threshold=0.5)

        # LLM 被跳过，但聚类可能仍然写入（description=None）
        self.assertEqual(stats["llm_calls"], 0)
        db.close()


# ---------------------------------------------------------------------------
# run_l2_repeated_prompts smoke test（mock Anthropic API）
# ---------------------------------------------------------------------------

class TestRunL2RepeatedPromptsSmoke(unittest.TestCase):
    """smoke test：验证重复提示分析流程，mock Anthropic API。"""

    def _insert_repeated_prompt(
        self,
        db: sqlite3.Connection,
        text: str,
        occurrences: int,
    ) -> None:
        """向 repeated_prompts 表插入测试数据。"""
        db.execute(
            """
            INSERT INTO repeated_prompts
                (canonical_text, occurrences, session_ids, project_ids, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                occurrences,
                '["session-1", "session-2", "session-3"]',
                '["project-a"]',
                "2024-01-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
            ),
        )
        db.commit()

    def _make_mock_response(self, results: list[dict]):
        """构造 mock Anthropic Messages response。"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(results, ensure_ascii=False))]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 80
        return mock_response

    def test_no_repeated_prompts(self):
        """空 repeated_prompts 表，应该返回全零统计。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        stats = run_l2_repeated_prompts(db, budget)

        self.assertEqual(stats["analyzed"], 0)
        self.assertEqual(stats["suggestions_generated"], 0)
        db.close()

    def test_analysis_column_created(self):
        """analysis_json 字段应该被自动创建。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        # 先确保字段不存在（通过直接查 schema）
        # 运行后字段应存在
        run_l2_repeated_prompts(db, budget)

        # 验证字段可以被查询（不抛异常）
        try:
            db.execute("SELECT analysis_json FROM repeated_prompts LIMIT 1")
        except sqlite3.OperationalError as e:
            self.fail(f"analysis_json 字段应该存在，但出现错误: {e}")
        db.close()

    def test_analysis_written_to_db(self):
        """有数据时，分析结果应该写入 analysis_json 字段。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        # 插入两条重复提示
        self._insert_repeated_prompt(db, "请用中文回复", 5)
        self._insert_repeated_prompt(db, "只改这一行，不要改其他的", 4)

        # mock Sonnet 返回
        mock_results = [
            {
                "text": "请用中文回复",
                "reason_type": "missing-context",
                "suggestion_location": "CLAUDE.md",
                "suggestion_content": "在 CLAUDE.md 开头加：请始终用中文回复用户",
            },
            {
                "text": "只改这一行，不要改其他的",
                "reason_type": "unclear-default",
                "suggestion_location": "CLAUDE.md",
                "suggestion_content": "在 CLAUDE.md 加：修改时遵循最小改动原则",
            },
        ]
        mock_response = self._make_mock_response(mock_results)

        with patch("anthropic.Anthropic") as MockAnthropicClass:
            mock_client = MagicMock()
            MockAnthropicClass.return_value = mock_client
            mock_client.messages.create.return_value = mock_response

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
                stats = run_l2_repeated_prompts(db, budget)

        self.assertEqual(stats["analyzed"], 2)
        self.assertEqual(stats["suggestions_generated"], 2)  # 两条都有 CLAUDE.md 建议

        # 验证数据库写入
        cur = db.execute("SELECT canonical_text, analysis_json FROM repeated_prompts ORDER BY occurrences DESC")
        rows = cur.fetchall()
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNotNone(row["analysis_json"])
            analysis = json.loads(row["analysis_json"])
            self.assertIn("reason_type", analysis)
            self.assertIn("suggestion_location", analysis)

        db.close()

    def test_suggestion_location_none_not_counted(self):
        """suggestion_location=none 的条目不计入 suggestions_generated。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        self._insert_repeated_prompt(db, "继续", 6)

        mock_results = [
            {
                "text": "继续",
                "reason_type": "habit",
                "suggestion_location": "none",
                "suggestion_content": "",
            }
        ]
        mock_response = self._make_mock_response(mock_results)

        with patch("anthropic.Anthropic") as MockAnthropicClass:
            mock_client = MagicMock()
            MockAnthropicClass.return_value = mock_client
            mock_client.messages.create.return_value = mock_response

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
                stats = run_l2_repeated_prompts(db, budget)

        self.assertEqual(stats["analyzed"], 1)
        self.assertEqual(stats["suggestions_generated"], 0)  # none 不计
        db.close()

    def test_no_api_key_returns_zero(self):
        """没有 API key 时，分析失败，返回全零统计。"""
        db = _make_in_memory_db()
        budget = _make_budget(db)

        self._insert_repeated_prompt(db, "请帮我看看这个错误", 3)

        # 清除环境变量
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("ANTHROPIC_API_KEY", "TOWOW_ANTHROPIC_KEY")}
        with patch.dict("os.environ", env, clear=True):
            stats = run_l2_repeated_prompts(db, budget)

        self.assertEqual(stats["analyzed"], 0)
        db.close()


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
