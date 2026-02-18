"""
test_l2.py — BudgetController 和 l2_correction 单元测试

覆盖：
1. BudgetController.get_strategy() 三个状态边界
2. BudgetController.record_call() 写入 DB 后 spent_usd 变化
3. build_prompt() prompt 构造正确性
4. _parse_llm_response() JSON 解析（裸 JSON 和 markdown 代码块）

mock anthropic API，不实际调用 LLM。
"""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# 把 src/ 加入路径（不依赖 pip install -e）
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cc_mirror.budget import BudgetController
from cc_mirror.db import init_db
from cc_mirror.l2_correction import (
    _parse_llm_response,
    build_prompt,
)


# ---------------------------------------------------------------------------
# 工具：创建内存数据库
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    """创建内存 SQLite，应用完整 schema。"""
    import tempfile, os
    # init_db 需要写入文件（WAL 模式不支持 :memory: 完整功能）
    # 用临时文件，测试结束后自动删除
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = init_db(Path(tmp.name))
    return conn


# ---------------------------------------------------------------------------
# 1. BudgetController.get_strategy() 边界测试
# ---------------------------------------------------------------------------

class TestBudgetGetStrategy(unittest.TestCase):
    """测试预算策略的三个区间。"""

    def setUp(self):
        self.db = _make_db()

    def tearDown(self):
        self.db.close()

    def _spend(self, budget: BudgetController, amount: float) -> None:
        """向 budget 记录一次 amount 元的 LLM 调用。"""
        budget.record_call(
            stage="L2",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            cost_usd=amount,
            duration_ms=100,
        )

    def test_strategy_full_at_zero(self):
        """初始状态（零消耗）→ full。"""
        budget = BudgetController(budget_usd=10.0, db=self.db)
        self.assertEqual(budget.get_strategy(), "full")

    def test_strategy_full_below_50_percent(self):
        """消耗 49% → full。"""
        budget = BudgetController(budget_usd=10.0, db=self.db)
        self._spend(budget, 4.9)  # 49%
        self.assertEqual(budget.get_strategy(), "full")

    def test_strategy_p0_only_at_50_percent(self):
        """消耗刚过 50% → p0_only。"""
        budget = BudgetController(budget_usd=10.0, db=self.db)
        self._spend(budget, 5.01)  # 50.1%
        self.assertEqual(budget.get_strategy(), "p0_only")

    def test_strategy_p0_only_between_50_and_80(self):
        """消耗 70% → p0_only。"""
        budget = BudgetController(budget_usd=10.0, db=self.db)
        self._spend(budget, 7.0)  # 70%
        self.assertEqual(budget.get_strategy(), "p0_only")

    def test_strategy_stop_at_80_percent(self):
        """消耗刚过 80% → stop。"""
        budget = BudgetController(budget_usd=10.0, db=self.db)
        self._spend(budget, 8.01)  # 80.1%
        self.assertEqual(budget.get_strategy(), "stop")

    def test_strategy_stop_over_budget(self):
        """消耗超过总预算 → stop。"""
        budget = BudgetController(budget_usd=10.0, db=self.db)
        self._spend(budget, 15.0)  # 150%
        self.assertEqual(budget.get_strategy(), "stop")

    def test_strategy_reads_from_db(self):
        """
        get_strategy() 必须从 DB 读取（不依赖内存缓存）。
        模拟多进程：另一个 BudgetController 实例写入，当前实例查询。
        """
        budget_writer = BudgetController(budget_usd=10.0, db=self.db)
        budget_reader = BudgetController(budget_usd=10.0, db=self.db)

        # writer 写入 6.0（60%）
        self._spend(budget_writer, 6.0)

        # reader 的内存缓存未更新，但 get_strategy() 从 DB 读
        self.assertEqual(budget_reader.get_strategy(), "p0_only")


# ---------------------------------------------------------------------------
# 2. BudgetController.record_call() 写入 DB 后 spent_usd 变化
# ---------------------------------------------------------------------------

class TestBudgetRecordCall(unittest.TestCase):
    """测试 record_call 写入行为。"""

    def setUp(self):
        self.db = _make_db()

    def tearDown(self):
        self.db.close()

    def test_record_call_updates_spent_usd(self):
        """record_call 后 spent_usd 应增加对应金额。"""
        budget = BudgetController(budget_usd=20.0, db=self.db)
        self.assertAlmostEqual(budget.spent_usd, 0.0)

        budget.record_call("L2", "claude-sonnet-4-6", 1000, 200, 0.005)
        self.assertAlmostEqual(budget.spent_usd, 0.005, places=5)

        budget.record_call("L3", "claude-opus-4-6", 2000, 500, 0.03)
        self.assertAlmostEqual(budget.spent_usd, 0.035, places=5)

    def test_record_call_persists_to_db(self):
        """record_call 写入的数据应能从 DB 查询到。"""
        budget = BudgetController(budget_usd=20.0, db=self.db)
        budget.record_call("L2", "claude-sonnet-4-6", 500, 100, 0.002, duration_ms=150)

        cur = self.db.execute(
            "SELECT stage, model, input_tokens, output_tokens, cost_usd, duration_ms "
            "FROM llm_calls"
        )
        rows = cur.fetchall()
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["stage"], "L2")
        self.assertEqual(row["model"], "claude-sonnet-4-6")
        self.assertEqual(row["input_tokens"], 500)
        self.assertEqual(row["output_tokens"], 100)
        self.assertAlmostEqual(row["cost_usd"], 0.002, places=5)
        self.assertEqual(row["duration_ms"], 150)

    def test_remaining_usd_decreases(self):
        """remaining_usd 应随调用减少。"""
        budget = BudgetController(budget_usd=10.0, db=self.db)
        self.assertAlmostEqual(budget.remaining_usd, 10.0)

        budget.record_call("L2", "claude-sonnet-4-6", 100, 50, 3.0)
        self.assertAlmostEqual(budget.remaining_usd, 7.0, places=5)

    def test_remaining_usd_not_negative(self):
        """remaining_usd 超出预算时应返回 0，不返回负数。"""
        budget = BudgetController(budget_usd=5.0, db=self.db)
        budget.record_call("L2", "claude-sonnet-4-6", 100, 50, 10.0)  # 超出预算
        self.assertGreaterEqual(budget.remaining_usd, 0.0)

    def test_new_instance_reads_existing_spent(self):
        """新创建的 BudgetController 实例应读取 DB 中已有的消耗记录。"""
        budget1 = BudgetController(budget_usd=20.0, db=self.db)
        budget1.record_call("L2", "claude-sonnet-4-6", 100, 50, 1.5)

        # 新实例，相同 DB
        budget2 = BudgetController(budget_usd=20.0, db=self.db)
        self.assertAlmostEqual(budget2.spent_usd, 1.5, places=5)


# ---------------------------------------------------------------------------
# 3. build_prompt() 测试
# ---------------------------------------------------------------------------

class TestBuildPrompt(unittest.TestCase):
    """测试 prompt 构造函数。"""

    def test_prompt_with_context(self):
        """有 assistant 上下文时，prompt 应包含上下文内容。"""
        context = ["我来实现登录功能。使用 default export。", "好的，已经写完了。"]
        user_msg = "不对，我们用 named export，用 import/export 语法"

        prompt = build_prompt(user_msg, context)

        self.assertIn("不对，我们用 named export", prompt)
        self.assertIn("我来实现登录功能", prompt)
        self.assertIn("已经写完了", prompt)
        self.assertIn("is_correction", prompt)
        self.assertIn("correction_type", prompt)

    def test_prompt_without_context(self):
        """没有 assistant 上下文时，prompt 应包含占位提示，但仍合法。"""
        prompt = build_prompt("这个方向不对，换一种方式", [])

        self.assertIn("无 assistant 上下文", prompt)
        self.assertIn("is_correction", prompt)

    def test_prompt_truncates_long_assistant_text(self):
        """超长 assistant 文本应被截断（避免 token 爆炸）。"""
        long_text = "A" * 1000  # 远超 500 字符限制
        prompt = build_prompt("纠正消息", [long_text])

        # 截断后应有"…（已截断）"标记
        self.assertIn("已截断", prompt)
        # 实际内容不超过 500 + 标记长度
        self.assertNotIn("A" * 600, prompt)

    def test_prompt_contains_json_template(self):
        """prompt 必须包含 JSON 输出模板所有字段。"""
        prompt = build_prompt("测试消息", [])

        required_fields = [
            "is_correction",
            "cc_did",
            "user_wanted",
            "correction_type",
            "is_generalizable",
            "confidence",
        ]
        for field in required_fields:
            self.assertIn(field, prompt, f"prompt 缺少字段: {field}")


# ---------------------------------------------------------------------------
# 4. _parse_llm_response() 测试
# ---------------------------------------------------------------------------

class TestParseLlmResponse(unittest.TestCase):
    """测试 LLM 响应 JSON 解析。"""

    _VALID_RESPONSE = """{
  "is_correction": true,
  "cc_did": "使用了 default export",
  "user_wanted": "使用 named export",
  "correction_type": "style",
  "is_generalizable": true,
  "confidence": 0.95
}"""

    def test_parse_bare_json(self):
        """裸 JSON 格式（无 markdown 代码块）应正常解析。"""
        result = _parse_llm_response(self._VALID_RESPONSE)

        self.assertIsNotNone(result)
        self.assertTrue(result["is_correction"])
        self.assertEqual(result["cc_did"], "使用了 default export")
        self.assertEqual(result["correction_type"], "style")
        self.assertAlmostEqual(result["confidence"], 0.95)

    def test_parse_markdown_wrapped_json(self):
        """markdown 代码块包裹的 JSON 应正确剥离后解析。"""
        wrapped = f"```json\n{self._VALID_RESPONSE}\n```"
        result = _parse_llm_response(wrapped)

        self.assertIsNotNone(result)
        self.assertTrue(result["is_correction"])

    def test_parse_markdown_no_lang_tag(self):
        """无语言标签的 markdown 代码块也应正确解析。"""
        wrapped = f"```\n{self._VALID_RESPONSE}\n```"
        result = _parse_llm_response(wrapped)

        self.assertIsNotNone(result)
        self.assertTrue(result["is_correction"])

    def test_parse_not_correction(self):
        """is_correction=false 的响应应正常解析。"""
        response = """{
  "is_correction": false,
  "cc_did": "",
  "user_wanted": "",
  "correction_type": "other",
  "is_generalizable": false,
  "confidence": 0.1
}"""
        result = _parse_llm_response(response)

        self.assertIsNotNone(result)
        self.assertFalse(result["is_correction"])

    def test_parse_invalid_json_returns_none(self):
        """无效 JSON 应返回 None，不抛异常。"""
        result = _parse_llm_response("这不是 JSON")
        self.assertIsNone(result)

    def test_parse_empty_string_returns_none(self):
        """空字符串应返回 None。"""
        result = _parse_llm_response("")
        self.assertIsNone(result)

    def test_parse_partial_json_returns_none(self):
        """不完整的 JSON 应返回 None。"""
        result = _parse_llm_response('{"is_correction": true,')
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 5. BudgetController 边界和输入验证
# ---------------------------------------------------------------------------

class TestBudgetEdgeCases(unittest.TestCase):
    """BudgetController 边界情况测试。"""

    def setUp(self):
        self.db = _make_db()

    def tearDown(self):
        self.db.close()

    def test_invalid_budget_raises(self):
        """budget_usd <= 0 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            BudgetController(budget_usd=0.0, db=self.db)

        with self.assertRaises(ValueError):
            BudgetController(budget_usd=-5.0, db=self.db)

    def test_budget_property(self):
        """budget_usd property 应返回正确总预算。"""
        budget = BudgetController(budget_usd=42.5, db=self.db)
        self.assertAlmostEqual(budget.budget_usd, 42.5)

    def test_multiple_calls_accumulate(self):
        """多次 record_call 应正确累加。"""
        budget = BudgetController(budget_usd=100.0, db=self.db)

        amounts = [0.001, 0.002, 0.003, 0.004, 0.005]
        for amount in amounts:
            budget.record_call("L2", "model", 100, 50, amount)

        expected = sum(amounts)
        self.assertAlmostEqual(budget.spent_usd, expected, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
