"""
test_l3.py — L3 聚合层最小测试

测试策略：
  1. generate_summary() 不调用 LLM，直接测（用空 DB）
  2. run_l3() 在空 DB 上运行（无 corrections/clusters），只生成 summary，不报错
  3. mock LLM 调用测试 generate_rules() 在有数据时能跑通
"""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保 src 在路径上（如果在 cc-mirror 根目录运行 pytest 时会自动处理）
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cc_mirror.db import init_db
from cc_mirror.budget import BudgetController
from cc_mirror.l3_aggregator import (
    generate_rules,
    generate_skill_suggestions,
    generate_summary,
    generate_synthesis,
    run_l3,
)


# ---------------------------------------------------------------------------
# 共用 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """返回一个空的、完全初始化的 SQLite 连接（内存路径在 tmp_path 下）。"""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def budget(tmp_db):
    """返回与 tmp_db 关联的 BudgetController（预算 $10）。"""
    return BudgetController(budget_usd=10.0, db=tmp_db)


@pytest.fixture
def output_dir(tmp_path):
    """返回一个临时输出目录。"""
    d = tmp_path / "mirror-output"
    # 注意：不预先创建，测试 run_l3 是否会自动创建
    return d


# ---------------------------------------------------------------------------
# 辅助：在 DB 中插入测试数据
# ---------------------------------------------------------------------------

def _insert_session(db: sqlite3.Connection, session_id: str = "sess-001") -> None:
    """向 sessions 表插入一条最小测试记录。"""
    db.execute(
        """
        INSERT OR IGNORE INTO sessions
          (id, project, start_time, jsonl_path, message_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, "test-proj", "2026-01-01T00:00:00Z", "/tmp/test.jsonl", 5),
    )
    db.commit()


def _insert_correction(
    db: sqlite3.Connection,
    session_id: str = "sess-001",
    correction_type: str = "style",
    is_generalizable: bool = True,
    confidence: float = 0.9,
) -> None:
    """向 corrections 表插入一条测试纠正记录（依赖已存在的 session 和 messages）。"""
    # 先插入 session
    _insert_session(db, session_id)
    # 插入两条 messages（user + assistant），满足外键约束
    user_uuid = f"msg-user-{session_id}"
    asst_uuid = f"msg-asst-{session_id}"
    for uuid, msg_type in [(user_uuid, "user"), (asst_uuid, "assistant")]:
        db.execute(
            """
            INSERT OR IGNORE INTO messages
              (uuid, session_id, type, timestamp, sequence_num)
            VALUES (?, ?, ?, ?, ?)
            """,
            (uuid, session_id, msg_type, "2026-01-01T00:00:00Z", 1),
        )
    db.execute(
        """
        INSERT INTO corrections
          (session_id, project, user_message_uuid, assistant_message_uuid,
           cc_did, user_wanted, correction_type, is_generalizable, confidence,
           raw_user_text, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            "test-proj",
            user_uuid,
            asst_uuid,
            "CC 使用了 snake_case",
            "用户想要 camelCase",
            correction_type,
            is_generalizable,
            confidence,
            "请用 camelCase",
            "2026-01-01T00:00:00Z",
        ),
    )
    db.commit()


def _insert_workflow_cluster(
    db: sqlite3.Connection,
    is_skill_candidate: bool = True,
    session_count: int = 3,
) -> None:
    """向 workflow_clusters 表插入一条测试聚类记录。"""
    skill_suggestion = json.dumps({
        "is_skill_candidate": is_skill_candidate,
        "skill_name": "git-commit-workflow",
        "skill_trigger_scenario": "提交代码前的标准流程",
    })
    db.execute(
        """
        INSERT INTO workflow_clusters
          (tool_sequence_pattern, session_ids, session_count,
           similarity_threshold, description, skill_suggestion)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            json.dumps(["Read", "Edit", "Bash", "Bash"]),
            json.dumps(["sess-a", "sess-b", "sess-c"]),
            session_count,
            0.6,
            "读取文件 → 编辑 → 运行测试 → 提交",
            skill_suggestion,
        ),
    )
    db.commit()


# ---------------------------------------------------------------------------
# 测试 1：generate_summary() 不调用 LLM，空 DB 正常运行
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    """generate_summary 是纯数字聚合，不调用 LLM。"""

    def test_returns_nonempty_string_on_empty_db(self, tmp_db, budget):
        """空 DB 时也应返回非空 markdown 字符串。"""
        result = generate_summary(tmp_db, budget)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_required_sections(self, tmp_db, budget):
        """摘要应包含核心章节标题。"""
        result = generate_summary(tmp_db, budget)
        assert "## 数据概览" in result
        assert "## 纠正行为检测" in result
        assert "## 重复提示词" in result
        assert "## 工作流聚类" in result

    def test_zero_counts_on_empty_db(self, tmp_db, budget):
        """空 DB 的数字应全为 0。"""
        result = generate_summary(tmp_db, budget)
        # sessions 和 messages 应为 0
        assert "Sessions | 0" in result
        assert "Messages | 0" in result

    def test_counts_reflect_db_data(self, tmp_db, budget):
        """插入数据后，摘要数字应正确反映。"""
        _insert_session(tmp_db, "sess-001")
        _insert_session(tmp_db, "sess-002")

        result = generate_summary(tmp_db, budget)
        # 有 2 个 session
        assert "Sessions | 2" in result

    def test_corrections_section_shows_type_distribution(self, tmp_db, budget):
        """有纠正记录时，应展示类型分布。"""
        _insert_correction(tmp_db, "sess-001", correction_type="style")
        _insert_correction(tmp_db, "sess-002", correction_type="scope")

        result = generate_summary(tmp_db, budget)
        # LLM 确认数量
        assert "LLM 确认：2" in result

    def test_repeated_prompts_listed(self, tmp_db, budget):
        """有重复提示时应在摘要中列出。"""
        tmp_db.execute(
            """
            INSERT INTO repeated_prompts
              (canonical_text, occurrences, session_ids, project_ids, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("请用中文回复", 5, '["s1","s2","s3","s4","s5"]', '["p1"]',
             "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z"),
        )
        tmp_db.commit()

        result = generate_summary(tmp_db, budget)
        assert "请用中文回复" in result
        assert "出现 5 次" in result

    def test_no_llm_calls_made(self, tmp_db, budget):
        """generate_summary 不应发起任何 LLM 调用（llm_calls 表不增加）。"""
        before_count = tmp_db.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]

        generate_summary(tmp_db, budget)

        after_count = tmp_db.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]

        assert before_count == after_count, "generate_summary 不应调用 LLM"


# ---------------------------------------------------------------------------
# 测试 2：run_l3() 在空 DB 上运行（无 corrections/clusters）
# ---------------------------------------------------------------------------

class TestRunL3EmptyDb:
    """run_l3 在空 DB 上应正常运行，只生成 summary，不报错。"""

    def test_returns_dict_with_expected_keys(self, tmp_db, budget, output_dir):
        """返回值应包含所有预期键（包括新增的 synthesis）。"""
        result = run_l3(tmp_db, budget, output_dir)
        assert "rules_generated" in result
        assert "skills_suggested" in result
        assert "repeated_prompts_addressed" in result
        assert "output_files" in result
        assert "llm_calls" in result
        assert "synthesis" in result

    def test_output_dir_created_automatically(self, tmp_db, budget, output_dir):
        """output_dir 不存在时应自动创建。"""
        assert not output_dir.exists()
        run_l3(tmp_db, budget, output_dir)
        assert output_dir.exists()

    def test_four_files_generated(self, tmp_db, budget, output_dir):
        """即使 DB 为空，也应生成四个输出文件（加上 synthesis.md）。"""
        result = run_l3(tmp_db, budget, output_dir)
        assert len(result["output_files"]) == 4

    def test_all_output_files_exist(self, tmp_db, budget, output_dir):
        """三个输出文件都应实际存在于磁盘。"""
        result = run_l3(tmp_db, budget, output_dir)
        for fpath in result["output_files"]:
            assert Path(fpath).exists(), f"文件不存在：{fpath}"

    def test_summary_file_is_nonempty(self, tmp_db, budget, output_dir):
        """analysis-summary.md 应为非空文件。"""
        run_l3(tmp_db, budget, output_dir)
        summary_path = output_dir / "analysis-summary.md"
        assert summary_path.exists()
        content = summary_path.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_rules_file_contains_placeholder_when_empty(self, tmp_db, budget, output_dir):
        """无 corrections 时，suggested-rules.md 应包含占位说明（而非空文件）。"""
        run_l3(tmp_db, budget, output_dir)
        rules_path = output_dir / "suggested-rules.md"
        content = rules_path.read_text(encoding="utf-8")
        # 应有标题头
        assert "建议的 CLAUDE.md 规则" in content

    def test_zero_rules_when_no_corrections(self, tmp_db, budget, output_dir):
        """空 DB 时 rules_generated 应为 0。"""
        result = run_l3(tmp_db, budget, output_dir)
        assert result["rules_generated"] == 0

    def test_zero_skills_when_no_clusters(self, tmp_db, budget, output_dir):
        """空 DB 时 skills_suggested 应为 0。"""
        result = run_l3(tmp_db, budget, output_dir)
        assert result["skills_suggested"] == 0

    def test_idempotent_rerun(self, tmp_db, budget, output_dir):
        """重复运行应覆盖文件，不报错。"""
        run_l3(tmp_db, budget, output_dir)
        run_l3(tmp_db, budget, output_dir)  # 第二次应正常
        summary_path = output_dir / "analysis-summary.md"
        assert summary_path.exists()

    def test_stop_budget_skips_llm_still_generates_summary(self, tmp_db, output_dir):
        """budget 超 80% 时，应跳过 L3a/L3b，但仍生成 summary。"""
        # 构造一个已花费 90% 的 budget（$9/$10）
        budget_ctrl = BudgetController(budget_usd=10.0, db=tmp_db)
        budget_ctrl.record_call("L2", "claude-sonnet-4-6", 1000, 100, 9.0)

        result = run_l3(tmp_db, budget_ctrl, output_dir)

        # summary 文件必须存在
        summary_path = output_dir / "analysis-summary.md"
        assert summary_path.exists()

        # rules/skills 应为 0（跳过了 LLM 调用）
        assert result["rules_generated"] == 0
        assert result["skills_suggested"] == 0


# ---------------------------------------------------------------------------
# 测试 3：generate_rules() mock LLM 调用，有数据时能跑通
# ---------------------------------------------------------------------------

class TestGenerateRulesMocked:
    """mock Anthropic 客户端，测试 generate_rules 有数据时的完整流程。"""

    def _make_mock_response(self, text: str = "## 代码风格规则\n- 使用 camelCase\n- 避免魔法数字"):
        """构造一个模拟 Anthropic API 响应对象。"""
        usage = MagicMock()
        usage.input_tokens = 500
        usage.output_tokens = 100

        content_item = MagicMock()
        content_item.text = text

        response = MagicMock()
        response.usage = usage
        response.content = [content_item]
        return response

    def test_no_corrections_returns_empty_string(self, tmp_db, budget):
        """没有符合条件的 corrections 时，直接返回空字符串，不调用 LLM。"""
        result = generate_rules(tmp_db, budget)
        assert result == ""

    def test_low_confidence_corrections_skipped(self, tmp_db, budget):
        """置信度 < 0.7 的记录不应触发 LLM 调用。"""
        _insert_correction(tmp_db, "sess-001", confidence=0.5, is_generalizable=True)
        result = generate_rules(tmp_db, budget)
        assert result == ""

    def test_not_generalizable_corrections_skipped(self, tmp_db, budget):
        """is_generalizable=False 的记录不应触发 LLM 调用。"""
        _insert_correction(tmp_db, "sess-001", confidence=0.9, is_generalizable=False)
        result = generate_rules(tmp_db, budget)
        assert result == ""

    def test_valid_correction_calls_opus_and_returns_markdown(self, tmp_db, budget):
        """有有效 correction 时，应调用 Opus 并返回 markdown 内容。"""
        _insert_correction(tmp_db, "sess-001", confidence=0.9, is_generalizable=True)

        mock_response = self._make_mock_response(
            "## 代码风格规则\n- 使用 camelCase\n\n## 工作流规则\n- 提交前运行测试"
        )

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                result = generate_rules(tmp_db, budget)

        assert "代码风格规则" in result
        assert "camelCase" in result

    def test_opus_call_recorded_in_budget(self, tmp_db, budget):
        """Opus 调用应被记录到 llm_calls 表。"""
        _insert_correction(tmp_db, "sess-001", confidence=0.9, is_generalizable=True)

        mock_response = self._make_mock_response()

        before_count = tmp_db.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                generate_rules(tmp_db, budget)

        after_count = tmp_db.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]

        assert after_count == before_count + 1, "应有一次 LLM 调用被记录"

    def test_opus_model_used_for_rules(self, tmp_db, budget):
        """generate_rules 应使用 Opus 模型（不是 Sonnet）。"""
        _insert_correction(tmp_db, "sess-001", confidence=0.9, is_generalizable=True)

        mock_response = self._make_mock_response()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                generate_rules(tmp_db, budget)

        # 检查 messages.create 被调用时的 model 参数
        call_kwargs = mock_client.messages.create.call_args
        # 支持位置参数或关键字参数
        model_used = call_kwargs.kwargs.get("model") or call_kwargs.args[0] if call_kwargs.args else None
        if model_used is None:
            model_used = call_kwargs[1].get("model")
        assert model_used is not None
        assert "opus" in model_used.lower() or "claude-opus" in model_used

    def test_no_api_key_returns_empty_string(self, tmp_db, budget):
        """没有 API key 时，应返回空字符串而不是 raise。"""
        _insert_correction(tmp_db, "sess-001", confidence=0.9, is_generalizable=True)

        # 确保环境变量不存在
        import os
        env_backup = {}
        for key in ("ANTHROPIC_API_KEY", "TOWOW_ANTHROPIC_KEY"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            result = generate_rules(tmp_db, budget)
            assert result == ""
        finally:
            # 恢复环境变量
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_multiple_correction_types_grouped(self, tmp_db, budget):
        """多种类型的 correction 应被正确分组传入 prompt。"""
        _insert_correction(tmp_db, "sess-001", correction_type="style", confidence=0.9)
        _insert_correction(tmp_db, "sess-002", correction_type="scope",  confidence=0.8)

        # 用 capture 记录 prompt 内容
        captured_prompt = []

        mock_response = self._make_mock_response("## 代码风格规则\n- 规则1")

        def fake_create(**kwargs):
            captured_prompt.append(kwargs.get("messages", []))
            return mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.side_effect = fake_create

                generate_rules(tmp_db, budget)

        # prompt 应包含两种类型
        assert len(captured_prompt) == 1
        prompt_text = captured_prompt[0][0]["content"]
        assert "style" in prompt_text
        assert "scope" in prompt_text


# ---------------------------------------------------------------------------
# 测试 4：generate_skill_suggestions() mock LLM
# ---------------------------------------------------------------------------

class TestGenerateSkillSuggestionsMocked:
    """mock Anthropic 客户端，测试 generate_skill_suggestions 有数据时的完整流程。"""

    def _make_mock_response(self, text: str = "## /git-commit-workflow\n**触发场景**：提交代码\n**工具序列**：Read → Edit → Bash"):
        usage = MagicMock()
        usage.input_tokens = 300
        usage.output_tokens = 150

        content_item = MagicMock()
        content_item.text = text

        response = MagicMock()
        response.usage = usage
        response.content = [content_item]
        return response

    def test_no_clusters_returns_empty_string(self, tmp_db, budget):
        """没有 Skill 候选聚类时返回空字符串。"""
        result = generate_skill_suggestions(tmp_db, budget)
        assert result == ""

    def test_non_skill_candidate_skipped(self, tmp_db, budget):
        """is_skill_candidate=false 的聚类应被跳过。"""
        _insert_workflow_cluster(tmp_db, is_skill_candidate=False)
        result = generate_skill_suggestions(tmp_db, budget)
        assert result == ""

    def test_skill_candidate_calls_sonnet_and_returns_markdown(self, tmp_db, budget):
        """有 Skill 候选时，应调用 Sonnet 并返回 markdown 内容。"""
        _insert_workflow_cluster(tmp_db, is_skill_candidate=True)

        mock_response = self._make_mock_response(
            "## /git-commit-workflow\n**触发场景**：提交代码\n**工具序列**：Read → Edit → Bash\n"
            "**建议步骤**：\n1. 读取文件\n2. 编辑\n3. 提交"
        )

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                result = generate_skill_suggestions(tmp_db, budget)

        assert len(result) > 0
        assert "git-commit-workflow" in result


# ---------------------------------------------------------------------------
# 测试 5：run_l3() 文件内容验证
# ---------------------------------------------------------------------------

class TestRunL3FileContents:
    """验证 run_l3 生成的文件名和内容结构。"""

    def test_output_filenames_are_fixed(self, tmp_db, budget, output_dir):
        """输出文件名应固定（不含时间戳），包括新增的 synthesis.md。"""
        run_l3(tmp_db, budget, output_dir)

        expected_files = {
            "suggested-rules.md",
            "suggested-skills.md",
            "analysis-summary.md",
            "synthesis.md",
        }
        actual_files = {f.name for f in output_dir.iterdir() if f.is_file()}
        assert expected_files == actual_files

    def test_summary_has_cc_mirror_title(self, tmp_db, budget, output_dir):
        """analysis-summary.md 应以 '# CC Mirror 分析摘要' 开头。"""
        run_l3(tmp_db, budget, output_dir)
        content = (output_dir / "analysis-summary.md").read_text(encoding="utf-8")
        assert content.startswith("# CC Mirror 分析摘要")

    def test_rules_file_has_auto_generated_header(self, tmp_db, budget, output_dir):
        """suggested-rules.md 应包含自动生成标注。"""
        run_l3(tmp_db, budget, output_dir)
        content = (output_dir / "suggested-rules.md").read_text(encoding="utf-8")
        assert "CC Mirror" in content
        assert "自动生成" in content

    def test_skills_file_has_auto_generated_header(self, tmp_db, budget, output_dir):
        """suggested-skills.md 应包含自动生成标注。"""
        run_l3(tmp_db, budget, output_dir)
        content = (output_dir / "suggested-skills.md").read_text(encoding="utf-8")
        assert "CC Mirror" in content
        assert "自动生成" in content

    def test_output_files_are_absolute_paths(self, tmp_db, budget, output_dir):
        """output_files 列表中的路径应全为绝对路径。"""
        result = run_l3(tmp_db, budget, output_dir)
        for fpath in result["output_files"]:
            assert Path(fpath).is_absolute(), f"路径应为绝对路径：{fpath}"

    def test_synthesis_file_created(self, tmp_db, budget, output_dir):
        """synthesis.md 应存在于 output_dir 中。"""
        run_l3(tmp_db, budget, output_dir)
        synthesis_path = output_dir / "synthesis.md"
        assert synthesis_path.exists(), "synthesis.md 文件不存在"

    def test_synthesis_key_in_result(self, tmp_db, budget, output_dir):
        """返回值中 synthesis 键应为字符串类型。"""
        result = run_l3(tmp_db, budget, output_dir)
        assert isinstance(result["synthesis"], str)


# ---------------------------------------------------------------------------
# 测试 6：generate_synthesis() mock LLM 调用
# ---------------------------------------------------------------------------

class TestGenerateSynthesisMocked:
    """mock Anthropic 客户端，测试 generate_synthesis 有数据时的完整流程。"""

    def _make_mock_response(self, text: str = "这位工程师频繁使用 CC 完成代码重构任务，整体工具调用强度偏高。"):
        """构造一个模拟 Anthropic API 响应对象。"""
        usage = MagicMock()
        usage.input_tokens = 600
        usage.output_tokens = 150

        content_item = MagicMock()
        content_item.text = text

        response = MagicMock()
        response.usage = usage
        response.content = [content_item]
        return response

    def test_no_api_key_returns_empty_string(self, tmp_db, budget):
        """没有 API key 时，返回空字符串而不是 raise。"""
        import os
        env_backup = {}
        for key in ("ANTHROPIC_API_KEY", "TOWOW_ANTHROPIC_API_KEY"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            result = generate_synthesis(tmp_db, budget)
            assert result == ""
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_empty_db_calls_opus_and_returns_text(self, tmp_db, budget):
        """空 DB 时也应尝试调用 Opus（数据全为 0），并返回文本。"""
        mock_response = self._make_mock_response()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                result = generate_synthesis(tmp_db, budget)

        assert len(result) > 0
        assert "工程师" in result

    def test_uses_opus_model(self, tmp_db, budget):
        """generate_synthesis 应使用 Opus 模型。"""
        mock_response = self._make_mock_response()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                generate_synthesis(tmp_db, budget)

        call_kwargs = mock_client.messages.create.call_args
        model_used = call_kwargs.kwargs.get("model")
        assert model_used is not None
        assert "opus" in model_used.lower()

    def test_call_recorded_in_budget(self, tmp_db, budget):
        """Opus 调用应被记录到 llm_calls 表。"""
        mock_response = self._make_mock_response()

        before_count = tmp_db.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                generate_synthesis(tmp_db, budget)

        after_count = tmp_db.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]

        assert after_count == before_count + 1

    def test_api_failure_returns_empty_string(self, tmp_db, budget):
        """API 调用失败时返回空字符串，不崩溃。"""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.side_effect = Exception("network error")

                result = generate_synthesis(tmp_db, budget)

        assert result == ""

    def test_synthesis_written_to_file_by_run_l3(self, tmp_db, budget, output_dir):
        """run_l3 应将 synthesis 写入 synthesis.md 文件。"""
        mock_response = self._make_mock_response("这是综合洞察分析内容。")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("cc_mirror.l3_aggregator.anthropic") as mock_anthropic_module:
                mock_client = MagicMock()
                mock_anthropic_module.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                result = run_l3(tmp_db, budget, output_dir)

        synthesis_path = output_dir / "synthesis.md"
        assert synthesis_path.exists()
        content = synthesis_path.read_text(encoding="utf-8")
        assert "这是综合洞察分析内容" in content
        assert result["synthesis"] == "这是综合洞察分析内容。"

    def test_synthesis_empty_on_stop_budget(self, tmp_db, output_dir):
        """budget 超 80% 时，synthesis 应为空字符串。"""
        budget_ctrl = BudgetController(budget_usd=10.0, db=tmp_db)
        budget_ctrl.record_call("L2", "claude-sonnet-4-6", 1000, 100, 9.0)

        result = run_l3(tmp_db, budget_ctrl, output_dir)

        assert result["synthesis"] == ""
        # synthesis.md 应存在但包含占位说明
        synthesis_path = output_dir / "synthesis.md"
        assert synthesis_path.exists()
        content = synthesis_path.read_text(encoding="utf-8")
        assert "未生成" in content
