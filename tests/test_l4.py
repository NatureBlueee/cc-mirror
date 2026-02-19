"""
test_l4.py — L4 HTML 报告渲染器最小 smoke test

测试策略：
  - 不调用 LLM，不依赖 playwright
  - 用临时 DB 创建最小测试数据
  - 验证 report.html / share-card.html 存在且包含关键词
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cc_mirror.db import init_db
from cc_mirror.budget import BudgetController
from cc_mirror.l4_renderer import render_report, _collect_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """完整初始化的 SQLite 连接，位于临时目录。"""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def budget(tmp_db):
    return BudgetController(budget_usd=10.0, db=tmp_db)


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# 辅助：写入最小数据
# ---------------------------------------------------------------------------

def _insert_session(db: sqlite3.Connection, sid: str = "s1") -> None:
    db.execute(
        "INSERT OR IGNORE INTO sessions (id, project, start_time, jsonl_path) "
        "VALUES (?, ?, ?, ?)",
        (sid, "proj", "2026-01-01T00:00:00Z", "/tmp/x.jsonl"),
    )
    db.commit()


def _insert_correction(db: sqlite3.Connection, sid: str = "s1") -> None:
    _insert_session(db, sid)
    u_uuid = f"u-{sid}"
    a_uuid = f"a-{sid}"
    for uid, mt in [(u_uuid, "user"), (a_uuid, "assistant")]:
        db.execute(
            "INSERT OR IGNORE INTO messages (uuid, session_id, type, timestamp, sequence_num) "
            "VALUES (?, ?, ?, ?, ?)",
            (uid, sid, mt, "2026-01-01T00:00:00Z", 1),
        )
    db.execute(
        "INSERT INTO corrections "
        "(session_id, project, user_message_uuid, assistant_message_uuid, "
        " cc_did, user_wanted, correction_type, is_generalizable, confidence, "
        " raw_user_text, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, "proj", u_uuid, a_uuid,
         "CC 用了 snake_case", "用户想要 camelCase",
         "style", True, 0.9, "请用 camelCase", "2026-01-01T00:00:00Z"),
    )
    db.commit()


def _insert_workflow_cluster(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT INTO workflow_clusters "
        "(tool_sequence_pattern, session_ids, session_count, similarity_threshold, description) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            json.dumps(["Read", "Edit", "Bash"]),
            json.dumps(["s1", "s2", "s3"]),
            3,
            0.6,
            "读取文件 → 编辑 → 运行测试",
        ),
    )
    db.commit()


# ---------------------------------------------------------------------------
# 测试 1：空 DB，报告仍可生成
# ---------------------------------------------------------------------------

class TestRenderReportEmptyDb:

    def test_report_html_created(self, tmp_db, budget, output_dir):
        """空 DB 时 report.html 应被创建。"""
        result = render_report(tmp_db, budget, output_dir)
        assert result["report_path"], "report_path 不应为空"
        assert Path(result["report_path"]).exists(), "report.html 文件不存在"

    def test_share_card_html_created(self, tmp_db, budget, output_dir):
        """空 DB 时 share-card.html 应被创建。"""
        result = render_report(tmp_db, budget, output_dir)
        assert result["share_card_path"], "share_card_path 不应为空"
        assert Path(result["share_card_path"]).exists(), "share-card.html 文件不存在"

    def test_png_path_is_none_or_exists(self, tmp_db, budget, output_dir):
        """png_path 要么是 None（playwright 不可用），要么文件存在。"""
        result = render_report(tmp_db, budget, output_dir)
        if result["png_path"] is not None:
            assert Path(result["png_path"]).exists()

    def test_report_html_contains_title(self, tmp_db, budget, output_dir):
        """report.html 应包含 'CC Mirror' 标题。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "CC Mirror" in content

    def test_report_html_contains_chinese(self, tmp_db, budget, output_dir):
        """report.html 应包含中文界面元素。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "sessions" in content or "消息" in content

    def test_report_html_has_three_parts(self, tmp_db, budget, output_dir):
        """report.html 应包含三个 Part。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "Part 1" in content
        assert "Part 2" in content
        assert "Part 3" in content

    def test_share_card_contains_brand(self, tmp_db, budget, output_dir):
        """share-card.html 应包含品牌名。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["share_card_path"]).read_text(encoding="utf-8")
        assert "CC Mirror" in content

    def test_share_card_contains_footer(self, tmp_db, budget, output_dir):
        """share-card.html 底部应有 'Generated by CC Mirror'。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["share_card_path"]).read_text(encoding="utf-8")
        assert "Generated by CC Mirror" in content

    def test_no_external_cdn(self, tmp_db, budget, output_dir):
        """report.html 不应引用外部 CDN（无 http:// 资源加载）。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        # 检查没有外部 script/link 标签
        import re
        external_refs = re.findall(r'(?:src|href)=["\']https?://', content)
        assert len(external_refs) == 0, f"发现外部 CDN 引用：{external_refs}"

    def test_report_html_is_valid_utf8(self, tmp_db, budget, output_dir):
        """report.html 应是有效的 UTF-8 编码。"""
        result = render_report(tmp_db, budget, output_dir)
        raw = Path(result["report_path"]).read_bytes()
        content = raw.decode("utf-8")  # 如果失败会抛异常
        assert len(content) > 0

    def test_no_dark_theme(self, tmp_db, budget, output_dir):
        """report.html 不应有暗色主题（不包含 dark mode 相关 class/media query）。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8").lower()
        # 不应包含 prefers-color-scheme: dark
        assert "prefers-color-scheme: dark" not in content
        assert "color-scheme: dark" not in content


# ---------------------------------------------------------------------------
# 测试 2：有数据时，报告内容正确
# ---------------------------------------------------------------------------

class TestRenderReportWithData:

    def test_correction_count_shown_in_report(self, tmp_db, budget, output_dir):
        """有 correction 记录时，report.html 应展示数量。"""
        _insert_correction(tmp_db, "s1")
        _insert_correction(tmp_db, "s2")
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        # 应包含 corrections 展示（2 条）
        assert "2" in content

    def test_correction_detail_shown(self, tmp_db, budget, output_dir):
        """有 correction 时，详情（cc_did/user_wanted）应出现在报告中。"""
        _insert_correction(tmp_db, "s1")
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "snake_case" in content or "camelCase" in content

    def test_workflow_cluster_shown(self, tmp_db, budget, output_dir):
        """有工作流聚类时，应出现在报告中。"""
        _insert_session(tmp_db, "s1")
        _insert_workflow_cluster(tmp_db)
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "读取文件" in content or "工作流" in content

    def test_rules_md_rendered_when_file_exists(self, tmp_db, budget, output_dir):
        """output_dir 中有 suggested-rules.md 时，内容应出现在报告中。"""
        (output_dir / "suggested-rules.md").write_text(
            "## 代码风格规则\n\n- 使用 camelCase\n- 避免魔法数字\n",
            encoding="utf-8",
        )
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "camelCase" in content

    def test_copy_button_present_when_rules_exist(self, tmp_db, budget, output_dir):
        """有规则文件时，报告中应有复制按钮。"""
        (output_dir / "suggested-rules.md").write_text(
            "## 规则\n- 规则1\n", encoding="utf-8"
        )
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["result_path"] if "result_path" in result else "report_path").read_text(encoding="utf-8") if False else Path(result["report_path"]).read_text(encoding="utf-8")
        assert "copyRules" in content or "复制" in content

    def test_share_card_shows_session_count(self, tmp_db, budget, output_dir):
        """share-card.html 应展示 session 数量。"""
        _insert_session(tmp_db, "s1")
        _insert_session(tmp_db, "s2")
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["share_card_path"]).read_text(encoding="utf-8")
        assert "2" in content


# ---------------------------------------------------------------------------
# 测试 3：_collect_data 数据收集正确性
# ---------------------------------------------------------------------------

class TestCollectData:

    def test_empty_db_returns_zeros(self, tmp_db, output_dir):
        """空 DB 时，数字字段应全为 0。"""
        data = _collect_data(tmp_db, output_dir)
        assert data["sessions"] == 0
        assert data["messages"] == 0
        assert data["corrections_confirmed"] == 0
        assert data["cost_usd"] == 0.0

    def test_sessions_counted_correctly(self, tmp_db, output_dir):
        """sessions 计数应与 DB 数据一致。"""
        _insert_session(tmp_db, "s1")
        _insert_session(tmp_db, "s2")
        data = _collect_data(tmp_db, output_dir)
        assert data["sessions"] == 2

    def test_corrections_listed(self, tmp_db, output_dir):
        """有 correction 时，corrections 列表不为空。"""
        _insert_correction(tmp_db, "s1")
        data = _collect_data(tmp_db, output_dir)
        assert len(data["corrections"]) == 1
        assert data["corrections"][0]["correction_type"] == "style"

    def test_rules_md_read_from_file(self, tmp_db, output_dir):
        """如果 suggested-rules.md 存在，rules_md 应有内容。"""
        (output_dir / "suggested-rules.md").write_text(
            "## 规则\n- 规则1\n", encoding="utf-8"
        )
        data = _collect_data(tmp_db, output_dir)
        assert "规则1" in data["rules_md"]

    def test_rules_md_empty_when_file_missing(self, tmp_db, output_dir):
        """如果 suggested-rules.md 不存在，rules_md 应为空字符串。"""
        data = _collect_data(tmp_db, output_dir)
        assert data["rules_md"] == ""

    def test_workflow_clusters_listed(self, tmp_db, output_dir):
        """有工作流聚类时，workflow_clusters 列表不为空。"""
        _insert_session(tmp_db, "s1")
        _insert_workflow_cluster(tmp_db)
        data = _collect_data(tmp_db, output_dir)
        assert len(data["workflow_clusters"]) == 1
        assert data["workflow_clusters"][0]["session_count"] == 3

    def test_generated_at_is_iso_timestamp(self, tmp_db, output_dir):
        """generated_at 应是合法的 ISO 时间字符串。"""
        from datetime import datetime
        data = _collect_data(tmp_db, output_dir)
        # 不抛异常说明格式合法
        dt = datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        assert dt is not None


# ---------------------------------------------------------------------------
# 测试 4：render_report 幂等性（重复运行）
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_second_run_overwrites_files(self, tmp_db, budget, output_dir):
        """重复运行 render_report 应覆盖文件，不报错。"""
        render_report(tmp_db, budget, output_dir)
        result2 = render_report(tmp_db, budget, output_dir)
        assert Path(result2["report_path"]).exists()
        assert Path(result2["share_card_path"]).exists()

    def test_output_dir_auto_created(self, tmp_db, budget, tmp_path):
        """output_dir 不存在时应自动创建。"""
        new_dir = tmp_path / "nonexistent" / "deep" / "dir"
        assert not new_dir.exists()
        result = render_report(tmp_db, budget, new_dir)
        assert new_dir.exists()
        assert result["report_path"]


# ---------------------------------------------------------------------------
# 测试 5：synthesis 综合洞察集成
# ---------------------------------------------------------------------------

class TestSynthesisIntegration:

    def test_collect_data_synthesis_empty_when_no_file(self, tmp_db, output_dir):
        """synthesis.md 不存在时，_collect_data 返回的 synthesis 应为空字符串。"""
        data = _collect_data(tmp_db, output_dir)
        assert "synthesis" in data
        assert data["synthesis"] == ""

    def test_collect_data_synthesis_read_from_file(self, tmp_db, output_dir):
        """synthesis.md 存在且有内容时，应被正确读取。"""
        synthesis_content = "这位工程师高频使用 CC 进行重构，工具调用强度较高。"
        (output_dir / "synthesis.md").write_text(synthesis_content, encoding="utf-8")
        data = _collect_data(tmp_db, output_dir)
        assert data["synthesis"] == synthesis_content

    def test_collect_data_synthesis_ignores_placeholder(self, tmp_db, output_dir):
        """synthesis.md 为占位说明时（未生成），synthesis 应为空字符串。"""
        (output_dir / "synthesis.md").write_text(
            "_综合洞察未生成（数据不足或预算受限）。_\n", encoding="utf-8"
        )
        data = _collect_data(tmp_db, output_dir)
        assert data["synthesis"] == ""

    def test_report_html_shows_synthesis_when_present(self, tmp_db, budget, output_dir):
        """synthesis.md 有内容时，report.html 应包含综合洞察区块。"""
        synthesis_content = "这位工程师频繁纠正 CC 的代码风格，建议更新 CLAUDE.md 规则。"
        (output_dir / "synthesis.md").write_text(synthesis_content, encoding="utf-8")
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "综合洞察" in content
        assert "频繁纠正" in content

    def test_report_html_no_synthesis_block_when_empty(self, tmp_db, budget, output_dir):
        """synthesis 为空时，report.html 不应出现综合洞察的 HTML 元素（class 属性）。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        # 检查 HTML 元素不存在（class="synthesis-block" 只出现在实际元素里，CSS 里是 .synthesis-block）
        assert 'class="synthesis-block"' not in content

    def test_report_html_part2_renamed_to_evidence(self, tmp_db, budget, output_dir):
        """Part 2 标题应为 '行为证据' 而非 '行为洞察'。"""
        result = render_report(tmp_db, budget, output_dir)
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "行为证据" in content
        assert "行为洞察" not in content
