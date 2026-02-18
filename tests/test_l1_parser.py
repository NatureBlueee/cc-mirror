"""最小测试：L1 解析器 smoke test。

哲学：把自己当第一用户。验证基本行为，不追求 100% 覆盖。
"""
import json
import sqlite3
import tempfile
from pathlib import Path
import sys

# 把 src 加到 path（在没 pip install 时可以直接跑）
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cc_mirror.db import get_or_create_db
from cc_mirror.l1_parser import parse_all_sessions, is_confirmation_only


FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.jsonl"


def make_fake_claude_dir(tmp_path: Path, session_jsonl: Path) -> Path:
    """创建一个假的 ~/.claude/projects 结构。"""
    project_dir = tmp_path / "projects" / "-Users-test------myproject"
    project_dir.mkdir(parents=True)
    # 复制 fixture 到 project 目录
    target = project_dir / "session-test-001.jsonl"
    target.write_bytes(session_jsonl.read_bytes())
    return tmp_path


def test_basic_parse():
    """基础解析：fixture 文件能被解析，没有崩溃。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        claude_dir = make_fake_claude_dir(tmp_path, FIXTURE)
        db_path = tmp_path / "test.db"

        db = get_or_create_db(db_path)
        stats = parse_all_sessions(claude_dir, db)
        db.close()

        assert stats["sessions"] >= 1, f"应该有至少 1 个 session，得到 {stats['sessions']}"
        assert stats["messages"] >= 10, f"应该有至少 10 条消息，得到 {stats['messages']}"
        assert stats["parse_errors"] == 0, f"不应该有解析错误，得到 {stats['parse_errors']}"


def test_correction_candidate_detected():
    """候选纠正检测：fixture 中的纠正消息（msg-005）应该被标记为候选。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        claude_dir = make_fake_claude_dir(tmp_path, FIXTURE)
        db_path = tmp_path / "test.db"

        db = get_or_create_db(db_path)
        parse_all_sessions(claude_dir, db)

        # 检查候选纠正
        rows = db.execute(
            "SELECT uuid, is_candidate_correction, user_text FROM messages "
            "WHERE is_candidate_correction = 1"
        ).fetchall()

        db.close()

        # fixture 中 msg-005 是纠正候选（"不对，我们项目用 named export..."）
        assert len(rows) >= 1, f"至少应该有 1 条纠正候选，得到 {len(rows)}"
        texts = [r[2] for r in rows]
        assert any("named export" in t for t in texts), \
            f"应该包含 named export 纠正，实际候选: {texts}"


def test_confirmation_not_flagged():
    """确认消息不应该被标记为候选纠正。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        claude_dir = make_fake_claude_dir(tmp_path, FIXTURE)
        db_path = tmp_path / "test.db"

        db = get_or_create_db(db_path)
        parse_all_sessions(claude_dir, db)

        # "好的，继续" 不应该是候选纠正
        rows = db.execute(
            "SELECT uuid, is_candidate_correction, user_text FROM messages "
            "WHERE user_text LIKE '%好的，继续%'"
        ).fetchall()

        db.close()

        for row in rows:
            assert not row[1], \
                f"'好的，继续' 不应该被标记为候选纠正，但被标记了: {row[2]}"


def test_tool_calls_stored():
    """工具调用应该被存入 tool_calls 表。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        claude_dir = make_fake_claude_dir(tmp_path, FIXTURE)
        db_path = tmp_path / "test.db"

        db = get_or_create_db(db_path)
        parse_all_sessions(claude_dir, db)

        count = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        db.close()

        # fixture 有 Bash + Write + Write + Bash = 4 次工具调用
        assert count >= 3, f"应该有至少 3 次工具调用，得到 {count}"


def test_incremental_skip():
    """增量运行：第二次 parse 应该跳过已处理的 session。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        claude_dir = make_fake_claude_dir(tmp_path, FIXTURE)
        db_path = tmp_path / "test.db"

        db = get_or_create_db(db_path)
        stats1 = parse_all_sessions(claude_dir, db)
        db.close()

        db = get_or_create_db(db_path)
        stats2 = parse_all_sessions(claude_dir, db)
        db.close()

        # 第二次 sessions 应该是 0（全部跳过），skipped_sessions 应该 > 0
        assert stats2["sessions"] == 0, \
            f"第二次运行不应该有新 session，得到 {stats2['sessions']}"
        assert stats2.get("skipped_sessions", 0) >= 1, \
            f"第二次运行应该有跳过的 session，得到 {stats2.get('skipped_sessions')}"


class TestIsConfirmationOnly:
    """is_confirmation_only 的单元测试。"""

    def test_confirmations(self):
        cases = ["好", "好的", "ok", "OK", "继续", "可以", "yes", "y", "行"]
        for text in cases:
            assert is_confirmation_only(text), f"'{text}' 应该是纯确认"

    def test_not_confirmations(self):
        cases = [
            "不对，应该用 named export",
            "我说的是重构不是重写",
            "这里有个 bug，应该是 === 不是 ==",
            "继续但是要注意这里的类型",  # 虽然有"继续"但有其他内容
        ]
        for text in cases:
            assert not is_confirmation_only(text), f"'{text}' 不应该是纯确认"


if __name__ == "__main__":
    # 可以直接运行：python tests/test_l1_parser.py
    import pytest
    pytest.main([__file__, "-v"])
