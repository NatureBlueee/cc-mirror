"""
l4_renderer.py — CC Mirror L4 HTML 报告渲染器

职责：
  - 从 DB 读取所有分析数据
  - 从 output_dir 读取 L3 生成的 suggested-rules.md / suggested-skills.md
  - 用 Jinja2 渲染 report.html 和 share-card.html
  - 如果 playwright 可用，截图生成 share-card.png

主接口：render_report(db, budget, output_dir) -> dict
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cc_mirror.budget import BudgetController

# Jinja2 延迟导入（可选依赖）
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    _jinja2_available = True
except ImportError:
    _jinja2_available = False

# playwright 延迟导入（可选依赖）
try:
    from playwright.sync_api import sync_playwright
    _playwright_available = True
except ImportError:
    _playwright_available = False


# ---------------------------------------------------------------------------
# 内部：数据收集
# ---------------------------------------------------------------------------

def _safe_query(db: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    """安全执行查询，出错返回空列表。"""
    try:
        cur = db.execute(sql, params)
        return cur.fetchall()
    except Exception as e:
        print(f"[l4] 查询失败: {e} | SQL: {sql[:80]}", file=sys.stderr)
        return []


def _safe_query_one(db: sqlite3.Connection, sql: str, params: tuple = (), default=0):
    """安全执行单行查询，出错返回 default。"""
    try:
        cur = db.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else default
    except Exception as e:
        print(f"[l4] 查询失败: {e}", file=sys.stderr)
        return default


def _collect_data(db: sqlite3.Connection, output_dir: Path) -> dict:
    """
    从 DB 和 output_dir 收集所有渲染所需数据。

    Returns:
        包含报告所需全部字段的 dict。
    """
    output_dir = Path(output_dir)

    # ---- 核心数字 ----
    sessions = _safe_query_one(db, "SELECT COUNT(*) FROM sessions")
    messages = _safe_query_one(db, "SELECT COUNT(*) FROM messages")
    tool_calls = _safe_query_one(db, "SELECT COUNT(*) FROM tool_calls")
    correction_candidates = _safe_query_one(
        db, "SELECT COUNT(*) FROM messages WHERE is_candidate_correction = 1"
    )
    corrections_confirmed = _safe_query_one(db, "SELECT COUNT(*) FROM corrections")
    cost_usd = _safe_query_one(
        db, "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls", default=0.0
    )
    llm_calls = _safe_query_one(db, "SELECT COUNT(*) FROM llm_calls")

    # ---- 纠正详情 ----
    raw_corrections = _safe_query(
        db,
        """
        SELECT cc_did, user_wanted, correction_type, confidence
        FROM corrections
        ORDER BY confidence DESC
        """,
    )
    corrections = [
        {
            "cc_did": row["cc_did"] or "",
            "user_wanted": row["user_wanted"] or "",
            "correction_type": row["correction_type"] or "other",
            "confidence": round(float(row["confidence"] or 0), 2),
        }
        for row in raw_corrections
    ]

    # ---- 重复提示（只取有建议且 suggestion_location != "none" 的）----
    # repeated_prompts 表可能没有 analysis_json / suggestion_location 列（旧 schema）
    # 先检查列是否存在
    try:
        cols_query = db.execute("PRAGMA table_info(repeated_prompts)")
        col_names = {row[1] for row in cols_query.fetchall()}
    except Exception:
        col_names = set()

    has_analysis = "analysis_json" in col_names

    if has_analysis:
        raw_rp = _safe_query(
            db,
            """
            SELECT canonical_text, occurrences, analysis_json
            FROM repeated_prompts
            WHERE analysis_json IS NOT NULL
              AND JSON_EXTRACT(analysis_json, '$.suggestion_location') != 'none'
              AND JSON_EXTRACT(analysis_json, '$.suggestion_location') IS NOT NULL
            ORDER BY occurrences DESC
            """,
        )
        repeated_prompts = []
        for row in raw_rp:
            try:
                aj = json.loads(row["analysis_json"]) if row["analysis_json"] else {}
            except Exception:
                aj = {}
            location = aj.get("suggestion_location", "")
            reason = aj.get("reason", "")
            suggestion = aj.get("suggestion", "")
            repeated_prompts.append({
                "text": (row["canonical_text"] or "")[:120],
                "occurrences": row["occurrences"],
                "reason": reason,
                "location": location,
                "suggestion": suggestion,
            })
    else:
        # 旧 schema：没有 analysis_json，直接展示所有重复提示
        raw_rp = _safe_query(
            db,
            "SELECT canonical_text, occurrences FROM repeated_prompts ORDER BY occurrences DESC",
        )
        repeated_prompts = [
            {
                "text": (row["canonical_text"] or "")[:120],
                "occurrences": row["occurrences"],
                "reason": "",
                "location": "",
                "suggestion": "",
            }
            for row in raw_rp
        ]

    # ---- 规则 markdown（从文件读）----
    rules_md = ""
    rules_file = output_dir / "suggested-rules.md"
    if rules_file.exists():
        try:
            rules_md = rules_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[l4] 读取 suggested-rules.md 失败: {e}", file=sys.stderr)

    # ---- 技能建议 markdown（从文件读）----
    skills_md = ""
    skills_file = output_dir / "suggested-skills.md"
    if skills_file.exists():
        try:
            skills_md = skills_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[l4] 读取 suggested-skills.md 失败: {e}", file=sys.stderr)

    # ---- 综合洞察（从文件读）----
    synthesis = ""
    synthesis_file = output_dir / "synthesis.md"
    if synthesis_file.exists():
        try:
            content = synthesis_file.read_text(encoding="utf-8").strip()
            # 过滤掉占位说明（未生成时写入的空白占位）
            if content and not content.startswith("_综合洞察未生成"):
                synthesis = content
        except Exception as e:
            print(f"[l4] 读取 synthesis.md 失败: {e}", file=sys.stderr)

    # ---- 工作流聚类 ----
    raw_clusters = _safe_query(
        db,
        """
        SELECT description, session_count
        FROM workflow_clusters
        ORDER BY session_count DESC
        """,
    )
    workflow_clusters = [
        {
            "description": row["description"] or "(无描述)",
            "session_count": row["session_count"],
        }
        for row in raw_clusters
    ]

    # ---- 时间信息 ----
    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "sessions": sessions,
        "messages": messages,
        "tool_calls": tool_calls,
        "correction_candidates": correction_candidates,
        "corrections_confirmed": corrections_confirmed,
        "cost_usd": float(cost_usd),
        "llm_calls": llm_calls,
        "corrections": corrections,
        "repeated_prompts": repeated_prompts,
        "rules_md": rules_md,
        "skills_md": skills_md,
        "workflow_clusters": workflow_clusters,
        "generated_at": generated_at,
        "synthesis": synthesis,
    }


# ---------------------------------------------------------------------------
# 内部：Playwright 截图
# ---------------------------------------------------------------------------

def _take_screenshot(html_path: Path, png_path: Path) -> bool:
    """
    用 Playwright 截图 share-card.html → share-card.png。

    Returns:
        True 表示成功，False 表示失败（不抛异常）。
    """
    if not _playwright_available:
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 630})
            page.goto(f"file://{html_path.absolute()}")
            page.wait_for_timeout(500)  # 等待字体/渲染
            page.screenshot(path=str(png_path), full_page=False)
            browser.close()
        return True
    except Exception as e:
        print(f"[l4] playwright 截图失败（非致命）: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def render_report(
    db: sqlite3.Connection,
    budget: "BudgetController",
    output_dir: Path,
) -> dict:
    """
    L4 渲染层：从 DB 读取数据，渲染 HTML 报告和分享卡片。

    Args:
        db:         已初始化的 SQLite 连接
        budget:     预算控制器（本层不调用 LLM，仅读取成本信息）
        output_dir: 输出目录（L3 已写入 suggested-rules.md 等文件）

    Returns:
        {
            "report_path": str,        # report.html 的绝对路径
            "share_card_path": str,    # share-card.html 的绝对路径
            "png_path": str | None,    # share-card.png（如果 playwright 可用）
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "report_path": "",
        "share_card_path": "",
        "png_path": None,
    }

    if not _jinja2_available:
        print("[l4] jinja2 未安装，跳过 HTML 报告生成（pip install jinja2）", file=sys.stderr)
        return result

    # ---- 收集数据 ----
    data = _collect_data(db, output_dir)
    print(
        f"[l4] 数据收集完成：sessions={data['sessions']}, "
        f"corrections={data['corrections_confirmed']}, "
        f"repeated_prompts={len(data['repeated_prompts'])}",
        file=sys.stderr,
    )

    # ---- 初始化 Jinja2 环境 ----
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )
    # 添加自定义过滤器
    env.filters["format_number"] = lambda v: f"{v:,}" if isinstance(v, int) else str(v)
    env.filters["format_cost"] = lambda v: f"${float(v):.4f}"

    # ---- 渲染 report.html ----
    try:
        tpl = env.get_template("report.html.j2")
        html_content = tpl.render(**data)
        report_path = output_dir / "report.html"
        report_path.write_text(html_content, encoding="utf-8")
        result["report_path"] = str(report_path.absolute())
        print(f"[l4] report.html 已生成：{report_path}", file=sys.stderr)
    except Exception as e:
        print(f"[l4] 渲染 report.html 失败: {e}", file=sys.stderr)

    # ---- 渲染 share-card.html ----
    try:
        tpl = env.get_template("share_card.html.j2")
        card_content = tpl.render(**data)
        card_path = output_dir / "share-card.html"
        card_path.write_text(card_content, encoding="utf-8")
        result["share_card_path"] = str(card_path.absolute())
        print(f"[l4] share-card.html 已生成：{card_path}", file=sys.stderr)
    except Exception as e:
        print(f"[l4] 渲染 share-card.html 失败: {e}", file=sys.stderr)

    # ---- 可选：Playwright 截图 ----
    if result["share_card_path"]:
        png_path = output_dir / "share-card.png"
        success = _take_screenshot(Path(result["share_card_path"]), png_path)
        if success:
            result["png_path"] = str(png_path.absolute())
            print(f"[l4] share-card.png 已生成：{png_path}", file=sys.stderr)

    return result
