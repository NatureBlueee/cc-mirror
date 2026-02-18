#!/usr/bin/env python3
"""CC Mirror CLI"""
import asyncio
import click
from pathlib import Path


@click.group()
@click.version_option()
def cli():
    """CC Mirror — 从你的 Claude Code 历史中提取可自动化的模式。"""
    pass


@cli.command()
@click.option("--claude-dir", default="~/.claude", show_default=True,
              help="Claude Code 数据目录")
@click.option("--output", default="./mirror.db", show_default=True,
              help="SQLite 数据库输出路径")
@click.option("--project", default=None, help="只分析特定项目（项目编码目录名）")
@click.option("--verbose", "-v", is_flag=True, help="显示详细进度")
def scan(claude_dir, output, project, verbose):
    """
    L1 扫描：解析 CC 历史 → SQLite（免费，无 LLM）。

    输出数据概况，告诉你有多少 sessions、候选纠正、重复提示。
    用这个命令先了解你的数据规模，再决定是否运行完整分析。

    示例：

        cc-mirror scan

        cc-mirror scan --claude-dir ~/.claude --project -Users-name------myproject

        cc-mirror scan --verbose
    """
    # 导入在这里做（避免 CLI 启动时加载所有模块）
    from cc_mirror.db import get_or_create_db
    from cc_mirror.l1_parser import parse_all_sessions

    claude_path = Path(claude_dir).expanduser()
    output_path = Path(output).expanduser()

    if not claude_path.exists():
        click.echo(f"找不到 Claude 数据目录：{claude_path}", err=True)
        raise click.Abort()

    click.echo(f"CC Mirror scan — {claude_path}")
    click.echo(f"数据库：{output_path}")
    if project:
        click.echo(f"项目过滤：{project}")
    click.echo()

    # 初始化 DB
    db = get_or_create_db(output_path)

    # 运行 L1 解析
    if verbose:
        click.echo("正在解析 JSONL 文件...")

    stats = parse_all_sessions(claude_path, db, project_filter=project)

    db.close()

    # 输出统计
    click.echo("=" * 50)
    click.echo("CC Mirror scan complete.")
    click.echo()
    click.echo(f"Sessions:             {stats['sessions']:,} (across {len(stats['projects'])} projects)")
    click.echo(f"Messages:             {stats['messages']:,}")
    click.echo(f"Tool calls:           {stats.get('tool_calls', 0):,}")
    click.echo(f"User text messages:   {stats.get('user_text_messages', 0):,}")
    click.echo()
    user_text_msgs = max(stats.get('user_text_messages', 1), 1)
    correction_rate = stats['candidate_corrections'] / user_text_msgs * 100
    click.echo(f"Candidate corrections: {stats['candidate_corrections']:,} "
               f"({correction_rate:.1f}% of user text messages)")
    click.echo(f"Repeated prompts:     {stats['repeated_prompts']:,} unique patterns")
    if stats.get('parse_errors', 0) > 0:
        click.echo(f"Parse errors:        {stats['parse_errors']:,} lines skipped", err=True)
    if stats.get('skipped_sessions', 0) > 0:
        click.echo(f"Skipped (cached):    {stats['skipped_sessions']:,} sessions already in DB")
    click.echo()
    click.echo(f"Database saved to:   {output_path}")
    click.echo()
    click.echo("Run `cc-mirror analyze` to proceed with full analysis.")


@cli.command()
@click.option("--claude-dir", default="~/.claude", show_default=True,
              help="Claude Code 数据目录")
@click.option("--output", default="./mirror-output/", show_default=True,
              help="报告输出目录")
@click.option("--budget", default=20.0, show_default=True,
              help="最大 LLM 成本（USD）")
@click.option("--parallelism", default=20, show_default=True,
              help="L2 并发数")
@click.option("--project", default=None,
              help="只分析特定项目")
@click.option("--skip-scan", is_flag=True,
              help="跳过 L1 扫描（使用现有 mirror.db）")
@click.option("--db", "db_path", default="./mirror.db", show_default=True,
              help="SQLite 数据库路径")
@click.option("--verbose", "-v", is_flag=True,
              help="显示详细进度")
def analyze(claude_dir, output, budget, parallelism, project, skip_scan, db_path, verbose):
    """完整分析：L1 扫描 → L2 检测 → L3 聚合 → 生成报告。"""
    # 延迟导入：避免 CLI 启动时加载所有模块
    from cc_mirror.db import get_or_create_db
    from cc_mirror.budget import BudgetController
    from cc_mirror.l2_correction import run_l2_corrections
    from cc_mirror.l2_workflow import run_l2_workflow
    from cc_mirror.l2_repeated_prompts import run_l2_repeated_prompts
    from cc_mirror.l3_aggregator import run_l3

    claude_path = Path(claude_dir).expanduser()
    output_path = Path(output).expanduser()
    db_file = Path(db_path).expanduser()

    click.echo(f"CC Mirror analyze — budget=${budget:.2f}")
    click.echo(f"输出目录：{output_path}")
    click.echo()

    # ---- L1: 扫描（可选跳过）----
    if not skip_scan:
        if not claude_path.exists():
            click.echo(f"找不到 Claude 数据目录：{claude_path}", err=True)
            raise click.Abort()

        from cc_mirror.l1_parser import parse_all_sessions

        db = get_or_create_db(db_file)
        l1_stats = parse_all_sessions(claude_path, db, project_filter=project)

        session_count = l1_stats.get("sessions", 0)
        msg_count = l1_stats.get("messages", 0)
        click.echo(
            f"[L1] Scan...          "
            f"{session_count} sessions, {msg_count:,} messages"
        )
    else:
        # 直接打开已有 DB
        if not db_file.exists():
            click.echo(f"找不到数据库文件：{db_file}（请先运行 scan 或去掉 --skip-scan）", err=True)
            raise click.Abort()

        db = get_or_create_db(db_file)
        click.echo(f"[L1] Scan...          skipped (using {db_file})")

    # 初始化预算控制器
    budget_ctrl = BudgetController(budget_usd=budget, db=db)

    # ---- L2a: 纠正检测 ----
    l2_corr_stats = asyncio.run(
        run_l2_corrections(db, budget_ctrl, parallelism=parallelism)
    )
    candidates_processed = l2_corr_stats.get("processed", 0) + l2_corr_stats.get("skipped", 0)
    corrections_found = l2_corr_stats.get("corrections_found", 0)
    click.echo(
        f"[L2] Corrections...   "
        f"{candidates_processed} candidates -> {corrections_found} confirmed"
    )

    # ---- L2b: 工作流聚类 ----
    l2_wf_stats = run_l2_workflow(db, budget_ctrl)
    clusters_found = l2_wf_stats.get("clusters_found", 0)
    click.echo(
        f"[L2] Workflows...     "
        f"{clusters_found} clusters found"
    )

    # ---- L2c: 重复提示 ----
    l2_rp_stats = run_l2_repeated_prompts(db, budget_ctrl)
    rp_analyzed = l2_rp_stats.get("analyzed", 0)
    click.echo(
        f"[L2] Repeated...      "
        f"{rp_analyzed} analyzed"
    )

    # ---- L3: 聚合生成报告 ----
    l3_stats = run_l3(db, budget_ctrl, output_path)
    click.echo(
        f"[L3] Aggregating...   "
        f"report saved to {output_path}"
    )

    db.close()

    # ---- 最终输出 ----
    click.echo()
    click.echo("=" * 50)
    click.echo("CC Mirror analyze complete.")
    click.echo()
    click.echo(f"Rules generated:      {l3_stats.get('rules_generated', 0)}")
    click.echo(f"Skills suggested:     {l3_stats.get('skills_suggested', 0)}")
    click.echo(f"Repeated prompts:     {l3_stats.get('repeated_prompts_addressed', 0)}")
    click.echo(f"LLM calls (L3):       {l3_stats.get('llm_calls', 0)}")
    click.echo(f"Budget spent:         ${budget_ctrl.spent_usd:.4f} / ${budget:.2f}")
    click.echo()
    click.echo("Output files:")
    for fpath in l3_stats.get("output_files", []):
        click.echo(f"  {fpath}")
    click.echo()


@cli.command("suggest-rules")
@click.option("--claude-dir", default="~/.claude", show_default=True)
@click.option("--output", default="./suggested-rules.md", show_default=True)
def suggest_rules(claude_dir, output):
    """只输出 CLAUDE.md 规则建议。[Coming Soon - Phase 1]"""
    click.echo("suggest-rules 命令还在开发中（Phase 1）。")


def main():
    cli()


if __name__ == "__main__":
    main()
