#!/usr/bin/env python3
"""CC Mirror CLI"""
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
        click.echo(f"❌ 找不到 Claude 数据目录：{claude_path}", err=True)
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
@click.option("--claude-dir", default="~/.claude", show_default=True)
@click.option("--output", default="./mirror-output/", show_default=True)
@click.option("--budget", default=20.0, show_default=True, help="最大 API 成本（USD）")
def analyze(claude_dir, output, budget):
    """完整分析：L1 + L2 + L3 + L4（调用 LLM）。[Coming Soon - Phase 1]"""
    click.echo("⚠️  analyze 命令还在开发中（Phase 1）。请先用 `cc-mirror scan` 验证数据。")


@cli.command("suggest-rules")
@click.option("--claude-dir", default="~/.claude", show_default=True)
@click.option("--output", default="./suggested-rules.md", show_default=True)
def suggest_rules(claude_dir, output):
    """只输出 CLAUDE.md 规则建议。[Coming Soon - Phase 1]"""
    click.echo("⚠️  suggest-rules 命令还在开发中（Phase 1）。")


def main():
    cli()


if __name__ == "__main__":
    main()
