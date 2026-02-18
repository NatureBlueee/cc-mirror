# CC Mirror

> 从你与 Claude Code 的全部协作历史中，提取可自动化的模式、可复用的思维、可固化的流程。

**一次性深度分析，两种产出**：
- **完整报告**（本地 HTML）：CLAUDE.md 规则建议、Skill 建议、工作流模板、思维画像
- **分享卡片**（HTML + PNG）：AI 协作成就，零隐私泄露，可发朋友圈

---

## 快速开始

```bash
# 无需安装（推荐）
uvx cc-mirror scan

# 或 pip 安装
pip install cc-mirror
cc-mirror scan
```

```bash
# 先扫描，免费预览数据规模
cc-mirror scan --claude-dir ~/.claude

# 完整分析（调用 LLM，~$10-20）
cc-mirror analyze --claude-dir ~/.claude --output ./mirror-output/

# 只生成 CLAUDE.md 规则建议
cc-mirror suggest-rules --claude-dir ~/.claude
```

---

## 它做什么

从你的 `~/.claude/projects/**/*.jsonl` 历史中提取：

**P0（务实产出）**
- CLAUDE.md 规则建议：你反复纠正 CC 的模式 → 自动生成规则
- Skill 建议：你重复执行的工作流 → 建议创建/更新 Skill
- 重复提示消除：你每次都手动说的话 → 告诉你一次性写入 CLAUDE.md

**P1（思维提取）**
- 问题分解模式、决策框架、元认知特征

**P2（成长认知）**
- 纠正率趋势、工具使用演化、效率曲线

---

## 隐私

- **所有分析在本机完成**，不上传任何数据
- LLM 调用使用你自己的 API key（`ANTHROPIC_API_KEY`）
- 分享卡片经过严格脱敏，不包含项目名、代码片段、具体内容

---

## 成本估算

| 用户规模 | session 数 | 估算成本 |
|----------|-----------|---------|
| 轻度 | ~50 | $2-5 |
| 中度 | ~200 | $6-12 |
| 重度 | ~500 | $10-20 |

`cc-mirror scan`（L1 扫描）完全免费，先预览再决定是否做完整分析。

---

## 开发状态

- [x] Phase 0：L1 解析 + scan 命令
- [ ] Phase 1：L2 检测 + L3 聚合（纠正/工作流/规则建议）
- [ ] Phase 2：L4 HTML 报告 + 分享卡片
- [ ] Phase 3：pip 发布 + uvx 支持

---

## License

MIT
