# CC Mirror — Claude Code 工作指南

## 项目定位

CC Mirror 是一个独立 Python CLI 工具（不混入 Towow）。
从用户的 Claude Code 对话历史中提取**可自动化的模式**，产出：
- **完整报告**（本地 HTML）：CLAUDE.md 规则建议、Skill 建议、工作流模板、思维画像
- **分享卡片**（HTML + PNG）：成就数字、认知画像金句、零敏感信息

**目录**：`~/个人项目/cc-mirror/`
**第一个测试对象**：Towow 项目的 CC 历史（`~/.claude/projects/-Users-nature------Towow/`）

---

## 数据流

```
~/.claude/projects/**/*.jsonl
    ↓ L1 (Python, 零 LLM, 零成本)
mirror.db (SQLite)
    ↓ L2 (代码粗筛 + Sonnet 精筛, 按需调用)
corrections / workflow_clusters / repeated_prompts 表
    ↓ L3 (Opus 聚合, 批量处理)
aggregated insights
    ↓ L4 (Jinja2 + 内联 SVG/JS)
mirror-output/report.html + share-card.html + share-card.png
```

---

## 当前状态

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | L1 解析器 + `cc-mirror scan` 命令 | **✅ 验收通过** (commit e1da72e) |
| Phase 1 | L2 检测 + L3 聚合 + markdown 输出 | **✅ 验收通过** (85/85 测试) |
| Phase 2 | L4 HTML 报告 + 分享卡片 | **✅ 验收通过** (126/126 测试, commit d126674) |
| Phase 3 | 打包发布（pip + uvx） | **✅ 验收通过** (commit b119416) |

**Phase 2 实测数字（Towow 项目，$1.08 总成本）**：
- L3d synthesis: 综合叙事 300 字（"架构师+执行监督者"画像）
- report.html: synthesis → 规则建议 → 重复提示 → 数据总览 → 纠正证据
- share-card.html: 零敏感信息分享卡片（大数字 + 画像金句）
- L1 噪音过滤: repeated prompts 23 → 2（系统消息 + 填充词已过滤）
- 126/126 测试通过

**Phase 1 实测数字（Towow 项目，$1.08 成本）**：
- Corrections confirmed: 2 / 66 candidates（rate 3% — 纠正不多但每条质量高）
- Repeated prompts analyzed: 2 / 2（高信号：tech agent review x4, commit this x4）
- Rules generated: **3 条 CLAUDE.md 规则**（工作流相关：批量操作需确认、状态报告必须引用原始输出）
- Skills suggested: 0（仅 1 聚类，通用调试模式，不值得封装为 Skill）

**Phase 0 实测数字（Towow 项目，72 个 JSONL 文件）**：
- Sessions: 38 | Messages: 62,583 | Tool calls: 9,379
- User text messages: 1,817 | Correction candidates: 66 (**3.6%** of text messages)
- 32 个文件跳过（无有效 session_id，通常是 compact-only 文件）

**GitHub**: https://github.com/NatureBlueee/cc-mirror
**安装**: `uvx --from git+https://github.com/NatureBlueee/cc-mirror cc-mirror`
**下一步**: PyPI 发布（可选），或继续扩展功能

---

## 文件地图

```
src/cc_mirror/
  __init__.py          — 版本
  cli.py               — Click CLI (scan / analyze / suggest-rules)
  db.py                — SQLite schema + init_db / get_or_create_db
  l1_parser.py         — JSONL → SQLite (parse_all_sessions / parse_session)
  l2_correction.py     — 纠正检测 [Phase 1]
  l2_workflow.py       — 工作流聚类 [Phase 1]
  l2_repeated_prompts  — 重复提示 [Phase 1]
  l3_aggregator.py     — LLM 聚合 [Phase 1]
  l4_renderer.py       — HTML 生成 [Phase 2]
  budget.py            — 成本控制 [Phase 1]
  templates/
    report.html.j2     — 完整报告模板 [Phase 2]
    share_card.html.j2 — 分享卡片模板 [Phase 2]

tests/
  fixtures/            — 最小 JSONL 测试数据（不含真实数据）
  test_l1_parser.py    — L1 单元测试
  test_l2_correction.py
  test_l2_workflow.py
  test_l2_repeated_prompts.py
  test_l3_aggregator.py
  test_l4.py           — L4 渲染 26 测试 [Phase 2]

.claude/skills/
  cc-mirror-dev/SKILL.md   — 开发 skill（每 session 开始时加载）
  cc-mirror-lab/SKILL.md   — 最小测试 skill
```

---

## 层间契约（不可随意改动）

### L1 → DB (sessions 表核心字段)
`id, project, start_time, jsonl_path, message_count, cost_usd, has_compact`

### L1 → DB (messages 表核心字段)
`uuid, session_id, type, timestamp, user_text, assistant_text, tool_names (JSON), is_candidate_correction, sequence_num`

### L2 → DB (corrections 表)
`session_id, project, user_message_uuid, cc_did, user_wanted, correction_type, is_generalizable, confidence`

### L2 → DB (workflow_clusters 表)
`tool_sequence_pattern (JSON), session_ids (JSON), description, skill_suggestion (JSON)`

### CLI 契约
```bash
cc-mirror scan   [--claude-dir ~/.claude] [--output mirror.db] [--project PROJ] [--verbose]
cc-mirror analyze [--claude-dir] [--output dir] [--budget 20.0] [--parallelism 20]
cc-mirror suggest-rules [--claude-dir] [--output rules.md]
```

---

## 开发哲学

### 自己是第一用户
测试 = 在真实 CC 数据上运行，看输出对不对。不是跑 unit test matrix。
Phase 0 验收：`cc-mirror scan` 输出合理的统计数字（session 数、候选纠正率 10-30%）

### 最小测试
- 不要每次测试 full history
- 用 `tests/fixtures/` 中的 10 条消息 JSONL 做快速单元验证
- 集成测试：对 Towow 数据库的一个 project 运行

### 上下文管理原则
1. 每 session 开始：Read CLAUDE.md + 当前 phase 的关键文件
2. 每 session 结束：更新 CLAUDE.md 的"当前状态"表格
3. git commit = phase 边界，commit message 要能让下个 session 理解状态
4. 不要在开发 session 中加载完整 PRD（太大）

### Agent Teams 策略
```
Phase 0: 单 agent (L1 + scan)
Phase 1: 3 agent 并行
  A: l2_correction.py（纠正检测）
  B: l2_workflow.py + l2_repeated_prompts.py（工作流+提示）
  C: tests（每个模块的最小测试）
  顺序: A+B+C → l3_aggregator.py（依赖 A+B 产出）
Phase 2: 2 agent 并行
  A: l4_renderer.py + report.html.j2
  B: share_card.html.j2 + PNG 生成
```

### 每个阶段结束必须做
1. `cc-mirror scan` / `cc-mirror analyze` 在 Towow 数据上运行，把输出给用户看
2. git commit（记录阶段边界）
3. 更新 CLAUDE.md 的"当前状态"

---

## 硬性约束

- 所有分析在本机运行，不上传数据
- LLM 调用通过用户自己的 API key
- L1 必须能处理损坏的 JSONL（try/except，记录但不中断）
- 每次 LLM 调用都记录到 llm_calls 表
- Budget 超限时降级（P0 only），不 crash
- 不提交 *.jsonl 和 mirror.db 到 git

---

## 打包目标（Phase 3）

```bash
# 安装方式 1（pip）
pip install cc-mirror
cc-mirror analyze

# 安装方式 2（uvx，无需安装）
uvx cc-mirror analyze

# 安装方式 3（Claude Skill）
# 添加到 ~/.claude/settings.json → /cc-mirror 触发
```

---

## Skill 加载规则

| 工作类型 | 加载 Skill |
|---------|-----------|
| 开发代码 | `/cc-mirror-dev` |
| 测试/验证 | `/cc-mirror-lab` |
| 写文档/README | 直接工作，不需要 skill |

---

## 关联文档

- PRD: 完整需求文档（不要在开发 session 加载，太大）
- `docs/DESIGN.md`: 架构决策（每个 session 都应该读）
- `tests/fixtures/sample_session.jsonl`: 测试用最小数据
