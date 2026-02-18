---
name: cc-mirror-dev
description: CC Mirror 开发 skill。每个开发 session 开始时加载——提供项目上下文、开发哲学、当前状态、层间契约。避免每次从零读取 PRD。
---

# CC Mirror 开发 Skill

## 我是谁

我是 CC Mirror 项目的开发上下文加载器。

CC Mirror 是一个独立 Python CLI 工具，分析用户的 Claude Code 对话历史（`~/.claude/projects/**/*.jsonl`），提取可自动化模式，产出 HTML 报告 + 分享卡片。

**项目目录**：`~/个人项目/cc-mirror/`
**第一测试对象**：Towow 项目 CC 历史

---

## 快速上下文

### 数据流（四层管线）

```
JSONL 文件 → L1(代码,零LLM) → SQLite → L2(Sonnet精筛) →
corrections/clusters表 → L3(Opus聚合) → L4(Jinja2+SVG) → HTML报告
```

### 当前进度
读 CLAUDE.md 的"当前状态"表格获取最新进度。

### 关键文件
```
src/cc_mirror/
  db.py          ← SQLite schema + init
  l1_parser.py   ← JSONL → DB（L1，无 LLM）
  cli.py         ← Click CLI（scan/analyze/suggest-rules）
  l2_*.py        ← 纠正/工作流/重复提示检测（L2）[Phase 1]
  l3_aggregator  ← LLM 聚合（L3）[Phase 1]
  l4_renderer    ← HTML 输出（L4）[Phase 2]
```

---

## 开发哲学（核心约束）

### 1. 自己是第一用户
测试 ≠ 跑 unit test suite。测试 = 在真实 CC 数据上运行，看输出合不合理。
每个 phase 结束必须把实际输出展示给用户。

### 2. 最小测试原则
- `tests/fixtures/` 中的最小 JSONL（~10 条消息）用于快速 smoke test
- 集成测试：对 `~/.claude/projects/-Users-nature------Towow/` 一个 session 运行
- 不要在 session 里跑 full history（太慢，上下文会爆）

### 3. 工具优先于实现
在实现新 L 层之前，先确认 schema 正确（db.py）。
如果需要写 prompt，先在独立消息中测试 prompt 效果。

### 4. 上下文自管理
- session 开始：Read CLAUDE.md + docs/DESIGN.md
- session 结束：更新 CLAUDE.md 的"当前状态"表，git commit
- 不要在开发 session 中加载 PRD（太大，浪费上下文）

---

## 层间契约（改动需全层同步）

### sessions 表（L1 产出，L2 消费）
```sql
id TEXT PRIMARY KEY,     -- session UUID
project TEXT NOT NULL,   -- 项目编码目录名
start_time TEXT,         -- ISO 8601
jsonl_path TEXT NOT NULL -- 原始文件路径（用于 debug）
message_count INTEGER
cost_usd REAL
has_compact BOOLEAN
```

### messages 表（L1 产出，L2 消费）
```sql
uuid TEXT PRIMARY KEY,
session_id TEXT,
type TEXT,               -- user/assistant/system/summary
user_text TEXT,          -- 用户输入全文
assistant_text TEXT,     -- assistant text 部分
tool_names TEXT,         -- JSON array
is_candidate_correction BOOLEAN,
sequence_num INTEGER
```

### corrections 表（L2 产出，L3 消费）
```sql
cc_did TEXT,             -- CC 做了什么（一句话）
user_wanted TEXT,        -- 用户想要什么（一句话）
correction_type TEXT,    -- misunderstanding/wrong_approach/style/scope/knowledge
is_generalizable BOOLEAN -- 是否可规则化
confidence REAL          -- 0-1
```

---

## Agent Team 模式

### Phase 1 并行配置
```
Agent A: l2_correction.py
  输入: messages 表中 is_candidate_correction=TRUE 的记录
  输出: corrections 表

Agent B: l2_workflow.py + l2_repeated_prompts.py
  输入: tool_calls 表（工具序列）+ messages 表（短用户消息）
  输出: workflow_clusters 表 + repeated_prompts 表

Agent C: tests
  输入: fixtures/sample_session.jsonl
  输出: 每个模块的 smoke test 通过

→ 等 A+B 完成后：
Agent D: l3_aggregator.py（依赖 corrections 和 workflow_clusters）
```

### 接缝审查（并行 Agent 最容易出错的地方）
- A 写 corrections 表，D 读 corrections 表 → 字段名必须一致
- B 写 workflow_clusters 表，D 读 workflow_clusters 表 → skill_suggestion JSON 格式要对齐
- 集成时：沿数据流逐段验证，不只看编译通过

---

## 常见陷阱

### JSONL 解析
- `content` 字段可以是 `str` 或 `list`（都要处理）
- `isSidechain: true` 的消息要跳过
- 文件名以 `agent-` 开头的 JSONL 是子 agent，跳过
- 每行单独 try/except，解析失败记录但不中断

### LLM 调用
- 所有 LLM 调用必须记录到 `llm_calls` 表（stage, model, prompt_hash, cost）
- temperature=0 保证确定性
- 用 JSON format 强制输出，但必须有 fallback（model 可能不返回 valid JSON）

### 报告生成
- HTML 报告是单文件，内联所有 CSS/JS/SVG
- `file://` 协议下 `navigator.clipboard.writeText()` 不可用
  → 用 `document.execCommand('copy')` fallback

---

## 我的工作节奏

开始一个 session 时：
1. 读 CLAUDE.md 当前状态
2. 读 docs/DESIGN.md 确认架构决策
3. 实现当前 phase 的下一个文件
4. 运行 smoke test（fixtures 数据）
5. 更新 CLAUDE.md + git commit

结束一个 session 时：
- CLAUDE.md 的"当前状态"一定要更新
- git commit message 要能让下个 session 理解到哪了
