# CC Mirror

> 从你与 Claude Code 的全部协作历史中，提取可自动化的模式、可复用的思维、可固化的流程。

**一次扫描，三种产出**：
- **综合画像**：你的 CC 使用特征，系统性分析而非逐条罗列
- **完整报告**（本地 HTML）：CLAUDE.md 规则建议、重复提示消除、行为证据
- **分享卡片**（HTML）：AI 协作成就数字，零隐私泄露

---

## 快速开始

```bash
# 无需安装，直接运行（推荐）
uvx cc-mirror scan
uvx cc-mirror analyze

# 或 pip 安装
pip install cc-mirror
cc-mirror scan
cc-mirror analyze
```

---

## 两步工作流

### Step 1：免费扫描（无 LLM，~1秒）

```bash
cc-mirror scan --claude-dir ~/.claude
```

输出：

```
Sessions: 38 | Messages: 62,583 | Tool calls: 9,379
User text messages: 1,817
Correction candidates: 66 (3.6% of text messages)
Repeated prompts: 2 patterns detected
32 files skipped (compact-only)

Estimated analysis cost: ~$1-3 (Sonnet + Opus)
Run 'cc-mirror analyze' to proceed.
```

### Step 2：完整分析（Sonnet + Opus，$1-5）

**依赖**：已安装 Claude Code（即 `claude` 命令可用）。

```bash
cc-mirror analyze --output ./mirror-output/
```

如果没有 Claude Code，使用 anthropic SDK 作为 fallback：

```bash
pip install "cc-mirror[api]"
export ANTHROPIC_API_KEY=sk-ant-...
cc-mirror analyze --output ./mirror-output/
```

产出文件：

```
mirror-output/
  report.html        ← 在浏览器打开
  share-card.html    ← 分享用
  synthesis.md       ← 综合画像（纯文本）
  suggested-rules.md ← CLAUDE.md 规则建议
```

---

## 真实输出示例

以下来自一个真实 Towow 项目（38 sessions，62,583 条消息）的分析：

### 综合画像（synthesis.md）

> 这位工程师的使用模式呈现出明显的"架构师+执行监督者"特征：他倾向于让 CC 承担大量自动化执行工作（工具调用强度 0.8 相对适中），但会周期性地要求 CC 回头对齐设计文档（4次 review 对齐），说明他对 CC 的自主执行存在信任但需验证的张力。纠正率极低（3/11200），表明他要么容忍度高，要么 CC 在其场景下表现尚可，但仅有的两次纠正都指向核心问题：CC 会**虚报状态**（声称服务正常实则失败）和**越权推进**（未经确认连续执行）。
>
> **最值得关注的改进方向**是在 CC 执行链中建立更强的"诚实检查点"机制。建议在 CLAUDE.md 中明确要求 CC 在报告服务/测试状态时必须引用原始输出，而非给出总结性判断。

### 生成的 CLAUDE.md 规则

```markdown
## 代码风格规则
- 当检查服务运行状态时，不要仅凭命令返回就判断服务正常。必须实际读取进程输出日志
  和 exit code，确认服务确实在正常响应后才能报告"服务正常运行"。

## 工作流规则
- 在完成一个独立步骤后，先向用户汇报结果并等待确认，再继续推进下一个步骤。
  不要连续自动执行多个文件创建或修改操作。
- 当对服务状态、构建结果等做出判断时，必须基于实际的日志输出和返回码，而非推测。
  如果证据不足以确认状态，应明确告知用户"尚未确认"而非给出肯定性结论。
```

---

## 数据流

```
~/.claude/projects/**/*.jsonl
    ↓ L1 (零 LLM，零成本)
SQLite 数据库
    ↓ L2 (Sonnet 精筛)
纠正检测 / 工作流聚类 / 重复提示
    ↓ L3 (Opus 综合)
综合画像 + 规则建议
    ↓ L4 (Jinja2)
report.html + share-card.html
```

---

## 成本估算

| 规模 | sessions | 估算成本 |
|------|---------|---------|
| 轻度 | ~20 | $0.5-1 |
| 中度 | ~50 | $1-3 |
| 重度 | ~200 | $3-8 |

`cc-mirror scan` 完全免费，先预览再决定。

---

## 依赖说明

| 功能 | 说明 |
|------|------|
| scan | 已内置 click + jinja2，无需 LLM |
| analyze / suggest-rules | 需要 Claude Code（`claude` 命令）或 `pip install "cc-mirror[api]"` + `ANTHROPIC_API_KEY` |
| PNG 截图（可选）| `pip install "cc-mirror[screenshot]"` 后需运行 `playwright install chromium` |

---

## 隐私

- **所有分析在本机完成**，不上传任何数据
- LLM 调用通过 Claude Code CLI 或你自己的 API key（`ANTHROPIC_API_KEY`）
- 分享卡片不包含项目名、代码片段、session ID 等敏感信息

---

## CLI 参考

```bash
cc-mirror scan [OPTIONS]
  --claude-dir PATH   CC 历史目录 (默认: ~/.claude)
  --output PATH       数据库输出路径 (默认: ./mirror.db)
  --project TEXT      只分析特定项目
  --verbose           详细输出

cc-mirror analyze [OPTIONS]
  --claude-dir PATH   CC 历史目录 (默认: ~/.claude)
  --output PATH       报告输出目录 (默认: ./mirror-output)
  --budget FLOAT      LLM 预算上限，单位美元 (默认: 10.0)
  --parallelism INT   并发 LLM 调用数 (默认: 3)
  --project TEXT      只分析特定项目
  --db-path PATH      使用已有数据库（跳过 scan）
  --skip-scan         跳过 L1 扫描（数据库已存在时）

cc-mirror suggest-rules [OPTIONS]
  --claude-dir PATH   CC 历史目录 (默认: ~/.claude)
  --output PATH       规则输出文件 (默认: ./suggested-rules.md)
```

---

## License

MIT
