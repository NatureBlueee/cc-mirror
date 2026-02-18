# CC Mirror — 架构决策记录

> 这是架构文档，不是实现文档。记录"为什么"，不记录"怎么写"。

---

## ADR-001: SQLite 作为中间状态存储

**决策**：用 SQLite（单文件 DB）存储所有中间结果，不用 JSON 文件或 Postgres。

**为什么**：
- L1 产出的数据需要被 L2 多次查询（"给我纠正候选消息前的 3 条 assistant 消息"）
- JSON 文件无法高效随机访问，Postgres 需要单独部署
- SQLite：零部署，单文件，FTS5 全文搜索，与 Python 标准库无缝集成
- 产出的 `mirror.db` 本身就是可审计的分析结果——用户可以自己查询

**影响**：
- L1 → DB → L2 的路径是确定的（不能绕过 DB 直接传内存）
- 开发时需要用 DB Browser 或 CLI 查看中间状态
- 增量运行：L1 可以通过检查 sessions 表跳过已处理的文件

---

## ADR-002: 代码粗筛 + LLM 精筛的两级模式

**决策**：L1 用代码做粗筛（标记候选），L2 用 LLM 做精筛（确认 + 分类）。

**为什么**：
- 全量 LLM：500 sessions × 50 消息/session = 25000 次 LLM 调用，成本不可接受
- 全量代码：精度不够，误判率高
- 两级：代码将候选降低到 10-30%，LLM 只处理候选 → 成本降低 70-90%

**粗筛标准（L1）**：
- user 消息 + 有文本 + > 15 字符 + 不是纯确认词
- 不是新主题（与前文关键词有交集）
- 前一条是 assistant 消息

**精筛任务（L2）**：
- 这真的是纠正吗？（是/否）
- 纠正类型？（5 类）
- 是否可规则化？

---

## ADR-003: 上下文窗口永远不装整个 session

**决策**：L2 构造 prompt 时，每次最多包含 3-10 条消息的上下文（从 SQLite 查询），绝不一次性读入整个 session。

**为什么**：
- 长 session 可能有数千条消息，远超 LLM 上下文窗口
- 对于纠正检测，只需要候选消息 + 前 1-3 条 assistant 消息（局部上下文）
- 从 SQLite 查询特定消息比从内存中切片更精确

**实现要求**：
- L2 从 SQLite 的 messages 表查询，不持有消息对象的内存引用
- 每个 L2 prompt 的输入 < 3k tokens（对应约 500 字符 × 3-5 条消息）

---

## ADR-004: 两种产出的隐私分离

**决策**：完整报告（本地）和分享卡片（公开）是完全独立的两个文件，隐私脱敏在生成时处理，不是事后过滤。

**为什么**：
- 完整报告包含项目名、代码片段、具体纠正内容——不能公开
- 但用户想展示"我用 AI 协作 X 天，纠正率下降 Y%"
- 分离是唯一安全的设计：脱敏逻辑在 L4 Renderer 中，模板级别就不会生成敏感信息

**分享卡片禁止出现**：
- 项目名/路径/仓库名
- 代码片段、环境变量
- 具体纠正内容（只显示数量和类型）
- session ID 或可追溯标识

---

## ADR-005: Budget 控制器的降级策略

**决策**：实现硬性的 Budget 控制器，超预算时降级到 P0 only（纠正 + 工作流 + 重复提示），不 crash。

**为什么**：
- 500 session 用户的全量分析成本 $10-20，超重度用户可达 $40
- 用户应该能控制成本上限
- 降级策略：预算 > 50% → full；20-50% → P0 only；< 20% → 用已有结果生成报告

**实现位置**：`budget.py` — `BudgetController` 类，记录每次 LLM 调用成本，决定策略。

---

## ADR-006: 单文件 HTML 输出（自包含）

**决策**：报告是单个自包含 HTML 文件（内联 CSS + JS + SVG），可在 `file://` 协议下打开。

**为什么**：
- 无需部署服务器，无需网络连接
- 用户可以直接发邮件、存档、备份
- `file://` 协议下 `navigator.clipboard.writeText()` 不可用 → 用 `document.execCommand('copy')` fallback

**图表实现**：内联 SVG（静态图表）+ vanilla JS（< 200 行，处理 tooltip、展开、复制）。不用 D3/Chart.js 等外部库。

---

## ADR-007: 增量运行设计

**决策**：L1 解析支持增量运行——如果 session 已在 DB 中，跳过。

**为什么**：
- 用户可能每周运行一次 CC Mirror
- 每次只解析新增的 session，不重新处理所有历史
- JSONL 文件是 append-only 的：已在 DB 中的 session ID → 跳过整个文件

**实现**：`parse_all_sessions` 开始时查询 sessions 表，得到已处理的 session ID set。

---

## ADR-008: Standalone 项目（不混入 Towow）

**决策**：CC Mirror 是独立的 Python 包，独立的 git repo，独立的 Skills 体系。

**为什么**：
- CC Mirror 的用户不一定是 Towow 用户
- 两者的依赖不同（Towow 用 FastAPI + 向量库，CC Mirror 用 Click + SQLite）
- 独立发布：pip install cc-mirror、uvx cc-mirror

**边界**：
- CC Mirror 不依赖 Towow 的任何代码
- Towow 数据（CC 历史）是 CC Mirror 的测试数据，不是集成点

---

## ADR-009: 打包策略

**决策**：优先支持 `uvx cc-mirror`（无需安装），同时支持 `pip install cc-mirror`。

**为什么**：
- `uvx` 是最低摩擦的使用方式（一行命令，无需创建 venv）
- `pip install` 是开发者熟悉的安装方式
- 两者的 entry point 相同：`cc-mirror` 命令

**结构**：标准 `pyproject.toml` + `src/cc_mirror/` 布局（setuptools）。

---

## 待决策

### 工作流聚类算法
PRD 建议 LCS（最长公共子序列）+ 层次聚类。阈值 0.6 是初始值，需要在真实数据上调优。
Phase 0 的任务之一是验证工作流重复率（目标：能找到 ≥ 3 个聚类）。

### 认知画像标签集
PRD 建议 LLM 自由生成标签（如"递归型思考者"）。是否需要预定义标签集（可比较性）？
待 Phase 2 决策，先看 LLM 自由生成的质量。

### 分享卡片 PNG 生成
Playwright 作为可选依赖。如果用户没有 Playwright，提供手动截图指引。
Phase 3 才需要决策具体方案。
