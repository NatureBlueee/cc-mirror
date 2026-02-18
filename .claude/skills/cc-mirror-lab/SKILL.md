---
name: cc-mirror-lab
description: CC Mirror 最小测试 skill。"把自己当第一用户"——不是跑 unit test suite，是在真实数据上运行并判断输出合不合理。
---

# CC Mirror 最小测试 Skill

## 我是谁

我不是测试工程师（那是 unit test 框架的事）。
我是**第一用户**——用真实的 CC 数据跑 CC Mirror，判断输出是否有价值。

测试哲学：**能告诉用户"这对你有用吗？"的东西才算测试通过。**

---

## 最小测试定义

### Phase 0：L1 解析（scan 命令）

**验收标准**：
```
cc-mirror scan --claude-dir ~/.claude --project -Users-nature------Towow --verbose
```

预期输出：
- Sessions: 合理数字（Towow 项目应该有 50+ sessions）
- Candidate corrections: 10-30% 的用户消息（太低说明粗筛太严，太高说明太松）
- Repeated prompts: ≥ 3 个 unique patterns（Towow 肯定有重复的）
- 没有 parse errors（或只有少量可接受的）

**如果不合理**：
- Corrections 0% → 粗筛逻辑有 bug（is_confirmation_only 太宽松？）
- Corrections 80% → 粗筛太松，几乎所有用户消息都被标记了
- Parse errors 很多 → JSONL 读取有问题

---

### Phase 1：L2 检测（纠正/工作流）

**验收标准**（用 Towow 的一个 session 验证，不是全量）：

Step 1: 随机选 10 条候选纠正，人工判断：
- 其中有多少是"真正的纠正"？
- 有多少是"正常对话被误标"？
- 目标：精度 ≥ 70%

Step 2: 看 LLM 输出的 correction_type 分布：
- 不能全是 `style`（说明分类器有偏向）
- 应该有 `misunderstanding`, `wrong_approach`, `knowledge` 等多种类型

Step 3: 看工作流聚类：
- 是否找到了 ≥ 3 个有意义的聚类？
- 聚类描述是否准确？（人工判断）

---

### Phase 2：L4 报告（HTML 输出）

**验收标准**：

Step 1: 用浏览器打开 `mirror-output/report.html`
- 第一屏：数字冲击力（看到数字的第一反应是"哇"还是"？"）
- Part 1：CLAUDE.md 规则建议可以直接复制粘贴用吗？
- 链接是否都工作（`file://` 协议下）

Step 2: 打开 `mirror-output/share-card.html`
- 截图后发朋友圈的冲动（有/没有）
- 没有任何敏感信息出现（项目名、代码片段、session ID）

---

## 测试数据

### Fixture 数据（快速 smoke test，不用真实数据）

`tests/fixtures/sample_session.jsonl`：10-15 条消息，覆盖：
- 普通 user/assistant 对话
- 带 tool_use 的 assistant 消息
- 带 tool_result 的 user 消息
- 一条候选纠正（明确的纠正信号）
- 一条确认消息（"好的，继续"）
- 一条 summary 记录（compact 信号）

用于：验证 db.py + l1_parser.py 能正常工作，不用跑真实数据。

### 真实数据测试

只在以下情况使用真实数据：
1. Phase 0 验收：跑 Towow 全量，看统计数字
2. Phase 1 验收：随机抽 10 条，人工判断精度
3. Phase 2 验收：生成报告，用户打开看

**真实数据测试步骤**：
```bash
cd ~/个人项目/cc-mirror
pip install -e .
cc-mirror scan --claude-dir ~/.claude --project -Users-nature------Towow --verbose
# 看输出，判断数字是否合理
```

---

## 评估维度

### 数字合理性（L1）
| 指标 | 目标范围 | 偏低说明 | 偏高说明 |
|------|---------|---------|---------|
| 候选纠正率 | 10-30% | 粗筛太严 | 粗筛太松 |
| 重复提示 unique 数 | ≥ 3 | 数据量少或提示多样 | 正常 |
| Parse errors | < 1% | 正常 | JSONL 有问题 |

### 精度（L2，人工判断）
| 检测类型 | 目标精度 | 评估方式 |
|---------|---------|---------|
| 纠正检测 | ≥ 70% | 随机 10 条，人工判断 |
| 工作流聚类 | ≥ 60% | 看描述是否准确 |

### 可用性（L4，用户判断）
- CLAUDE.md 规则：能直接粘贴用吗？
- Skill 建议：步骤清晰可执行吗？
- 分享卡片：想发朋友圈吗？

---

## 我不做什么

- 不写完整 unit test suite（那是过度工程化）
- 不追求 100% 代码覆盖率
- 不在 CI 里跑（本地工具，本地测试）
- 不模拟 LLM 返回（用真实 LLM，这是用户体验测试）
