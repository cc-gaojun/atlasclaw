# AtlasClaw Self-Improving Runtime Design

## Goal

将 `atlasclaw-providers/skills/self-improving` 从仅供模型参考的 Markdown skill，升级为 AtlasClaw 原生的自动学习能力层。该能力默认自动生效，能够在运行时捕获纠正、失败和反思事件，分层写入用户私有存储，并通过分级确认机制决定哪些经验可以晋升为长期记忆。

## Status

- 设计状态：已确认，可进入 implementation plan
- 范围：AtlasClaw core runtime + memory/session integration
- 非目标：不实现自动改代码、自动改 prompt、自动改配置

## Problem Statement

当前 `self-improving` 目录中的 `SKILL.md` 可以作为 Markdown skill 被发现并暴露给模型，但它只能提供行为说明，不能稳定实现以下系统级能力：

1. 默认自动生效，而不是依赖用户显式调用 skill
2. 监听 agent 生命周期、模型失败、工具失败、用户纠正等运行时事件
3. 将学习事件写入 AtlasClaw 自己的用户隔离存储，而不是 `~/self-improving/`
4. 对候选经验进行分层、审计和用户确认
5. 将确认后的经验接入现有 `memory_*.md` 长期记忆与上下文回注机制

这意味着当前 `self-improving` 更像“规则说明文档”，而不是 OpenClaw / Claude Code 那类可持续自动运行的 memory + hooks 能力。

## Design Principles

1. 默认自动生效，但保持被动学习
2. 用户隔离优先，所有数据必须按 `workspace/users/<user_id>/...` 存储
3. 原始事件与长期记忆分层，避免噪声污染长期记忆
4. 所有晋升到长期记忆的内容必须走分级确认机制
5. 只学习显式信号，不从沉默、猜测或隐式观察中推断偏好
6. 不绕开现有 session、memory、prompt compaction 架构
7. 优先复用当前 `runner.py`、`runtime_events.py`、`history_memory.py`、`MemoryManager`、`SessionManagerRouter`

## OpenClaw Alignment

本设计需要显式对齐 OpenClaw 官方能力分层，而不是把 `self-improving` 做成一次性的 AtlasClaw 特例。

### OpenClaw Capability Model

根据 OpenClaw 官方文档：

- `Skills` 更偏向能力说明和行为包
- `Hooks` 负责监听 command / session / agent / message 等事件并执行自动化
- `Plugins` 负责扩展新的能力边界，并通过 slot 机制接入 memory / contextEngine 等独占能力
- memory/context 的自动捕获与回注并不依赖单个 `SKILL.md`，而是依赖 hooks + memory/context engine 的运行时集成

参考：

- OpenClaw Hooks: <https://docs.openclaw.ai/automation/hooks>
- OpenClaw Plugins: <https://docs.openclaw.ai/tools/plugin>
- OpenClaw Skills: <https://docs.openclaw.ai/tools/skills>

### AtlasClaw Mapping

本设计在 AtlasClaw 中采用以下映射：

- `SKILL.md`：保留为规则说明与提示源，等价于 OpenClaw 的 skill 行为文档层
- `runtime_events.py` + `runner.py`：承担 OpenClaw hooks 的事件捕获职责
- `self_improving` store：承担 memory plugin 的持久化职责
- confirmed lesson recall：承担简化版 context engine 的上下文回注职责

### Explicit Non-Goals for Phase 1

第一阶段不做完整 OpenClaw 风格 plugin engine，但会为它留出边界：

- 不实现通用插件安装 / 启停系统
- 不实现通用 hook 包发现机制
- 不实现完整 slot registry

但 Phase 1 的接口和数据模型必须可向以下方向扩展：

- `memory` slot：未来可替换或并存更多 memory backends
- `contextEngine` slot：未来可替换 recall / selection 策略
- hook subscription：未来支持更多自动化能力，而不仅仅是 `self-improving`

## User-Approved Behavioral Rules

以下规则来自本轮对话，属于已确认需求：

- 默认自动生效
- 触发事件包括：
  - 用户明确纠正或否定
  - 一次任务结束后的轻量自我反思
  - 工具、模型或执行失败事件
- 存储策略：
  - `memory_*.md + self_improving/` 子目录日志
- 晋升策略：分层写入 + 分级确认
- 长期记忆晋升不是“全部自动”，也不是“每条都即时打断确认”
- 推荐策略：
  - 原始事件先入日志
  - 候选经验进入 pending
  - 用户在合适时机确认后才进入 `memory_*.md`

## High-Level Architecture

在现有架构中新增一个轻量的“自改进运行时能力层”，而不是引入一个完整的 plugin engine。这个能力层由三部分组成：

1. Runtime Hook Capture
- 基于现有 `AgentRunner` 和 `RuntimeEventDispatcher`
- 捕获 run 成功、run 失败、模型错误、工具错误、用户纠正等事件

2. Self-Improving Store
- 将事件写入用户专属 `self_improving/` 子目录
- 形成 `events -> pending -> decisions -> confirmed memory` 的分层链路

3. Memory Context Integration
- 已确认经验进入现有长期记忆文件
- 上下文回注仍由现有 memory / compaction 流程负责
- self-improving 原始日志不直接注入 prompt

## Storage Model

### Primary Storage

所有自改进数据都存放在：

`workspace/users/<user_id>/memory/self_improving/`

建议目录结构：

```text
workspace/users/<user_id>/memory/
├── memory_20260328_101530_123456.md
├── memory_20260328_142001_654321.md
└── self_improving/
    ├── events.jsonl
    ├── pending.jsonl
    ├── decisions.jsonl
    ├── snapshots/
    └── summaries/
```

### File Responsibilities

- `events.jsonl`
  - 记录原始学习事件
  - 包括 correction / reflection / failure 三类
  - 只追加，不回写

- `pending.jsonl`
  - 记录候选经验
  - 包含状态字段：`pending` / `confirmed` / `rejected`
  - 保留来源事件引用

- `decisions.jsonl`
  - 记录用户的确认或拒绝决策
  - 可用于审计、回放和统计

- `snapshots/`
  - 可选，用于定期导出聚合状态或故障排查

- `summaries/`
  - 可选，用于后续增加 digest / weekly review 时使用

### Confirmed Memory

已确认候选经验不写入 `self_improving/` 目录本身，而是追加进入现有长期记忆链：

`workspace/users/<user_id>/memory/memory_<timestamp>.md`

这样可以与现有 `HistoryMemoryCoordinator` 的 memory recall / context compaction 流程保持一致。

## Event Types

### 1. User Correction Event

触发条件：
- 用户明确否定或纠正上一轮结果
- 典型模式：
  - “不对”
  - “不是这个意思”
  - “我说的是...”
  - “你错了”
  - “应该是 ...”

记录内容：
- `session_key`
- `run_id`（如可用）
- `previous_assistant_message_excerpt`
- `user_correction_message`
- `candidate_lesson`
- `confidence`
- `source = user_correction`

### 2. Reflection Event

触发条件：
- run 正常完成后
- 仅对满足“显著任务”阈值的 run 做轻量反思
- 阈值可参考：
  - 有工具调用
  - 回复长度超过最小值
  - 会话不是 trivial small talk

记录内容：
- 任务上下文摘要
- 完成质量自评
- 潜在改进项
- 是否形成可复用 lesson

### 3. Failure Event

触发条件：
- 模型 4xx/5xx 失败
- 工具执行失败
- 运行异常中断
- 明确的 hook / queue / stream error

记录内容：
- 失败类型
- 错误摘要
- 发生阶段
- 输入上下文摘要
- lesson 候选

## Confirmation Strategy

### Tier 1: Raw Event Logging

所有三类事件都自动记录到 `events.jsonl`。

### Tier 2: Candidate Promotion to Pending

- 用户纠正：可立即生成 pending 候选
- failure / reflection：先允许聚合和去重，再生成 pending 候选
- 敏感、一次性、上下文专属内容永不生成 pending

### Tier 3: User Confirmation

只有用户确认后，候选经验才写入 `memory_*.md`。

确认方式第一阶段不做强打断弹窗，而是提供：
- 查询 pending 列表
- confirm 单条
- reject 单条

后续可以再加前端提醒入口。

## Context Injection Rules

### What Gets Injected

只允许以下内容进入后续上下文：
- 已确认并写入 `memory_*.md` 的 lesson
- 由现有 memory recall 机制筛选出的少量相关长期记忆

### What Does Not Get Injected

不直接注入：
- `events.jsonl` 原始日志
- `pending.jsonl` 中未确认项
- `decisions.jsonl` 纯审计记录

### Why

这样可以：
- 保持 prompt 清洁
- 避免错误 lesson 在未确认前污染行为
- 与当前 memory compaction / recall 模型保持一致

## Runtime Integration Points

### AgentRunner

在 `runner.py` 中增加三个接入点：

1. run start / early metadata
- 准备 self-improving runtime context

2. run success path
- 在 transcript 持久化后，触发 reflection candidate analysis

3. run failure path
- 在异常或模型失败时，记录 failure event

### RuntimeEventDispatcher

在 `runtime_events.py` 中补充：
- tool failure collection
- model error classification
- lifecycle event forwarding for self-improving runtime

### HistoryMemoryCoordinator

不改现有 compaction 语义，但新增“确认后 lesson 写入长期记忆”的辅助入口。

### Session / User Routing

必须复用当前：
- `SessionManagerRouter`
- `build_scoped_deps(...)`
- `user_info.user_id`

确保所有自改进数据按真实用户隔离，不落到共享 default bucket。

## API Surface

新增一组最小 API 即可，不做过度系统化：

### Proposed Endpoints

- `GET /api/self-improving/pending`
  - 返回当前用户待确认候选经验

- `POST /api/self-improving/pending/{id}/confirm`
  - 确认某条候选经验
  - 写入 `decisions.jsonl`
  - 追加写入新的 `memory_<timestamp>.md`

- `POST /api/self-improving/pending/{id}/reject`
  - 拒绝某条候选经验
  - 写入 `decisions.jsonl`
  - 更新 pending 状态

- `GET /api/self-improving/events`
  - 查看最近学习事件（可选分页）

后续如果需要，再提供汇总和统计接口。

## Data Model

### SelfImprovingEvent

建议字段：
- `id`
- `user_id`
- `session_key`
- `run_id`
- `event_type` (`correction|reflection|failure`)
- `source_phase`
- `content`
- `candidate_lesson`
- `sensitive`
- `created_at`
- `metadata`

### PendingLesson

建议字段：
- `id`
- `user_id`
- `source_event_ids`
- `lesson_text`
- `category`
- `status` (`pending|confirmed|rejected`)
- `requires_confirmation` (always true for this phase)
- `created_at`
- `updated_at`

### DecisionRecord

建议字段：
- `id`
- `pending_id`
- `decision` (`confirm|reject`)
- `decided_by`
- `decided_at`
- `note`

## Safety Rules

必须遵守以下约束：

- 不存储凭证、token、密码、密钥
- 不存储健康、财务、第三方敏感信息
- 不从沉默中学习偏好
- 不自动修改代码、配置、prompt、skill 文件
- 不在未确认前把 lesson 注入长期记忆
- 不让原始失败细节无限进入 prompt

## Compatibility with Existing self-improving SKILL.md

当前 `atlasclaw-providers/skills/self-improving/SKILL.md` 继续保留，其角色调整为：

- 行为规则文档
- 人类可读规范
- prompt 里的补充说明来源

它不再承担“默认自动运行”的唯一实现责任。默认自动运行由 AtlasClaw runtime 负责。

## Documentation Updates Required

本设计落地时需要同步更新：

- `docs/architecture.md`
  - 增加 self-improving runtime / memory hook 描述
- `docs/module-details.md`
  - 增加 self-improving store / API / event model
- `docs/development-spec.md`
  - 增加学习事件与安全边界说明
- `docs/SKILL_GUIDE.md`
  - 说明“说明型 skill”和“runtime-native automated skill”的区别

## Testing Strategy

### Unit Tests

- correction event detection
- failure event logging
- reflection event generation
- pending candidate creation
- confirm -> write memory_*.md
- reject -> no memory write
- sensitive event filtering
- per-user isolation

### Integration Tests

- runner success path emits reflection event
- runner failure path emits failure event
- confirmed lesson enters memory recall chain

### E2E Tests

- create correction candidate via real chat flow
- list pending lessons
- confirm lesson and verify memory file created
- reject lesson and verify no memory promotion

## Phasing

### Phase 1

- Store + event capture + pending/confirm/reject API
- Confirmed lesson -> `memory_*.md`
- No UI entry yet

### Phase 2

- Frontend pending review panel or lightweight notification
- Session-level digest
- weekly / periodic summaries

### Phase 3

- generalized runtime-native automation substrate for more skills beyond self-improving

## Recommended Implementation Direction

采用“轻量 Hook + Memory 框架，只先落地 self-improving”方案：

- 不做完整 plugin engine
- 不做 self-improving 特例硬编码到无法复用
- 先建设最小可复用 runtime 能力层
- 让 self-improving 成为第一个原生接入者

这样既满足当前需求，又为未来自动化 skill、memory plugin、context engine 预留扩展点。
