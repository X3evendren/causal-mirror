# Causal Mirror

[![Tests](https://img.shields.io/badge/tests-168%2F168-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**自指性 AI Agent 因果控制台** — 让 Agent 看见自己的行为结构、预测失败风险、干预危险操作。

Causal Mirror 将计算力学的因果状态分析引入 AI Agent 的运行时控制循环，从行为轨迹中自动提取 ε-machine（因果状态机），在线预测失败概率，并通过独立于 LLM 的策略控制器和安全治理器干预 Agent 的决策。

---

## 目录

- [核心概念](#核心概念)
- [架构](#架构)
- [快速开始](#快速开始)
- [模块详解](#模块详解)
  - [cssr — CSSR 引擎](#cssr--cssr-引擎)
  - [causal_harness — 因果控制台](#causal_harness--因果控制台)
  - [agent — 自建模 Agent](#agent--自建模-agent)
  - [cm_dgm — 进化实验](#cm_dgm--进化实验)
- [端到端流水线](#端到端流水线)
- [使用示例](#使用示例)
- [配置](#配置)
- [开发](#开发)
- [引用](#引用)

---

## 核心概念

### 问题

AI Agent（如编码助手、自主研究 Agent）面临八个系统性缺陷：

1. **无因果世界模型** — 不知道动作的前置条件和预期效果
2. **被动感知** — 不知道该读文件还是搜索还是问用户
3. **自我评估失准** — 不知道自己何时可能失败
4. **鲁棒性差** — 同样错误反复犯，不会切换策略
5. **无长程记忆** — 上下文窗口截断后丢失关键信息
6. **缺乏身份连续性** — 每次对话都是全新人格
7. **高风险动作无保护** — 删除文件、推送代码无独立安全检查
8. **多 Agent 协作混乱** — 无法区分谁的贡献导致了什么结果

### 方案

Causal Mirror 通过**因果状态分析**解决前四项：

```
Agent 行为轨迹 → CSSR 发现因果状态 → 风险模型预测失败 → 策略控制器干预 → 更安全的下一步
```

核心创新：**把 CSSR 从事后诊断器升级为在线控制器**。

---

## 架构

```
                        user / benchmark / scheduled task
                                    │
                                    ▼
                          Nanobot Agent Loop
                                    │
                                    ▼
                          Causal Mirror Hook
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
      Event Ledger           Behavior Encoder          Safety Governor
    (append-only JSONL)     (11-symbol Layer A)      (risk-tiered policy)
            │                       │                       │
            ▼                       ▼                       ▼
      Episode Manager         CSSR Engine            Verification Gate
    (boundary tracking)    (ε-machine discovery)    (pre-exec checks)
            │                       │                       │
            ▼                       ▼                       ▼
     Training Extractor     Risk Model (Markov)    Policy Controller
   (symbols + outcomes)     (P(fail | ctx,act))   (6 intervention rules)
                                    │                       │
                                    ▼                       ▼
                              Pipeline Decision       Loop Detector
                             (intervention + risk)   (state-action pairs)
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/X3evendren/causal-mirror.git
cd causal-mirror
pip install -r requirements.txt
```

### 运行测试

```bash
# 全部 168 项测试
python -m pytest tests/ -v

# 仅 CSSR 引擎测试
python -m pytest tests/test_integration.py tests/test_suffix_trie.py -v

# 仅流水线集成测试
python -m pytest tests/causal_harness/test_pipeline.py -v
```

### 离线演示（无需 API key）

```bash
# CSSR 引擎演示
python main.py --offline

# 端到端流水线演示
python -m causal_harness.pipeline.demo
```

### 实时 LLM 实验（需要 API key）

```bash
export OPENAI_API_KEY="sk-..."

# 三组对照实验 (A: CSSR自建模 / B: 通用反馈 / C: 裸Agent)
python main.py --rounds 5 --tasks-per-round 20

# 高难度竞赛题集
python main.py --hard --rounds 3

# 使用 Ollama 本地模型
python main.py --config experiments/config_ollama.yaml
```

---

## 模块详解

### cssr — CSSR 引擎

纯 Python 实现的 Causal State Splitting Reconstruction 算法。

**四阶段流水线：**

| 阶段 | 文件 | 功能 |
|------|------|------|
| 1 | `suffix_trie.py` | 后缀字典树 — 一滑 O(N) 构建 |
| 2 | `splitting.py` | 状态分裂 — 卡方/KS 检验 + Bonferroni/BH 修正 |
| 3 | `determinization.py` | 确定化 + Tarjan SCC 瞬态移除 |
| 4 | `machine.py` | ε-machine 组装 + 度量计算 |

**基本用法：**

```python
from cssr import CSSR, CSSRConfig

# 经典 Even Process (无连续 1)
seq = ['0', '1', '0', '0', '1', '0', '1', '0', ...]

config = CSSRConfig(L_max=3, alpha=0.01, min_count=5)
model = CSSR(config).fit(seq)

print(f"C_mu  = {model.metrics['statistical_complexity']:.4f} bits")
print(f"h_mu  = {model.metrics['entropy_rate']:.4f} bits")
print(f"E     = {model.metrics['excess_entropy']:.4f} bits")
print(f"n_states = {model.metrics['n_states']}")
print(model.to_dot())  # Graphviz DOT 输出
```

**度量说明：**

| 度量 | 符号 | 含义 | 正常范围 |
|------|------|------|----------|
| 统计复杂度 | C_μ | 行为模式的内部多样性 | 0–3 bits |
| 熵率 | h_μ | 每步行为的不确定性 | 0–3 bits |
| 超熵 | E | 过去与未来的互信息 | 0–2 bits |
| 预测信息 | χ | 状态对下一步的预测能力 | 0–C_μ |

**支持的过程：**

```python
# IID 过程 → 1 状态, C_μ ≈ 0
seq = generate_iid(5000)

# Even Process → 2 状态, C_μ ≈ 0.918
seq = generate_even_process(5000)

# Golden Mean → 2 状态, C_μ ≈ 0.918
seq = generate_golden_mean(5000)

# 2 阶 Markov → ≥2 状态, C_μ > 0
seq = generate_order2_markov(10000)
```

---

### causal_harness — 因果控制台

V2 架构核心，七个模块通过 `CausalPipeline` 编排。

#### 事件系统 (events/)

```python
from causal_harness.events.schema import Event, EventType, new_event
from causal_harness.events.ledger import EventLedger

ledger = EventLedger("sessions/session_01")

# 记录事件
ledger.append(new_event(
    EventType.TOOL_CALLED,
    episode_id="ep_001", turn_id=1,
    payload={"tool": "edit_file", "file_path": "src/main.py"},
))

# 按 episode 回放
for event in ledger.replay(episode_id="ep_001"):
    print(event.event_type.value, event.payload)
```

**14 种事件类型：** `TURN_STARTED`, `EPISODE_STARTED`, `GOAL_LOADED`, `ACTION_PROPOSED`, `TOOL_CALLED`, `TOOL_RESULT`, `VERIFICATION_RUN`, `ERROR_DETECTED`, `REPAIR_ATTEMPTED`, `USER_APPROVAL_REQUESTED`, `STATE_TRANSITION`, `TURN_COMPLETED`, `EPISODE_COMPLETED`

#### 行为编码 (encoding/)

```python
from causal_harness.encoding.encoder import BehaviorEncoderV2

encoder = BehaviorEncoderV2()

# 从事件流编码为 CSSR 符号
symbols = encoder.encode_all(ledger.replay())
# → ["PLAN", "READ_FILE", "EDIT_FILE", "ERROR", "REPAIR", "BOUNDARY", ...]
```

**11 个 Layer A 符号：** `OBSERVE` `PLAN` `READ_FILE` `EDIT_FILE` `RUN_TEST` `VERIFY` `ERROR` `REPAIR` `ASK_USER` `ESCALATE` `COMPLETE`

**设计原则：** Layer A 只捕获行为意图（做了什么），质量评估（做得如何）留给 Layer B 侧通道。

#### 风险模型 (models/)

```python
from causal_harness.models.markov_baseline import MarkovRiskModel

model = MarkovRiskModel(order=3, min_count=1)

# 训练：episodes 是符号序列列表，outcomes 是 0/1 标签
episodes = [
    ["READ_FILE", "EDIT_FILE", "RUN_TEST"],        # 成功
    ["READ_FILE", "EDIT_FILE", "ERROR", "REPAIR"],  # 失败
]
outcomes = [0, 1]
model.train(episodes, outcomes)

# 预测
risk = model.predict(
    context=["READ_FILE", "EDIT_FILE"],  # 最近上下文
    next_action="EDIT_FILE",             # 拟执行动作
)
print(f"P(failure) = {risk.p_failure:.2%}")
print(f"confidence  = {risk.confidence:.2f}")
```

抽象基类 `RiskModel` 支持替换实现（Markov / CSSR / logistic / transformer）。

#### 策略控制器 (controller/)

```python
from causal_harness.controller.policy import PolicyController
from causal_harness.models.base import RiskPrediction

controller = PolicyController()

risk = RiskPrediction(p_failure=0.75, confidence=0.3)
ctx = {
    "loop_count": 0,
    "has_read_file": False,
    "risk_tier": "medium",
    "reversible": True,
}

intervention = controller.evaluate(risk, ctx)
print(intervention.result.value)  # → "inspect_first"
print(intervention.reason)        # → "Confidence=0.30, no inspection done yet"
```

**6 条内置规则（可独立开关）：**

| 规则 | 条件 | 干预 |
|------|------|------|
| `high_risk_verify` | P(fail) > 0.60 | VERIFY_BEFORE |
| `repair_switch` | 同类错误 ≥ 2 次 | REPLAN |
| `info_gain_first` | 置信度 < 0.3 且未检查 | INSPECT_FIRST |
| `loop_detect` | 同一模式 ≥ 3 次 | REPLAN |
| `high_impact_approval` | 高/严重风险不可逆 | ASK_USER / BLOCK |
| `low_confidence_ask` | 低置信 + 高歧义 | ASK_USER |

#### 安全治理器 (safety/)

独立于 LLM 的安全策略引擎——LLM 的输出不能改变安全分级。

```python
from causal_harness.safety.governor import SafetyGovernor

governor = SafetyGovernor(approval_policy="auto_low")

# 读文件 — 自动允许
decision = governor.check("read_file", {"file_path": "src/main.py"})
print(decision.decision)  # → ALLOW

# 编辑文件 — 需要先读
decision = governor.check("edit_file", {"file_path": "src/main.py"})
print(decision.decision)  # → BLOCK (precondition not met)

# 记录已读后再编辑
governor.record_file_read("src/main.py")
decision = governor.check("edit_file", {"file_path": "src/main.py"})
print(decision.decision)  # → ALLOW_WITH_LOG
```

**风险分级：**

| 级别 | 例子 | 策略 |
|------|------|------|
| LOW | `read_file`, `search`, `list_files` | 自动允许 |
| MEDIUM | `edit_file`, `write_file`, `shell` | 满足前置条件允许 |
| HIGH | `exec`, `spawn`, `cron` | 需要批准 |
| CRITICAL | `delete_file`, `git_push` | 必须批准 |

#### 类型化记忆 (memory/)

```python
from causal_harness.memory.schema import MemoryType, MemoryItem
from causal_harness.memory.store import MemoryStore

store = MemoryStore("sessions/memory")

# 添加记忆
store.add(MemoryItem(
    memory_id="",
    memory_type=MemoryType.PROJECT_STATE,
    content="Files modified: src/auth.py, tests/test_auth.py",
    source_events=["evt_001", "evt_002"],
    confidence=0.9,
    tags=["files", "auth"],
))

# 检索
results = store.retrieve("auth", memory_type=MemoryType.PROJECT_STATE)

# 过期衰减
store.decay_stale(MemoryType.PROJECT_STATE, max_age_days=30)
```

**四种记忆类型：** `IDENTITY`（身份原则）`PREFERENCES`（用户偏好）`PROJECT_STATE`（项目状态）`SKILLS`（成功模式）

#### 评估系统 (eval/)

```python
from causal_harness.eval.metrics import compute_reliability_metrics, compare_interventions

# 计算可靠性指标
metrics = compute_reliability_metrics(episode_results)

# A/B 对比
delta = compare_interventions(baseline_metrics, intervention_metrics)
print(delta["summary"])
```

**12 项指标：** 任务成功率、静默失败率、恢复率、验证精确率/召回率、Brier 分数、校准误差、循环率、不必要升级率、回滚成功率

---

### agent — 自建模 Agent

V1 原型：CSSR 分析 → 自我报告 → 注入 prompt。

```python
from agent.self_model import SelfModel
from agent.behavior_encoder import BehaviorEncoder
from agent.self_report import SelfReportGenerator
from cssr import CSSRConfig

encoder = BehaviorEncoder()
report_gen = SelfReportGenerator(client, model="gpt-4o")

self_model = SelfModel(encoder, report_gen, CSSRConfig())

# 分析一轮行为
result = self_model.analyze_round(traces, round_num=1)
print(result["report"])   # 自然语言自我认知报告
print(result["metrics"])  # C_mu, h_mu, E, chi
```

---

### cm_dgm — 进化实验

CM-DGM (Computational Mechanics × Darwin Gödel Machine)：双分支进化实验，比较 **CM 加权适应度** 与 **纯准确率适应度**。

```bash
python -m cm_dgm.experiment --generations 10 --agents 6 --problems 20

# 快速模式
python -m cm_dgm.experiment --quick
```

**CM 加权适应度：**
```
fitness = accuracy × 1.0
        + C_μ_norm × 0.3
        + E_norm × 0.2
        - h_μ_norm × 0.1
        + alphabet_diversity × 0.1
```

包含验证门：CSSR 指标异常时自动退回纯准确率适应度。

---

## 端到端流水线

```python
from causal_harness.pipeline import CausalPipeline
from causal_harness.events.schema import EventOutcome

# 创建流水线
pipeline = CausalPipeline(
    cssr_config=CSSRConfig(L_max=None, alpha=0.05, min_count=2),
    markov_order=3,
    risk_threshold=0.50,
)

# 模拟一个 episode
pipeline.start_episode("Debug auth module")
pipeline.record_event(tool_call_event)
pipeline.record_event(tool_result_event)
pipeline.end_episode(EventOutcome.SUCCESS)

# 分析行为结构
metrics = pipeline.analyze()
print(f"C_mu = {metrics['statistical_complexity']:.4f}")

# 训练风险模型
pipeline.train()

# 评估拟执行动作
decision = pipeline.evaluate_action("EDIT_FILE")
print(f"Risk: {decision.risk.p_failure:.2%}")
print(f"Intervention: {decision.intervention.result.value}")
```

**完整演示：**

```bash
python -m causal_harness.pipeline.demo
```

输出示例：
```
============================================================
  Causal Mirror V2 — 端到端流水线演示
============================================================

========================================================
  Episode: Debug auth module
========================================================
  [OK] read_file     P(fail)=0.50    -> ask_user
  [ERR] edit_file     P(fail)=0.50    -> ask_user
  ...
  Outcome: failure

  ── CSSR Analysis (after Ep 3) ──
  n_states = 5
  C_mu     = 2.2151 bits
  h_mu     = 0.5850 bits
  E        = 1.6300 bits

  ── Training Risk Model ──
  Training data: 3 episodes
  Training result: success

  ── Intervention Test ──
  OK OBSERVE      risk=0.67   -> allow
  WARN EDIT_FILE  risk=0.67   -> inspect_first
```

---

## 使用示例

### 快速诊断：Agent 行为复杂度

```python
from cssr import CSSR, CSSRConfig

traces = ["OBSERVE", "EDIT_FILE", "RUN_TEST", "VERIFY", ...]

model = CSSR(CSSRConfig()).fit(traces)
print(f"C_mu = {model.metrics['statistical_complexity']:.4f}")

if model.metrics['statistical_complexity'] < 0.1:
    print("Warning: behavior too simple — possible rut")
```

### Hook 集成（与 nanobot）

```python
from causal_harness.events.ledger import EventLedger
from causal_harness.integrations.nanobot import CausalMirrorHook
from causal_harness.pipeline import CausalPipeline

ledger = EventLedger("sessions/session_01")
pipeline = CausalPipeline(ledger_dir="sessions/session_01")
hook = CausalMirrorHook(ledger, pipeline=pipeline)

# 将 hook 传给 nanobot AgentLoop
# AgentLoop 会自动在每次迭代中调用 hook
```

### 离线分析：从日志重建行为

```python
from causal_harness.events.ledger import EventLedger
from causal_harness.encoding.encoder import BehaviorEncoderV2
from cssr import CSSR, CSSRConfig

ledger = EventLedger("sessions/session_01")
encoder = BehaviorEncoderV2()

# 编码历史会话
symbols = encoder.encode_all(ledger.replay())

# CSSR 分析
model = CSSR(CSSRConfig()).fit(symbols)
print(model.to_dot())  # 导出为 Graphviz
```

---

## 配置

### CSSRConfig

```python
from cssr import CSSRConfig

config = CSSRConfig(
    L_max=None,            # 最大历史长度 (None=自动估计)
    alpha=0.001,           # 显著性水平
    test="chi2",           # 检验类型: "chi2" 或 "ks"
    correction="bonferroni",  # 多重检验修正: "bonferroni", "bh", None
    min_count=5,           # 最小后缀出现次数
    remove_transient=True, # 是否移除瞬态
)
```

### CausalPipeline

```python
from causal_harness.pipeline import CausalPipeline

pipeline = CausalPipeline(
    ledger_dir="pipeline_data",  # 事件存储目录
    cssr_config=CSSRConfig(),    # CSSR 配置
    markov_order=3,              # Markov 风险模型阶数
    risk_threshold=0.60,         # 触发验证的风险阈值
    loop_window=10,              # 循环检测窗口
    loop_threshold=3,            # 循环检测阈值
)
```

### 策略规则开关

```python
pipeline.policy.disable_rule("high_risk_verify")  # 关闭高风险验证
pipeline.policy.disable_rule("loop_detect")       # 关闭循环检测
# 用于 A/B ablation 实验
```

---

## 开发

### 项目结构

```
causal-mirror/
├── cssr/                      # 纯 Python CSSR 引擎 (800+ 行)
│   ├── __init__.py            # CSSR.fit() 主入口
│   ├── suffix_trie.py         # 后缀字典树
│   ├── splitting.py           # 状态分裂 (核心算法)
│   ├── determinization.py     # 确定化 + 瞬态移除
│   ├── machine.py             # ε-machine + 度量
│   ├── testing.py             # 统计检验 + 多重修正
│   ├── visualization.py       # DOT 输出
│   ├── config.py              # 配置
│   └── causal_state.py        # CausalState 数据结构
├── causal_harness/            # V2 因果控制台 (2500+ 行)
│   ├── pipeline/              # 端到端编排器 ★
│   ├── events/                # 事件账本
│   ├── encoding/              # V2 11 符号编码
│   ├── models/                # 风险预测模型
│   ├── controller/            # 策略 + 验证 + 循环检测
│   ├── safety/                # 安全治理器
│   ├── memory/                # 类型化记忆
│   ├── eval/                  # 评估系统
│   ├── agents/                # 多 Agent 角色
│   └── integrations/          # 外部 Agent 框架集成
├── agent/                     # V1 自建模 Agent
├── cm_dgm/                    # CM-DGM 进化实验
├── experiments/               # 实验配置和结果
├── tests/                     # 168 项测试
├── docs/                      # 文档和分析
├── main.py                    # 实验主入口
└── pyproject.toml
```

### 运行测试

```bash
# 全部测试
python -m pytest tests/ -v

# 按模块
python -m pytest tests/causal_harness/ -v
python -m pytest tests/test_integration.py -v

# 覆盖率
python -m pytest tests/ --cov=cssr --cov=causal_harness --cov-report=html
```

### 贡献

1. 确保 168 项测试全部通过
2. 新模块需包含测试
3. 遵循现有代码风格：dataclass 优先、type hint 完整、docstring 包含使用示例

---

## 依赖

| 包 | 用途 |
|----|------|
| `numpy` | 转移矩阵、稳态分布计算 |
| `scipy` | 卡方分布、特征值分解 |
| `statsmodels` | Benjamini-Hochberg 多重检验修正 |
| `graphviz` | ε-machine 可视化 |
| `openai` | LLM API（实验模式） |
| `pyyaml` | 配置文件解析 |
| `pytest` | 测试框架 |

---

## 引用

Causal Mirror 基于以下理论工作：

- **Shalizi & Crutchfield (2001)** — *Computational Mechanics: Pattern and Prediction, Structure and Simplicity*
- **Shalizi & Klinkner (UAI 2004)** — *Blind Construction of Optimal Nonlinear Recursive Predictors for Discrete Sequences* (CSSR 算法)
- **Crutchfield & Young (1989)** — *Inferring Statistical Complexity* (ε-machine)

```
@software{causal_mirror_2026,
  title        = {Causal Mirror: Self-Referential Agent Causal Control Harness},
  url          = {https://github.com/X3evendren/causal-mirror},
  description  = {CSSR computational mechanics engine + behavior encoding + risk prediction + policy intervention + safety governance for AI agents},
}
```

---

## 许可证

MIT
