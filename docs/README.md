# Causal Mirror 项目文档

## 项目概述

Causal Mirror 是一个自指性 AI Agent 原型系统，探索"让 AI 看见自己、理解自己、改进自己"的技术路径。

核心思路：将计算力学（Shalizi & Crutchfield, 2001）的因果状态分析引入 AI Agent 的自我认知循环，让 Agent 能够从自己的行为序列中提取因果结构模型（ε-machine），并将此模型转化为可执行的自我改进反馈。

## 文档索引

### 理论基础

| 文档 | 内容 |
|------|------|
| [五向综合分析](./analysis/五向综合分析.md) | 元学习、自我对弈、内在动机、递归自改进、Epiplexity信息度量——五个方向的论文综述与交叉分析 |
| [Epiplexity论文评析](./analysis/Epiplexity批判分析.md) | 对 arXiv:2601.03220 的详细批判：逻辑深度、有效复杂度、计算力学的前人工作与 Epiplexity 的创新性评估 |
| [计算力学×DGM深度分析](./analysis/计算力学×DGM深度分析.md) | Shalizi & Crutchfield 计算力学框架与 Darwin Gödel Machine 的全面对比与融合路径 |

### 实施规划

| 文档 | 内容 |
|------|------|
| [实施计划书](./analysis/实施计划书.md) | Causal Mirror 原型系统的详细实施计划：CSSR引擎 → 自建模Agent → CM-DGM集成 |

### 代码文档

| 文档 | 内容 |
|------|------|
| [CSSR引擎API](../cssr/__init__.py) | CSSR.fit(sequence) 公共API入口 |
| [实验入口](../main.py) | main.py 三组对照实验运行器 |

## 项目结构

```
E:\causal\
├── cssr/                    # 纯Python CSSR引擎
│   ├── __init__.py           # CSSR.fit() API
│   ├── suffix_trie.py        # 后缀字典树 (Phase 1)
│   ├── splitting.py          # 状态分裂算法 (Phase 2)
│   ├── testing.py            # 统计检验 + 多重校正
│   ├── determinization.py    # 确定化 + 瞬态移除 (Phase 3)
│   ├── machine.py            # EpsilonMachine + 度量 (Phase 4)
│   ├── visualization.py      # DOT输出 (Phase 5)
│   ├── config.py / utils.py  # 配置与工具
│   └── causal_state.py       # CausalState 数据结构
├── agent/                    # 自建模Agent
│   ├── task_solver.py        # LLM任务求解器
│   ├── behavior_encoder.py   # 行为→符号编码器
│   ├── self_model.py         # 自我模型管理器
│   ├── self_report.py        # 自我认知报告生成器
│   └── prompts/              # 提示词模板
├── tests/                    # 24项测试，全部通过
├── experiments/              # 实验配置与结果输出
├── docs/                     # 📚 文档（当前目录）
└── main.py                   # 实验主入口
```

## 快速开始

```bash
# 安装依赖
pip install scipy statsmodels numpy graphviz pyyaml openai

# 运行CSSR测试（24项）
python -m pytest tests/ -v

# 离线演示模式
python main.py --offline

# 三组对照实验（需要API key）
python main.py --rounds 5 --tasks-per-round 20
```

## 当前状态

| 组件 | 状态 |
|------|------|
| CSSR引擎 | ✅ 完成，24/24测试通过 |
| 自建模Agent框架 | ✅ 完成，离线演示跑通 |
| 三组对照实验 | ⏳ 等待API key后可运行 |
| CM-DGM集成 | 📋 远期规划，依赖Phase 2实验结果 |
