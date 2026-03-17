# RegMixer 项目文档

## 1. 项目介绍

### 1.1 概述

RegMixer 是由 Allen Institute for Artificial Intelligence (AI2) 开发的机器学习训练和评估工具，主要用于大规模语言模型的数据混合、训练实验管理和性能评估。该项目支持灵活的数据混合策略、多集群部署和智能实验管理。

### 1.2 核心功能

- **数据混合与合成**：使用狄利克雷分布生成多种数据混合方案，支持源级别和主题级别的权重控制
- **Transformer 模型训练**：自动配置和启动 Transformer 模型训练，支持 LLaMA 等架构
- **实验管理**：通过 Beaker 集群进行批量实验部署、状态监控和任务管理
- **评估与回归分析**：使用 LightGBM、线性回归等模型进行性能预测和优化
- **可视化工具**：提供权重分布、交互矩阵、相关性分析等可视化功能

### 1.3 技术栈

**核心依赖**：
- Python >= 3.9
- ai2-olmo-core（AI2 核心库）
- Beaker（实验管理平台）
- Click（CLI 框架）
- Pydantic（数据验证）

**数据处理**：
- PyArrow（高性能数据处理）
- datasets（Hugging Face 数据集）
- s3fs/gcsfs（云存储接口）

**评估分析**：
- LightGBM（回归模型）
- pandas/numpy（数据处理）
- scipy/seaborn（统计分析）
- matplotlib（可视化）

**实验追踪**：
- WandB（实验日志和可视化）

### 1.4 应用场景

- 大规模语言模型预训练的数据配比优化
- 多源数据混合策略的自动探索
- 模型性能预测和数据配比推荐
- 批量实验的并行部署和管理

### 1.5 Round1a 拟合与可视化

`scripts/run_round1a.sh` 现在把 round1a 控制面拆成三层产物：

- `round1a_actual_quality_mixes.json`
  - 正式训练输入。
  - schema 保持兼容，仍然只包含顶层 `mixes`。
- `round1a_actual_quality_design_summary.json`
  - 候选采样与最终选点的控制面摘要。
  - 记录 sampling strategy、selection method、anchors、rank、logdet、condition number。
- `design/`
  - 设计诊断图目录。
  - 包含候选池 simplex 图、最终选点 simplex 图、奇异值图、条件数图。

拟合阶段继续拆成两层产物：

- `p_star_actual_quality.json`
  - 官方输出。
  - 默认来自 `log_linear`。
  - 只包含 `ROUND1A_FIT_REGRESSION_TYPE` 指定的单个回归器结果。
  - 下游继续消费 round1b 或手工拷贝权重时，应以它为准。
- `fit_compare/`
  - 诊断输出。
  - 只有在 `ROUND1A_COMPARE_REGRESSIONS=1` 时生成。
  - 包含 `quadratic`、`log_linear`、`lightgbm` 三路结果，以及各自的可视化。

#### 1.5.0 Round1a 默认设计

新的 round1a 配置默认使用：

- `candidate_sampling_strategy=mixed`
  - 40% `dirichlet`
  - 40% `uniform`
  - 20% `vertex_biased`
- `design_selection_method=d_opt`
- `design_basis=quadratic_logratio_2d`
- 固定 anchors：`vertices + centroid`

控制面边界不变：

- `generate-mixes` 负责 candidate sampling + design selection
- `control_plane.py` 只负责 probe / slot plan / launch
- `fit-mixture` 只负责拟合与搜索

#### 1.5.1 如何选择回归器

- `quadratic`
  - 旧的 OLS 二次面基线。
  - 速度最快，最适合保留历史行为或做最小改动复现。
  - 缺点是容易把最优点推到 simplex 边界，尤其是 15-run 这种小样本外推时。
- `log_linear`
  - 参数化、平滑的 scaling-law 风格拟合。
  - 适合你认为“分数随配比变化更像平滑单调曲面”的场景。
  - 比 `quadratic` 慢，但通常比 OLS 更稳。
- `lightgbm`
  - 非参数局部拟合。
  - 更适合作为诊断器，看“如果尽量少做远距离外推，最优点会落在哪”。
  - 它更容易贴近已有观测点，所以不一定适合作为默认官方 p*。

推荐工作流：

1. 先用新的 `mixed + d_opt` 设计跑完整个 round1a。
2. 默认用 `ROUND1A_FIT_REGRESSION_TYPE=log_linear` 产出 canonical `p*`。
3. 如需诊断，再用 `ROUND1A_COMPARE_REGRESSIONS=1` 同时看三路结果。
3. 下游只消费 `p_star_actual_quality.json`，不要直接拿 `fit_compare/` 里的诊断结果替代官方输出。

#### 1.5.2 直接运行 fit-mixture

单回归器：

```bash
source ~/.bashrc
conda activate regmixer
cd /home/staff/jiayining/vibe_research/regmixer

PYTHONPATH=src python -m regmixer.cli fit-mixture \
  --config src/regmixer/config/nemotron-cc-round1a-actual-quality.yaml \
  --eval-metrics /path/to/eval_metrics.csv \
  --mix-file /path/to/round1a_actual_quality_mixes.json \
  --output /path/to/p_star_actual_quality.json \
  --regression-type log_linear
```

三路 compare：

```bash
PYTHONPATH=src python -m regmixer.cli fit-mixture \
  --config src/regmixer/config/nemotron-cc-round1a-actual-quality.yaml \
  --eval-metrics /path/to/eval_metrics.csv \
  --mix-file /path/to/round1a_actual_quality_mixes.json \
  --output /path/to/fit_compare/comparison_summary.json \
  --output-dir /path/to/fit_compare \
  --compare-regressions \
  --regression-type log_linear
```

说明：

- `--regression-type` 总是表示“官方/主输出”的回归器。
- `--compare-regressions` 会额外把另外两种回归器也跑出来。
- compare 模式下，方法结果固定写到 `<output-dir>/methods/*.json`。

#### 1.5.3 生成图

单回归器结果：

```bash
PYTHONPATH=src python scripts/visualize_round1a_results.py \
  --eval-metrics /path/to/eval_metrics.csv \
  --fit-json /path/to/p_star_actual_quality.json \
  --output-dir /path/to/visualizations
```

三路 compare 结果：

```bash
PYTHONPATH=src python scripts/visualize_round1a_fit_comparison.py \
  --eval-metrics /path/to/eval_metrics.csv \
  --fit-comparison-json /path/to/fit_compare/comparison_summary.json \
  --output-dir /path/to/fit_compare/visualizations
```

compare 可视化会额外生成：

- `visualizations/regression_comparison.png`
- `visualizations/<method>/round1a_actual_quality_simplex.png`
- `visualizations/<method>/round1a_actual_quality_weight_comparison.png`
- `visualizations/<method>/round1a_actual_quality_task_heatmap.png`

#### 1.5.4 接入 run_round1a.sh

最常用的两个环境变量：

- `ROUND1A_FIT_REGRESSION_TYPE`
  - `quadratic` / `log_linear` / `lightgbm`
  - 决定 `p_star_actual_quality.json` 的生成方式。
- `ROUND1A_COMPARE_REGRESSIONS`
  - `0` 或 `1`
  - 为 `1` 时，额外生成 `fit_compare/` 诊断目录。

示例：

```bash
ROUND1A_FIT_REGRESSION_TYPE=log_linear \
ROUND1A_COMPARE_REGRESSIONS=1 \
bash scripts/run_round1a.sh outputs/round1a_all
```

这会生成：

- `outputs/round1a_all/p_star_actual_quality.json`
- `outputs/round1a_all/round1a_actual_quality_design_summary.json`
- `outputs/round1a_all/design/`
- `outputs/round1a_all/visualizations/`
- `outputs/round1a_all/fit_compare/comparison_summary.json`
- `outputs/round1a_all/fit_compare/methods/*.json`
- `outputs/round1a_all/fit_compare/visualizations/`

---

## 2. 代码结构

### 2.1 目录结构

```
regmixer/
├── src/regmixer/
│   ├── cli.py                   # 主命令行接口
│   ├── train.py                 # 训练脚本入口
│   ├── synthesize_mixture.py    # 混合算法核心实现
│   ├── utils.py                 # 通用工具函数
│   ├── aliases.py               # 类型别名定义
│   ├── workspace.py             # 工作空间管理
│   │
│   ├── config/                  # 实验配置文件（YAML）
│   │   ├── dclm-example.yaml
│   │   ├── olmo2-*.yaml
│   │   ├── for_paper/
│   │   └── premix/
│   │
│   ├── data/                    # 数据处理模块
│   │   └── dataset.py         # 数据混合构建器
│   │
│   ├── model/                   # 模型配置模块
│   │   ├── aliases.py         # 模型别名
│   │   ├── evaluators.py      # 评估器定义
│   │   └── transformer.py    # Transformer 配置构建器
│   │
│   ├── eval/                    # 评估模块
│   │   ├── cli.py             # 评估命令行工具
│   │   ├── aliases.py         # 评估类型别名
│   │   ├── constants.py       # 评估常量（WandB 指标）
│   │   ├── utils.py           # 回归模型和工具函数
│   │   ├── evaluate_checkpoint.py  # 检查点评估
│   │   ├── law.py            # 缩放定律实现
│   │   └── config/           # 评估配置（YAML）
│   │
│   ├── internal/              # 内部工具模块
│   │   ├── cli.py            # 内部 CLI
│   │   ├── convert_mixture.py
│   │   ├── convert_cookbook_mixture.py
│   │   └── config/
│   │
│   └── scripts/
│       └── update_beaker_image.sh
│
├── eval/
│   └── convert_checkpoint.py  # 检查点转换工具
│
├── *.ipynb                   # Jupyter Notebooks（可视化分析）
├── *.sh                      # Shell 脚本（批量实验脚本）
├── pyproject.toml           # 项目配置
├── README.md
└── .github/workflows/        # CI/CD 配置
```

### 2.2 核心模块解析

#### 2.2.1 主 CLI 模块 (`cli.py`)

主命令行工具，提供以下核心命令：

- `launch` - 启动实验组
  - 支持配置文件或预定义混合方案
  - 支持 dry-run 模式（仅打印配置不执行）
  - 支持缓存管理
  - 使用线程池并行部署实验

- `generate_mixes` - 生成混合方案
  - 基于配置文件生成数据混合权重
  - 输出 JSON 格式的混合配置

- `validate` - 验证实验配置
  - 检查配置文件的正确性
  - 验证 Transformer 配置和数据集构建

- `status` - 查看实验状态
  - 查询实验组中所有任务的状态

- `cancel` - 取消实验
  - 停止实验组中的所有运行任务

#### 2.2.2 混合合成模块 (`synthesize_mixture.py`)

这是项目的核心算法实现，包含以下关键功能：

**核心函数**：

- `generate_weights_dirichlet()` - 使用狄利克雷分布生成权重
  - 支持源级别和主题级别的权重分配
  - 通过温度参数控制分布的均匀性
  - 支持权重边界检查（基于可用 token 数量）
  - 支持去重和最小权重约束

- `calculate_priors()` - 计算先验分布
  - 从数据源统计 token 数量
  - 支持缓存加速重复计算
  - 支持多种文件系统（S3、GCS、本地）

- `clip_candidates_by_level()` - 按层级裁剪权重
  - 对源级别和主题级别分别应用最小权重约束
  - 归一化处理确保权重和为 1

- `sort_and_deduplicate()` - 排序和去重
  - 移除重复的混合配置
  - 支持大规模数据集的高效去重（基于哈希）

**主要特性**：
- 支持 64 线程并行 token 计数
- 支持通配符路径扩展
- 自动生成权重分布直方图
- 可配置的重复次数约束

#### 2.2.3 训练模块 (`train.py`)

训练入口脚本，主要功能：

- 支持从 checkpoint 续训
- 集成 WandB 和配置保存回调
- 自动设置训练环境

**核心配置参数**：
- `max_tokens` - 最大训练 token 数
- `sources` - 数据源配置
- `sequence_length` - 序列长度
- `checkpoint_path` - 检查点路径（可选）
- `train_type` - 训练类型（pretrain/anneal）

#### 2.2.4 Transformer 配置构建器 (`model/transformer.py`)

智能配置构建器，自动计算训练参数：

**核心方法**：

- `get_batch_size()` - 自动计算全局批大小
  - 基于模型参数量和序列长度
  - 使用经验公式：`160 * (params/108M)^(2/3) / seq_divisor`

- `get_lr()` - 计算学习率
  - 基于模型参数量和序列长度
  - 公式：`0.0047 * (params/108M)^(-1/3)`

- `get_warmup_steps()` - 计算 warmup 步数
  - 基于模型参数量和批大小

- `get_scheduler()` - 配置学习率调度器
  - 支持 CosWithWarmupAndLinearDecay
  - 支持 LinearWithWarmup（用于 annealing）

**支持的回调函数**：
- WandB（实验追踪）
- Checkpointer（检查点保存）
- ConfigSaver（配置保存）
- GPUMemoryMonitor（GPU 内存监控）
- Profiler（性能分析）

#### 2.2.5 评估模块 (`eval/`)

提供完整的评估和回归分析功能：

**CLI 工具** (`eval/cli.py`)：

- `fit` - 拟合回归模型并提议混合方案
  - 支持 LightGBM、线性回归、对数线性回归
  - 支持多种提议器类型（simulation、search）
  - 支持约束优化（token 限制、pareto 改进）
  - 支持多种评估指标（WandB、Dashboard）

**常量定义** (`eval/constants.py`)：
- 定义 WandB 指标组
- 任务分类（如 MMLU、代码任务等）
- 目标权重配置

**工具函数** (`eval/utils.py`)：
- 回归模型构建和训练
- 相关系数计算和可视化
- 交互矩阵分析
- 混合方案优化

### 2.3 数据流

**数据混合流程**：
1. 读取配置文件，定义数据源和路径
2. 计算每个数据源的 token 数量（先验分布）
3. 使用狄利克雷分布生成多种权重组合
4. 应用约束条件（最小权重、边界、去重）
5. 输出混合方案配置

**训练流程**：
1. 从混合配置构建数据集
2. 使用 TransformerConfigBuilder 构建完整训练配置
3. 初始化模型、优化器和数据加载器
4. 启动训练，定期保存检查点
5. WandB 记录训练指标

**评估流程**：
1. 从 WandB 获取实验运行历史
2. 提取混合权重和评估指标
3. 训练回归模型（预测性能）
4. 优化混合权重（最小化目标函数）
5. 可视化结果并保存提议的混合方案

---

## 3. 使用指南

### 3.1 安装配置

#### 3.1.1 环境要求

- Python >= 3.9
- CUDA 环境（用于 GPU 训练）
- AWS/GCS 凭证（用于访问云存储）

#### 3.1.2 安装步骤

**基础安装**：
```bash
git clone https://github.com/allenai/regmixer.git
cd regmixer
pip install -e .
```

**安装完整依赖**：
```bash
pip install -e ".[all]"
```

**可选依赖**：
- `[dev]` - 开发工具（ruff、mypy、black、pytest）
- `[beaker]` - Beaker 集群支持
- `[wandb]` - WandB 实验追踪
- `[eval]` - 评估和回归分析

#### 3.1.3 环境变量配置

**AWS 凭证**（使用 S3 存储）：
```bash
export AWS_PROFILE=your_profile
```

**Google Cloud 凭证**（使用 GCS）：
```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON=/path/to/credentials.json
export GOOGLE_CLOUD_PROJECT=your_project
```

**WandB**：
```bash
wandb login
```

### 3.2 CLI 命令详解

#### 3.2.1 主命令 `rmc`

**启动实验组**：
```bash
rmc launch -c path/to/config.yaml
```

**选项**：
- `-c, --config` - 实验配置文件路径（必需）
- `-m, --mixture-file` - 预定义混合文件（可选）
- `--dry-run` - 仅打印配置，不实际执行
- `-n, --no-cache` - 不使用缓存

**生成混合方案**：
```bash
rmc generate_mixes -c path/to/config.yaml -o output.json
```

**验证配置**：
```bash
rmc validate -c path/to/config.yaml
```

**查看实验状态**：
```bash
rmc status -c path/to/config.yaml -g <group-id>
```

**取消实验**：
```bash
rmc cancel -c path/to/config.yaml -g <group-id>
```

#### 3.2.2 评估命令 `rmc-eval`

**拟合回归模型**：
```bash
rmc-eval fit -g <group-id> -c path/to/config.yaml
```

**常用选项**：
- `-g, --experiment-groups` - 实验组 ID（可多个）
- `-c, --config` - 配置文件路径（可多个）
- `-a, --alpha` - 模拟分布的 alpha 参数（默认 1.0）
- `-s, --num-samples` - 每个指标的评估样本数（默认 1）
- `-S, --simulation-samples` - 模拟样本数（默认 100,000）
- `-r, --regression-type` - 回归类型（lightgbm/linear/log_linear）
- `--proposer-type` - 提议器类型（simulation/search）

**高级选项**：
- `--constrain-swarm` - 仅使用约束的 swarm 运行
- `--constrain-objective` - 生成约束的混合方案
- `--obj-weights` - 非均匀的目标权重
- `--temperature` - 狄利克雷先验的温度参数
- `--fixed-weight` - 固定某些域的权重

#### 3.2.3 训练命令

直接训练入口（内部使用）：
```bash
python -m regmixer.train \
    --run-name <name> \
    --max-tokens <tokens> \
    --source <source_config> \
    --sequence-length 2048 \
    --seed 42
```

### 3.3 配置文件说明

#### 3.3.1 实验配置文件 (YAML)

**基础结构**：
```yaml
name: "实验名称"
description: "实验描述"
budget: "beaker budget"
workspace: "beaker workspace"
nodes: 1
gpus: 8
variants: 16              # 混合方案数量
preemptible: true
max_tokens: 2_910_233_600 # 最大训练 token 数
allow_repetition: false   # 是否允许重复数据
sequence_length: 2048
seed: 42
proxy_model_id: "olmo_30m"
tokenizer: "dolma2"
weka: false
dtype: "uint32"
priority: urgent
cluster: ai2/augusta-google-1
device_batch_size: 32
```

**混合参数**：
```yaml
source_temperature: 0.75  # 源级别温度（0-1，越小越均匀）
topic_temperature: 1.0   # 主题级别温度
minimum_weight: 0.0001   # 最小权重
min_strength: 0.1        # 最小狄利克雷强度
max_strength: 20         # 最大狄利克雷强度
sample_multiplier: 50    # 每个混合方案的尝试次数
```

**数据源配置**：
```yaml
sources:
  - name: adult_content
    paths:
      - s3://bucket/path/**/*.npy

  - name: software
    topics:              # 支持主题级别的细分
      - name: development
        paths:
          - s3://bucket/path/dev/**/*.npy
      - name: infrastructure
        paths:
          - s3://bucket/path/infra/**/*.npy
```

**约束条件**：
```yaml
nonzero_weight:          # 指定必须非零的数据源
  - adult_content
  - software

fixed_source_weights:    # 固定某些源的权重
  adult_content: 0.1
  software: 0.3

manual_prior:            # 手动指定先验分布
  adult_content: 0.046
  software: 0.053
```

#### 3.3.2 配置示例

完整示例见 `src/regmixer/config/dclm-example.yaml`

关键点：
- 支持 S3、GCS、本地路径
- 支持通配符（`**/*.npy`）
- 每个源可以定义主题细分
- 可以指定手动先验或自动计算

### 3.4 典型使用场景

#### 3.4.1 场景一：数据混合探索

**目标**：自动探索多种数据混合方案

```bash
# 1. 生成混合方案
rmc generate_mixes -c config/dclm-example.yaml -o mixes.json

# 2. 查看生成的方案
cat mixes.json

# 3. 启动实验
rmc launch -c config/dclm-example.yaml -m mixes.json
```

#### 3.4.2 场景二：基于回归优化混合

**目标**：利用历史实验数据优化数据混合

```bash
# 1. 拟合回归模型
rmc-eval fit \
    -g <experiment-group-id> \
    -c config/dclm-example.yaml \
    -r lightgbm \
    -S 100000

# 2. 查看优化结果
# 结果会保存在 cache/<group-id>/ 目录下
```

#### 3.4.3 场景三：从 checkpoint 续训

**目标**：在已训练模型基础上继续训练

```bash
# 在训练命令中指定 checkpoint 路径
python -m regmixer.train \
    --run-name continued_training \
    --max-tokens 5000000000 \
    --checkpoint-path /path/to/checkpoint \
    --train-type anneal \
    ...
```

#### 3.4.4 场景四：批量实验管理

**目标**：管理多个实验组

```bash
# 启动多个实验组
for config in config/*.yaml; do
    rmc launch -c $config
done

# 查看所有实验组状态
rmc status -c config/dclm-example.yaml -g <group-id-1>
rmc status -c config/dclm-example.yaml -g <group-id-2>

# 取消实验
rmc cancel -c config/dclm-example.yaml -g <group-id>
```

#### 3.4.5 场景五：可视化分析

使用 Jupyter Notebooks 进行交互式分析：

- `visualize_proposed_mixes.ipynb` - 可视化提议的混合方案
- `visualize_per_task_improvements.ipynb` - 可视化每个任务的改进
- `visualize_s2pdf_pes2o_mixes.ipynb` - S2PDF/PES2O 数据混合可视化

```bash
jupyter notebook visualize_proposed_mixes.ipynb
```

---

## 4. 开发指南

### 4.1 开发环境搭建

#### 4.1.1 克隆仓库

```bash
git clone https://github.com/allenai/regmixer.git
cd regmixer
```

#### 4.1.2 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

#### 4.1.3 安装开发依赖

```bash
pip install -e ".[dev,beaker,wandb,eval]"
```

这将安装：
- ruff（代码格式化）
- mypy（类型检查）
- black（代码格式化）
- isort（导入排序）
- pytest（测试框架）

### 4.2 依赖管理

#### 4.2.1 核心依赖

**必需依赖**（在 `pyproject.toml` 中定义）：
- ai2-olmo-core - AI2 核心库
- click - CLI 框架
- pydantic - 数据验证
- s3fs/gcsfs - 云存储接口
- pyarrow - 数据处理
- datasets - Hugging Face 数据集

**可选依赖**：
- beaker - Beaker 集群集成
- wandb - 实验追踪
- lightgbm - 回归模型
- pandas/numpy - 数据分析
- scipy/seaborn - 统计分析

#### 4.2.2 添加新依赖

编辑 `pyproject.toml`：

```toml
[project.dependencies]
# 添加新的依赖包
new-package = "^1.0.0"

[project.optional-dependencies]
dev = [
    # 添加新的开发工具
    "new-dev-tool",
]
```

然后重新安装：
```bash
pip install -e .
```

### 4.3 代码规范

#### 4.3.1 代码格式化

项目使用以下工具：

**Black**（代码格式化）：
```bash
black src/regmixer/
```

配置：
- 行长度：100
- 排除目录：`__pycache__`, `.git`, `.mypy_cache`, `.venv`, `doc`, `scratch/`, `build/`

**isort**（导入排序）：
```bash
isort src/regmixer/
```

配置：
- profile: black
- multi_line_output: 3

**Ruff**（快速 linter）：
```bash
ruff check src/regmixer/
```

配置：
- 行长度：115
- 忽略规则：`F403`, `F405`, `E501`
- 忽略文件：`__init__.py`（F401）

#### 4.3.2 类型检查

使用 mypy 进行类型检查：
```bash
mypy src/regmixer/
```

配置：
- 忽略缺失导入
- 无 site-packages
- 检查未定义类型

### 4.4 测试

#### 4.4.1 运行测试

```bash
pytest tests/
```

**pytest 配置**（在 `pyproject.toml` 中）：
- 测试路径：`tests/`
- 测试类命名：`Test*` 或 `*Test`
- 日志级别：DEBUG
- 忽略警告：
  - huggingface_hub 文件下载（FutureWarning）
  - pkg_resources（DeprecationWarning）
  - google.rpc（DeprecationWarning）
  - torch distributed checkpoint（FutureWarning）

#### 4.4.2 测试结构

虽然当前仓库中没有 `tests/` 目录，但计划支持以下测试结构：

```
tests/
├── test_cli.py           # 测试 CLI 命令
├── test_synthesis.py     # 测试混合合成
├── test_training.py      # 测试训练流程
└── test_evaluation.py    # 测试评估功能
```

### 4.5 开发工作流

#### 4.5.1 功能开发流程

1. 创建功能分支：
```bash
git checkout -b feature/new-feature
```

2. 编写代码：
   - 遵循代码规范
   - 添加类型注解
   - 编写文档字符串

3. 本地测试：
```bash
# 格式化代码
black src/regmixer/
isort src/regmixer/
ruff check src/regmixer/

# 类型检查
mypy src/regmixer/

# 运行测试
pytest tests/
```

4. 提交更改：
```bash
git add .
git commit -m "feat: add new feature"
```

#### 4.5.2 提交信息规范

使用 Conventional Commits 规范：

- `feat:` - 新功能
- `fix:` - 修复 bug
- `docs:` - 文档更新
- `style:` - 代码格式调整
- `refactor:` - 代码重构
- `test:` - 测试相关
- `chore:` - 构建/工具相关

示例：
```bash
git commit -m "feat: add support for new tokenizer"
git commit -m "fix: resolve issue with token counting"
```

#### 4.5.3 CI/CD

项目使用 GitHub Actions 进行持续集成（`.github/workflows/main.yaml`）：

- 触发条件：推送标签
- 功能：自动发布到 PyPI

### 4.6 调试技巧

#### 4.6.1 启用调试日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 4.6.2 使用 dry-run 模式

```bash
rmc launch -c config.yaml --dry-run
```

这将打印配置而不实际执行，便于调试。

#### 4.6.3 WandB 调试

```bash
# 查看实验
wandb ai2-llm/regmixer

# 查看特定实验组
wandb ai2-llm/regmixer/groups/<group-id>
```

### 4.7 常见问题

#### 4.7.1 认证问题

**S3 认证**：
```bash
export AWS_PROFILE=your_profile
aws configure
```

**GCS 认证**：
```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON=/path/to/key.json
gcloud auth application-default login
```

**Beaker 认证**：
```bash
beaker login
```

#### 4.7.2 内存问题

如果遇到 OOM 错误：

1. 减小 `device_batch_size`
2. 减小 `sequence_length`
3. 减小模型参数量
4. 使用梯度累积

#### 4.7.3 速度优化

**加速 token 计数**：
- 使用缓存（默认启用）
- 增加线程数（已默认使用 64 线程）

**加速训练**：
- 使用多个 GPU
- 增加 `global_batch_size`
- 使用混合精度训练

### 4.8 贡献指南

#### 4.8.1 提交 Issue

在提交 issue 时，请提供：
- 清晰的标题和描述
- 复现步骤
- 预期行为 vs 实际行为
- 环境信息（Python 版本、依赖版本）

#### 4.8.2 Pull Request

提交 PR 时，请：
- 提供清晰的 PR 描述
- 关联相关 issue
- 确保所有测试通过
- 更新相关文档
- 遵循代码规范

#### 4.8.3 代码审查

PR 需要通过：
- 至少一位维护者的审查
- CI 检查（如果配置）
- 代码风格检查（black、ruff、mypy）

---

## 5. 附录

### 5.1 命令速查表

| 命令 | 功能 | 示例 |
|------|------|------|
| `rmc launch` | 启动实验 | `rmc launch -c config.yaml` |
| `rmc generate_mixes` | 生成混合方案 | `rmc generate_mixes -c config.yaml` |
| `rmc validate` | 验证配置 | `rmc validate -c config.yaml` |
| `rmc status` | 查看状态 | `rmc status -c config.yaml -g <id>` |
| `rmc cancel` | 取消实验 | `rmc cancel -c config.yaml -g <id>` |
| `rmc-eval fit` | 拟合回归 | `rmc-eval fit -g <id> -c config.yaml` |

### 5.2 配置参数速查表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_tokens` | 最大训练 token 数 | - |
| `variants` | 混合方案数量 | - |
| `sequence_length` | 序列长度 | 2048 |
| `seed` | 随机种子 | 42 |
| `source_temperature` | 源级别温度 | 1.0 |
| `topic_temperature` | 主题级别温度 | 1.0 |
| `minimum_weight` | 最小权重 | 0.002 |
| `min_strength` | 最小狄利克雷强度 | 0.1 |
| `max_strength` | 最大狄利克雷强度 | 5.0 |
| `sample_multiplier` | 尝试倍数 | 10 |
| `allow_repetition` | 是否允许重复 | false |

### 5.3 相关资源

- **GitHub 仓库**：https://github.com/allenai/regmixer
- **OLMo-core**：https://github.com/allenai/OLMo-core
- **Beaker**：https://github.com/allenai/beaker
- **WandB 文档**：https://docs.wandb.ai/

### 5.4 许可证

本项目由 Allen Institute for Artificial Intelligence 开发，具体许可证信息请参考项目根目录的 LICENSE 文件。

---

**文档版本**：1.0
**最后更新**：2025-01-26
**维护者**：Allen Institute for Artificial Intelligence
