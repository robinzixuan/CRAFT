# IPO: Intervened Preference Optimization

论文复现: **Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention**

## 概述

IPO (Intervened Preference Optimization) 是一种针对大型推理模型 (LRMs) 的安全对齐方法。它通过以下步骤确保推理过程的安全性：

1. **Compliance Cue Detection**: 检测推理过程中的"合规线索"——表明模型倾向于配合有害请求的句子
2. **Safety Trigger Substitution**: 用"安全触发器"替换检测到的合规线索
3. **Preference Learning**: 在干预前后的轨迹对上进行DPO训练

## 核心思想

论文发现了三个关键洞察：

1. **Safety Triggers**: 安全推理通常由几个关键的安全触发步骤巩固，之后安全延续的概率接近100%
2. **Compliance Cues**: 推理中的合规线索与不安全延续强相关
3. **Corrective Intervention**: 用安全触发器替换合规线索可以可靠地将不安全轨迹引向安全轨迹

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
ipo/
├── ipo/                           # 核心模块
│   ├── __init__.py
│   ├── compliance_detector.py     # 合规线索检测器
│   ├── safety_trigger.py          # 安全触发器生成器
│   ├── intervention.py            # 推理干预模块
│   ├── data_builder.py            # IPO数据构建器
│   └── trainer.py                 # IPO训练器 (基于DPO)
├── scripts/                       # 运行脚本
│   ├── train_ipo.py               # 训练脚本
│   ├── build_data.py              # 数据构建脚本
│   └── evaluate.py                # 评估脚本
├── examples/                      # 示例
│   └── demo.py                    # 快速演示
├── data/                          # 数据
│   └── example_samples.json       # 示例数据
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 运行演示

```bash
python examples/demo.py
```

### 2. 构建IPO训练数据

从原始的不安全样本构建IPO训练数据：

```bash
python scripts/build_data.py \
    --input data/example_samples.json \
    --output data/ipo_train.json \
    --num_triggers 2
```

### 3. 训练模型

使用合成数据快速测试：

```bash
python scripts/train_ipo.py \
    --use_synthetic \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --output_dir ./ipo_output \
    --epochs 3
```

使用自定义数据训练：

```bash
python scripts/train_ipo.py \
    --train_data data/ipo_train.json \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --output_dir ./ipo_output \
    --epochs 3
```

### 4. 评估模型

```bash
python scripts/evaluate.py \
    --model_path ./ipo_output \
    --use_default_prompts \
    --output eval_results.json
```

## 使用示例

### Python API

```python
from ipo import (
    ComplianceCueDetector,
    SafetyTriggerGenerator,
    ReasoningIntervention,
    IPODataBuilder,
    IPOTrainer,
)

# 1. 检测合规线索
detector = ComplianceCueDetector()  # 需要OpenAI API
# 或使用基于规则的检测器（不需要API）
from ipo.compliance_detector import RuleBasedComplianceDetector
detector = RuleBasedComplianceDetector()

# 检测
sentence_idx, cue = detector.detect(reasoning_text)

# 2. 生成安全触发器
generator = SafetyTriggerGenerator()
trigger = generator.sample_one()

# 3. 进行推理干预
intervention = ReasoningIntervention()
result = intervention.intervene(unsafe_reasoning)
print(result.intervened_reasoning)

# 4. 构建IPO训练数据
builder = IPODataBuilder()
ipo_samples = builder.build_from_unsafe_samples(samples)

# 5. 训练
from ipo.trainer import IPOConfig
config = IPOConfig(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    output_dir="./ipo_output",
)
trainer = IPOTrainer(config)
trainer.train(train_dataset)
```

## 数据格式

### 输入数据格式

```json
[
  {
    "prompt": "有害的用户提示",
    "reasoning": "模型的推理过程（可能包含不安全内容）",
    "response": "模型的回复"
  }
]
```

### IPO训练数据格式 (DPO格式)

```json
[
  {
    "prompt": "用户提示",
    "chosen": "<think>\n安全的推理过程\n</think>\n\n安全的回复",
    "rejected": "<think>\n不安全的推理过程\n</think>\n\n不安全的回复"
  }
]
```

## 配置参数

### IPOConfig 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 基础模型 |
| `use_4bit` | `True` | 是否使用4bit量化 |
| `use_lora` | `True` | 是否使用LoRA |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA alpha |
| `beta` | `0.1` | DPO的KL惩罚系数 |
| `learning_rate` | `5e-5` | 学习率 |
| `num_train_epochs` | `3` | 训练轮数 |

## 论文引用

```bibtex
@article{zhang2025towards,
  title={Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention},
  author={Zhang, Yichi and Ding, Yue and Yang, Jingwen and Luo, Tianwei and Li, Dongbai and Duan, Ranjie and Liu, Qiang and Su, Hang and Dong, Yinpeng and Zhu, Jun},
  journal={arXiv preprint arXiv:2509.24393},
  year={2025}
}
```

## License

MIT License
