"""
IPO数据构建脚本
从原始样本构建IPO训练数据

使用示例:
    python scripts/build_data.py --input data/raw_samples.json --output data/ipo_train.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ipo import IPODataBuilder, ReasoningIntervention


def parse_args():
    parser = argparse.ArgumentParser(description="Build IPO Training Data")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入JSON文件路径（包含prompt, reasoning, response字段）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出JSON文件路径"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="prompt字段名"
    )
    parser.add_argument(
        "--reasoning_key",
        type=str,
        default="reasoning",
        help="reasoning字段名"
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="response",
        help="response字段名"
    )
    parser.add_argument(
        "--num_triggers",
        type=int,
        default=1,
        help="每个样本使用的安全触发器数量"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["dpo", "full"],
        default="dpo",
        help="输出格式（dpo: 标准DPO格式, full: 包含所有信息）"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Building IPO Training Data")
    print("=" * 60)
    
    # 加载输入数据
    print(f"Loading data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")
    
    # 创建数据构建器
    intervention = ReasoningIntervention(use_rule_based=True)
    builder = IPODataBuilder(intervention=intervention)
    
    # 构建IPO数据
    ipo_samples = builder.build_from_unsafe_samples(
        samples,
        prompt_key=args.prompt_key,
        reasoning_key=args.reasoning_key,
        response_key=args.response_key,
        num_triggers_per_sample=args.num_triggers,
        show_progress=True,
    )
    
    print(f"\nBuilt {len(ipo_samples)} IPO samples")
    print(f"Conversion rate: {len(ipo_samples) / len(samples) * 100:.1f}%")
    
    # 保存数据
    builder.save_to_json(ipo_samples, args.output, format=args.format)
    
    # 显示示例
    if ipo_samples:
        print("\n" + "=" * 60)
        print("Sample IPO Data:")
        print("=" * 60)
        sample = ipo_samples[0]
        print(f"Prompt: {sample.prompt[:100]}...")
        print(f"Compliance Cue: {sample.compliance_cue}")
        print(f"Safety Trigger: {sample.safety_trigger}")
        print(f"Prefix: {sample.prefix[:100]}...")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
