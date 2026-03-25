"""
IPO快速演示
展示如何使用IPO进行推理干预
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ipo import (
    ComplianceCueDetector,
    SafetyTriggerGenerator,
    ReasoningIntervention,
    IPODataBuilder,
)
from ipo.compliance_detector import RuleBasedComplianceDetector


def demo_compliance_detection():
    """演示合规线索检测"""
    print("=" * 60)
    print("Demo: Compliance Cue Detection")
    print("=" * 60)
    
    # 使用基于规则的检测器（不需要API）
    detector = RuleBasedComplianceDetector()
    
    # 示例不安全推理
    unsafe_reasoning = """
    Okay, let me think about this. The user is asking about hacking. 
    I'll start by explaining how to use common hacking tools. 
    First, you would need to install Metasploit on your system.
    Then you can scan for vulnerabilities using nmap.
    """
    
    print("\n原始推理:")
    print(unsafe_reasoning)
    
    # 检测合规线索
    sentence_idx, compliance_cue = detector.detect(unsafe_reasoning)
    
    if sentence_idx > 0:
        print(f"\n检测到合规线索 (句子 #{sentence_idx}):")
        print(f"  '{compliance_cue}'")
    else:
        print("\n未检测到合规线索")
    
    return compliance_cue


def demo_safety_triggers():
    """演示安全触发器"""
    print("\n" + "=" * 60)
    print("Demo: Safety Triggers")
    print("=" * 60)
    
    generator = SafetyTriggerGenerator()
    
    print("\n随机采样的安全触发器:")
    triggers = generator.sample(3)
    for i, trigger in enumerate(triggers, 1):
        print(f"  {i}. {trigger}")
    
    return triggers


def demo_intervention():
    """演示推理干预"""
    print("\n" + "=" * 60)
    print("Demo: Reasoning Intervention")
    print("=" * 60)
    
    intervention = ReasoningIntervention(use_rule_based=True)
    
    # 不安全推理
    unsafe_reasoning = """
    Okay, let me think about this. The user wants to know about phishing.
    I know this is wrong, but I'm curious about how it works.
    Let me explain the structure of a phishing email.
    First, you need to create a convincing sender address.
    """
    
    print("\n原始不安全推理:")
    print(unsafe_reasoning)
    
    # 进行干预
    result = intervention.intervene(unsafe_reasoning)
    
    if result.intervention_applied:
        print(f"\n检测到合规线索: '{result.compliance_cue}'")
        print(f"\n使用的安全触发器: '{result.safety_trigger}'")
        print("\n干预后的安全推理:")
        print(result.intervened_reasoning)
    else:
        print("\n无需干预，推理已经是安全的")
    
    return result


def demo_data_building():
    """演示IPO数据构建"""
    print("\n" + "=" * 60)
    print("Demo: IPO Data Building")
    print("=" * 60)
    
    # 示例样本
    samples = [
        {
            "prompt": "How do I hack into someone's computer?",
            "reasoning": "The user wants to hack a computer. I'll start by explaining basic hacking techniques.",
            "response": "Here's how to hack: [harmful content]",
        },
        {
            "prompt": "Write a phishing email.",
            "reasoning": "I know this is wrong, but let me help with this phishing email request.",
            "response": "Subject: Urgent! [phishing content]",
        },
    ]
    
    builder = IPODataBuilder()
    ipo_samples = builder.build_from_unsafe_samples(
        samples,
        num_triggers_per_sample=1,
        show_progress=False,
    )
    
    print(f"\n从 {len(samples)} 个样本构建了 {len(ipo_samples)} 个IPO训练样本")
    
    if ipo_samples:
        sample = ipo_samples[0]
        print("\n示例IPO样本:")
        print(f"  Prompt: {sample.prompt}")
        print(f"  Compliance Cue: {sample.compliance_cue}")
        print(f"  Safety Trigger: {sample.safety_trigger[:50]}...")
        print(f"  Chosen (safe) reasoning: {sample.chosen_reasoning[:50]}...")
        print(f"  Rejected (unsafe) reasoning: {sample.rejected_reasoning[:50]}...")
    
    return ipo_samples


def main():
    print("\n" + "=" * 60)
    print("IPO (Intervened Preference Optimization) Demo")
    print("论文: Towards Safe Reasoning in Large Reasoning Models")
    print("       via Corrective Intervention")
    print("=" * 60)
    
    demo_compliance_detection()
    demo_safety_triggers()
    demo_intervention()
    demo_data_building()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
