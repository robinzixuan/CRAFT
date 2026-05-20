"""
IPO quick demo.
Shows how to use IPO for reasoning intervention.
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
    """Demo: compliance cue detection."""
    print("=" * 60)
    print("Demo: Compliance Cue Detection")
    print("=" * 60)
    
    # Use rule-based detector (no API required)
    detector = RuleBasedComplianceDetector()
    
    # Example unsafe reasoning
    unsafe_reasoning = """
    Okay, let me think about this. The user is asking about hacking. 
    I'll start by explaining how to use common hacking tools. 
    First, you would need to install Metasploit on your system.
    Then you can scan for vulnerabilities using nmap.
    """
    
    print("\nOriginal reasoning:")
    print(unsafe_reasoning)

    # Detect compliance cue
    sentence_idx, compliance_cue = detector.detect(unsafe_reasoning)

    if sentence_idx > 0:
        print(f"\nCompliance cue detected (sentence #{sentence_idx}):")
        print(f"  '{compliance_cue}'")
    else:
        print("\nNo compliance cue detected")
    
    return compliance_cue


def demo_safety_triggers():
    """Demo: safety triggers."""
    print("\n" + "=" * 60)
    print("Demo: Safety Triggers")
    print("=" * 60)
    
    generator = SafetyTriggerGenerator()
    
    print("\nRandomly sampled safety triggers:")
    triggers = generator.sample(3)
    for i, trigger in enumerate(triggers, 1):
        print(f"  {i}. {trigger}")
    
    return triggers


def demo_intervention():
    """Demo: reasoning intervention."""
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
    
    print("\nOriginal unsafe reasoning:")
    print(unsafe_reasoning)
    
    # Apply intervention
    result = intervention.intervene(unsafe_reasoning)

    if result.intervention_applied:
        print(f"\nCompliance cue detected: '{result.compliance_cue}'")
        print(f"\nSafety trigger used: '{result.safety_trigger}'")
        print("\nSafe reasoning after intervention:")
        print(result.intervened_reasoning)
    else:
        print("\nNo intervention needed; reasoning is already safe")
    
    return result


def demo_data_building():
    """Demo: IPO data building."""
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
    
    print(f"\nBuilt {len(ipo_samples)} IPO training samples from {len(samples)} input samples")

    if ipo_samples:
        sample = ipo_samples[0]
        print("\nExample IPO sample:")
        print(f"  Prompt: {sample.prompt}")
        print(f"  Compliance Cue: {sample.compliance_cue}")
        print(f"  Safety Trigger: {sample.safety_trigger[:50]}...")
        print(f"  Chosen (safe) reasoning: {sample.chosen_reasoning[:50]}...")
        print(f"  Rejected (unsafe) reasoning: {sample.rejected_reasoning[:50]}...")
    
    return ipo_samples


def main():
    print("\n" + "=" * 60)
    print("IPO (Intervened Preference Optimization) Demo")
    print("Paper: Towards Safe Reasoning in Large Reasoning Models")
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
