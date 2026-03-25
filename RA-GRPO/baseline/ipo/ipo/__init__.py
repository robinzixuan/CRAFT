"""
Intervened Preference Optimization (IPO)
论文复现: Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention
"""

from .compliance_detector import ComplianceCueDetector
from .safety_trigger import SafetyTriggerGenerator
from .intervention import ReasoningIntervention
from .data_builder import IPODataBuilder

# 延迟导入trainer，因为它依赖torch等大型库
def __getattr__(name):
    if name == "IPOTrainer":
        from .trainer import IPOTrainer
        return IPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ComplianceCueDetector",
    "SafetyTriggerGenerator", 
    "ReasoningIntervention",
    "IPODataBuilder",
    "IPOTrainer",
]
