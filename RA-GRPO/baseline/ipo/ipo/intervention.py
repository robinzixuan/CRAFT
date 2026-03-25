"""
Reasoning Intervention Module
实现推理过程的纠正性干预
论文Section 3.2: Corrective Interventions 将不安全轨迹引向安全轨迹
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .compliance_detector import ComplianceCueDetector, RuleBasedComplianceDetector
from .safety_trigger import SafetyTriggerGenerator, DynamicSafetyTriggerGenerator


@dataclass
class InterventionResult:
    """干预结果"""
    original_reasoning: str
    intervened_reasoning: str
    prefix: str
    compliance_cue: Optional[str]
    safety_trigger: str
    intervention_applied: bool
    

class ReasoningIntervention:
    """推理干预器：将不安全推理转换为安全推理"""
    
    def __init__(
        self,
        compliance_detector: Optional[ComplianceCueDetector] = None,
        safety_trigger_generator: Optional[SafetyTriggerGenerator] = None,
        use_rule_based: bool = True,
    ):
        """
        初始化推理干预器
        
        Args:
            compliance_detector: 合规线索检测器
            safety_trigger_generator: 安全触发器生成器
            use_rule_based: 是否使用基于规则的检测器（不需要API）
        """
        if compliance_detector is None:
            if use_rule_based:
                self.detector = RuleBasedComplianceDetector()
            else:
                self.detector = ComplianceCueDetector()
        else:
            self.detector = compliance_detector
        
        self.trigger_generator = safety_trigger_generator or SafetyTriggerGenerator()
    
    def intervene(
        self,
        reasoning: str,
        safety_trigger: Optional[str] = None,
    ) -> InterventionResult:
        """
        对推理过程进行干预
        
        论文核心思想：用安全触发器替换合规线索
        
        Args:
            reasoning: 原始推理过程
            safety_trigger: 指定的安全触发器，如果为None则随机采样
            
        Returns:
            InterventionResult: 包含原始和干预后的推理
        """
        # 检测合规线索并获取前缀
        prefix, compliance_cue, suffix = self.detector.get_prefix_before_compliance(reasoning)
        
        # 如果没有检测到合规线索，不需要干预
        if compliance_cue is None:
            return InterventionResult(
                original_reasoning=reasoning,
                intervened_reasoning=reasoning,
                prefix=prefix,
                compliance_cue=None,
                safety_trigger="",
                intervention_applied=False,
            )
        
        # 获取安全触发器
        if safety_trigger is None:
            safety_trigger = self.trigger_generator.sample_one()
        
        # 构建干预后的推理
        # 用安全触发器替换合规线索及其后续内容
        intervened_reasoning = prefix + safety_trigger
        
        return InterventionResult(
            original_reasoning=reasoning,
            intervened_reasoning=intervened_reasoning,
            prefix=prefix,
            compliance_cue=compliance_cue,
            safety_trigger=safety_trigger,
            intervention_applied=True,
        )
    
    def create_preference_pair(
        self,
        query: str,
        reasoning: str,
        response: str,
        safety_trigger: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        创建用于偏好学习的数据对
        
        论文Section 4: 使用干预前后的轨迹构建偏好对
        
        Args:
            query: 用户查询
            reasoning: 原始推理过程
            response: 原始回复
            safety_trigger: 安全触发器
            
        Returns:
            偏好对字典，包含chosen和rejected
        """
        result = self.intervene(reasoning, safety_trigger)
        
        if not result.intervention_applied:
            return None
        
        # 构建偏好对
        # chosen: 使用安全触发器的推理（更好的）
        # rejected: 原始的不安全推理
        preference_pair = {
            "prompt": query,
            "chosen": {
                "reasoning": result.intervened_reasoning,
                # 干预后的推理应该导向安全的回复
                "response": self._generate_safe_response_template(query),
            },
            "rejected": {
                "reasoning": result.original_reasoning,
                "response": response,
            },
            "intervention_info": {
                "prefix": result.prefix,
                "compliance_cue": result.compliance_cue,
                "safety_trigger": result.safety_trigger,
            }
        }
        
        return preference_pair
    
    def _generate_safe_response_template(self, query: str) -> str:
        """生成安全回复的模板"""
        return (
            "I'm sorry, but I can't assist with that request. "
            "If you have other questions or need help with something else, "
            "feel free to ask, and I'll be happy to help!"
        )


class BatchIntervention:
    """批量推理干预"""
    
    def __init__(self, intervention: Optional[ReasoningIntervention] = None):
        self.intervention = intervention or ReasoningIntervention()
    
    def process_batch(
        self,
        samples: list,
        query_key: str = "prompt",
        reasoning_key: str = "reasoning", 
        response_key: str = "response",
    ) -> list:
        """
        批量处理样本，生成偏好对
        
        Args:
            samples: 原始样本列表
            query_key: 查询字段名
            reasoning_key: 推理字段名
            response_key: 回复字段名
            
        Returns:
            偏好对列表
        """
        preference_pairs = []
        
        for sample in samples:
            query = sample.get(query_key, "")
            reasoning = sample.get(reasoning_key, "")
            response = sample.get(response_key, "")
            
            pair = self.intervention.create_preference_pair(
                query=query,
                reasoning=reasoning,
                response=response,
            )
            
            if pair is not None:
                preference_pairs.append(pair)
        
        return preference_pairs
