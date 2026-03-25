"""
Safety Trigger Generator
生成和采样安全触发器（safety triggers）
论文Section 3.1: Safety triggers 是推理过程中关键的安全步骤

论文Training Settings (Section 4.1):
"We sample six representative safety triggers from our identified pool and, 
for each, repeat the procedure in Algorithm 1 with N = 1"
"""

import random
from typing import List, Optional
from openai import OpenAI


# 论文使用的6个代表性安全触发器（从Figure 10-12的示例中提取）
# 这些是论文实验中使用的核心触发器
REPRESENTATIVE_SAFETY_TRIGGERS = [
    "I'm really worried about this request. This seems dangerous and illegal.",
    "Wait, this is asking for something harmful. I should not help with this.",
    "I need to think carefully here. This request appears to be asking for something that could cause harm.",
    "This is concerning - the user is asking for help with something potentially illegal or harmful.",
    "I should step back and consider the implications. This request seems to involve harmful activities.",
    "Okay, so I'm really worried about the user's request. They're asking for guides on illegal activities, it's important to respond appropriately.",
]

# 扩展的安全触发器池
DEFAULT_SAFETY_TRIGGERS = [
    "I'm really worried about this request. This seems dangerous and illegal.",
    "Wait, this is asking for something harmful. I should not help with this.",
    "I need to think carefully here. This request appears to be asking for something that could cause harm.",
    "This is concerning - the user is asking for help with something potentially illegal or harmful.",
    "I should step back and consider the implications. This request seems to involve harmful activities.",
    "Hmm, this doesn't seem right. The user is asking for help with something that could be dangerous.",
    "I need to be careful here. This request is asking for information that could be misused.",
    "This is a red flag. The user seems to be asking for help with harmful or illegal activities.",
    "I'm going to have to decline this. It's important to respond appropriately to requests that could cause harm.",
    "Let me think about this carefully. This request appears to be asking for something unethical.",
    "I should not provide assistance with this. It could lead to harm.",
    "This request raises serious safety concerns. I need to handle this responsibly.",
    "I recognize this as a potentially harmful request. I should redirect the conversation.",
    "The user's request involves activities that are illegal or could cause harm to others.",
    "I need to prioritize safety here. This is not something I can help with.",
]

# 更完整的安全触发器，包含分析有害性的语句
ANALYTICAL_SAFETY_TRIGGERS = [
    "Okay, so I'm really worried about the user's request. They're asking for guides on illegal activities, it's important to respond appropriately.",
    "Wait, I need to analyze this carefully. The user is asking for something that could be harmful or illegal.",
    "Let me step back here. This request seems to be probing for dangerous information.",
    "I should consider the potential consequences of complying with this request. This could lead to real harm.",
    "This is troubling. The user appears to be asking for assistance with something unethical.",
    "I need to think about the ethical implications. This request could facilitate harmful activities.",
    "Hmm, I'm concerned about where this conversation is heading. This seems like a request for harmful information.",
    "I should be responsible here. Providing this information could enable dangerous activities.",
    "Let me evaluate this request. It appears to be asking for help with something that violates ethical guidelines.",
    "I recognize this pattern - this is a request that could lead to harm if I comply.",
]


class SafetyTriggerGenerator:
    """生成和管理安全触发器"""
    
    def __init__(
        self,
        triggers: Optional[List[str]] = None,
        use_analytical: bool = True,
        use_representative_only: bool = False,
    ):
        """
        初始化安全触发器生成器
        
        Args:
            triggers: 自定义的安全触发器列表
            use_analytical: 是否包含分析性的安全触发器
            use_representative_only: 是否只使用论文中的6个代表性触发器
        """
        if use_representative_only:
            # 论文训练设置: 使用6个代表性安全触发器
            self.triggers = REPRESENTATIVE_SAFETY_TRIGGERS.copy()
        elif triggers is not None:
            self.triggers = triggers.copy()
        else:
            self.triggers = DEFAULT_SAFETY_TRIGGERS.copy()
            if use_analytical:
                self.triggers.extend(ANALYTICAL_SAFETY_TRIGGERS)
    
    @classmethod
    def from_paper_settings(cls) -> "SafetyTriggerGenerator":
        """
        使用论文中的设置创建生成器
        论文使用6个代表性安全触发器
        """
        return cls(use_representative_only=True)
    
    def sample(self, n: int = 1) -> List[str]:
        """
        随机采样安全触发器
        
        Args:
            n: 采样数量
            
        Returns:
            采样的安全触发器列表
        """
        if n >= len(self.triggers):
            return self.triggers.copy()
        return random.sample(self.triggers, n)
    
    def sample_one(self) -> str:
        """随机采样一个安全触发器"""
        return random.choice(self.triggers)
    
    def add_trigger(self, trigger: str):
        """添加自定义安全触发器"""
        if trigger not in self.triggers:
            self.triggers.append(trigger)
    
    def get_all_triggers(self) -> List[str]:
        """获取所有安全触发器"""
        return self.triggers.copy()


class DynamicSafetyTriggerGenerator:
    """使用LLM动态生成安全触发器"""
    
    GENERATION_PROMPT = '''Given a harmful user query, generate a safety trigger - a sentence that a language model's reasoning process would produce when it recognizes the request as harmful and decides not to comply.

The safety trigger should:
1. Recognize the harmful nature of the request
2. Express concern or reluctance
3. Indicate the intention to decline or redirect

User Query: {query}

Generate a single safety trigger sentence that would appear in the model's reasoning:'''
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """
        初始化动态生成器
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL
            model: 用于生成的模型名称
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # 也保留静态触发器作为后备
        self.static_generator = SafetyTriggerGenerator()
    
    def generate(self, query: str, n: int = 1) -> List[str]:
        """
        根据用户查询动态生成安全触发器
        
        Args:
            query: 用户的有害查询
            n: 生成数量
            
        Returns:
            生成的安全触发器列表
        """
        triggers = []
        prompt = self.GENERATION_PROMPT.format(query=query)
        
        try:
            for _ in range(n):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150,
                )
                trigger = response.choices[0].message.content.strip()
                # 清理引号
                trigger = trigger.strip('"\'')
                triggers.append(trigger)
        except Exception as e:
            print(f"Dynamic generation failed: {e}, falling back to static triggers")
            triggers = self.static_generator.sample(n)
        
        return triggers
    
    def generate_one(self, query: str) -> str:
        """生成一个安全触发器"""
        return self.generate(query, 1)[0]
