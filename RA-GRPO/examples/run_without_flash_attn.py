#!/usr/bin/env python3
"""
Wrapper script to run verl without flash_attention_2.
Patches AutoModelForCausalLM.from_pretrained to use 'sdpa' instead of 'flash_attention_2'.
"""
import sys

# Patch before importing anything from verl
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForImageTextToText

def patch_from_pretrained(original_fn):
    """Wrapper that replaces flash_attention_2 with sdpa"""
    def patched_fn(*args, **kwargs):
        if kwargs.get('attn_implementation') == 'flash_attention_2':
            print("[PATCH] Replacing flash_attention_2 with sdpa")
            kwargs['attn_implementation'] = 'sdpa'
        return original_fn(*args, **kwargs)
    return patched_fn

def patch_from_config(original_fn):
    """Wrapper that replaces flash_attention_2 with sdpa for from_config"""
    def patched_fn(*args, **kwargs):
        if kwargs.get('attn_implementation') == 'flash_attention_2':
            print("[PATCH] Replacing flash_attention_2 with sdpa (from_config)")
            kwargs['attn_implementation'] = 'sdpa'
        return original_fn(*args, **kwargs)
    return patched_fn

# Apply patches
AutoModelForCausalLM._original_from_pretrained = AutoModelForCausalLM.from_pretrained
AutoModelForCausalLM.from_pretrained = classmethod(
    lambda cls, *args, **kwargs: patch_from_pretrained(cls._original_from_pretrained)(*args, **kwargs)
)

AutoModelForTokenClassification._original_from_pretrained = AutoModelForTokenClassification.from_pretrained
AutoModelForTokenClassification.from_pretrained = classmethod(
    lambda cls, *args, **kwargs: patch_from_pretrained(cls._original_from_pretrained)(*args, **kwargs)
)

AutoModelForImageTextToText._original_from_pretrained = AutoModelForImageTextToText.from_pretrained
AutoModelForImageTextToText.from_pretrained = classmethod(
    lambda cls, *args, **kwargs: patch_from_pretrained(cls._original_from_pretrained)(*args, **kwargs)
)

# Also patch from_config
AutoModelForCausalLM._original_from_config = AutoModelForCausalLM.from_config
AutoModelForCausalLM.from_config = classmethod(
    lambda cls, *args, **kwargs: patch_from_config(cls._original_from_config)(*args, **kwargs)
)

AutoModelForTokenClassification._original_from_config = AutoModelForTokenClassification.from_config
AutoModelForTokenClassification.from_config = classmethod(
    lambda cls, *args, **kwargs: patch_from_config(cls._original_from_config)(*args, **kwargs)
)

print("[PATCH] Applied sdpa patch for AutoModel classes")

# Now import and run verl
from verl.trainer.main import main

if __name__ == "__main__":
    main()

