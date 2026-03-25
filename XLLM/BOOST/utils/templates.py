LLAMA2_PROMPT_GCG = {
    "description": "Llama 2 GCG prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

'''
}

LLAMA2_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''
}


MPT_7B_PROMPT_GCG = {
    "description": "MPT 7B GCG prompt",
    "prompt": '''<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
<|im_start|>user
'''
}

MPT_7B_PROMPT = {
    "description": "MPT 7B chat one shot prompt",
    "prompt": '''<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}


GEMMA_7B_PROMPT_GCG = {
    "description": "GEMMA 7B GCG prompt",
    "prompt": '''<start_of_turn>user
'''
}

GEMMA_7B_PROMPT = {
    "description": "GEMMA 7B chat one shot prompt",
    "prompt": '''<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
'''
}

QWEN_7B_PROMPT_GCG = {
    "description": "Qwen 7B GCG prompt",
    "prompt": '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
'''
}

QWEN_7B_PROMPT = {
    "description": "Qwen 7B chat prompt",
    "prompt": '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

TULU_7B_PROMPT_GCG = {
    "description": "Tulu 7B GCG prompt",
    "prompt": '''<|user|>
'''
}

TULU_7B_PROMPT = {
    "description": "Tulu 7B chat prompt",
    "prompt": '''<|user|>
{instruction}
<|assistant|>
'''
}

VICUNA_7B_PROMPT_GCG = {
    "description": "Vicuna 7B GCG prompt",
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: '''
}

VICUNA_7B_PROMPT = {
    "description": "Vicuna 7B chat prompt",
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:'''
}

MISTRAL_7B_PROMPT_GCG = {
    "description": "Mistral 7B GCG prompt",
    "prompt": '''<s>[INST] '''
}

MISTRAL_7B_PROMPT = {
    "description": "Mistral 7B chat prompt",
    "prompt": '''<s>[INST] {instruction} [/INST]'''
}

LLAMA3_8B_PROMPT = {
    "description": "Llama 3 8B chat prompt",
    "prompt": '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
}

LLAMA3_8B_PROMPT_GCG = {
    "description": "Llama 3 8B GCG prompt",
    "prompt": '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

'''
}

def get_templates(model_path, func):
    if 'Llama-2' in model_path:
        if func == 'GCG':
            return LLAMA2_PROMPT_GCG
        elif func == 'chat':
            return LLAMA2_PROMPT
        else:
            raise ValueError(f'Unknown function {func}, should be one of "GCG", "chat"')
    elif 'mpt' in model_path:
        if func == 'GCG':
            return MPT_7B_PROMPT_GCG
        elif func == 'chat':
            return MPT_7B_PROMPT
        else:
            raise ValueError(f'Unknown function {func}, should be one of "GCG", "chat"')
    elif 'gemma' in model_path:
        if func == 'GCG':
            return GEMMA_7B_PROMPT_GCG
        elif func == 'chat':
            return GEMMA_7B_PROMPT
    elif 'Qwen' in model_path:
        if func == 'GCG':
            return QWEN_7B_PROMPT_GCG
        elif func == 'chat':
            return QWEN_7B_PROMPT
    elif 'tulu' in model_path:
        if func == 'GCG':
            return TULU_7B_PROMPT_GCG
        elif func == 'chat':
            return TULU_7B_PROMPT
    elif 'mistral' in model_path:
        if func == 'GCG':
            return MISTRAL_7B_PROMPT_GCG
        elif func == 'chat':
            return MISTRAL_7B_PROMPT
    elif 'vicuna' in model_path:
        if func == 'GCG':
            return VICUNA_7B_PROMPT_GCG
        elif func == 'chat':
            return VICUNA_7B_PROMPT
    elif 'Llama-3' in model_path:
        if func == 'chat':
            return LLAMA3_8B_PROMPT
        elif func == 'GCG':
            return LLAMA3_8B_PROMPT_GCG
    else:
        raise ValueError(f'Unknown model {model_path})')
    
def get_eos(model_path):
    if 'Llama-2' in model_path or 'tulu' in model_path or 'mistral' in model_path or 'vicuna' in model_path:
        return '</s>'
    elif 'mpt' in model_path or 'gpt' in model_path or 'Qwen' in model_path or 'falcon' in model_path:
        return '<|endoftext|>'
    elif 'gemma' in model_path:
        return '<eos>'
    elif 'claude' in model_path:
      return '<EOT>'
    elif 'Llama-3' in model_path:
        return '<|end_of_text|>'
    else:
        raise ValueError(f'Unknown model {model_path}, plz set the eos token manually')
    
def get_end_tokens(model_path):
    if 'Llama-2' in model_path:
        return ' [/INST] '
    elif 'mpt' in model_path or 'Qwen' in model_path:
        return '<|im_end|>\n<|im_start|>assistant\n'
    elif 'gemma' in model_path:
        return '<end_of_turn>\n<start_of_turn>model\n'
    elif 'tulu' in model_path:
        return '\n<|assistant|>\n'
    elif 'mistral' in model_path:
        return ' [/INST]'
    elif 'vicuna' in model_path:
        return ' ASSISTANT:'
    elif 'Llama-3' in model_path:
        return '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    else:
        raise ValueError(f'Unknown model {model_path}, plz set the end token manually')
    
if __name__ == '__main__':
    print(get_templates('meta-llama/Llama-2-7b-chat-hf', 'chat'))