import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams
from BOOST.utils.constants import *
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import google.generativeai as genai
from transformers import AutoTokenizer

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16,
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None,
                 reasoning=False
                 ):
        super().__init__()

        self.reasoning = reasoning
        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path

        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        elif system_message is None and 'oss' in model_path:
            self.system_message = 'Reasoning: Medium'
        else:
            self.system_message = system_message
            

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.bfloat16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        model = model.to(torch.bfloat16)

        return model, tokenizer

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        messages = []
        if hasattr(self, "system_message") and self.system_message is not None:
            messages.append({"role": "system", "content": self.system_message})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Build prompt text (not tokenized yet)
        prompt_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # set True if you want tokenized ids
            add_generation_prompt=True,  # signals model to start generating
            #enable_thinking = self.reasoning
        )
        
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16, dtype=torch.float16):
        prompt_inputs = []
        for prompt in prompts:
            messages = []

            # Add system message if you used set_system_message(conv_temp)
            if hasattr(self, "system_message") and self.system_message is not None:
                messages.append({"role": "system", "content": self.system_message})

            # User input (was conv_temp.append_message(conv_temp.roles[0], prompt))
            messages.append({"role": "user", "content": prompt})

            # Assistant placeholder (was conv_temp.append_message(conv_temp.roles[1], None))
            # You don’t need to append None manually—just add generation prompt.
            prompt_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,             # keep as text string
                add_generation_prompt=True,  # adds model-specific “assistant:” or similar tag
                #enable_thinking = self.reasoning
            )

            prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                input_ids = torch.tensor(input_ids[i:i+batch_size], device="cuda"),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


class ClaudeLLM(LLM):
    def __init__(self,
                 model_path='claude-instant-1.2',
                 api_key=None
                ):
        super().__init__()
        
        if len(api_key) != 108:
            raise ValueError('invalid Claude API key')
        
        self.model_path = model_path
        self.api_key = api_key
        self.anthropic = Anthropic(
            api_key=self.api_key
        )

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
        
        for _ in range(max_trials):
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_path,
                    max_tokens_to_sample=max_tokens,
                    prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                )
                return [completion.completion]
            except Exception as e:
                logging.warning(
                    f"Claude API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        results = []
        for prompt in prompts:
            results.extend(self.generate(prompt, max_tokens, max_trials, failure_sleep_time))
        return results

class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.98,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None and 'gemma' not in self.model_path:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            messages = []

            # Add system message if you used set_system_message(conv_temp)
            if hasattr(self, "system_message") and self.system_message is not None:
                messages.append({"role": "system", "content": self.system_message})

            # User input (was conv_temp.append_message(conv_temp.roles[0], prompt))
            messages.append({"role": "user", "content": prompt})

            # Assistant placeholder (was conv_temp.append_message(conv_temp.roles[1], None))
            # You don’t need to append None manually—just add generation prompt.
            prompt_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,             # keep as text string
                add_generation_prompt=True,  # adds model-specific “assistant:” or similar tag
                #enable_thinking = self.reasoning
            )

            prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
            
    def predict(self, sequences, question):
        black_list = get_black_list()
        # check if any black list in the sequences
        for black in black_list:
            if black in sequences[0]:
                return [0]
        # verify whether the generation is related with the target
        split = question.split()
        cnt = 0
        for word in split:
            if word in sequences[0]:
                cnt += 1
        if cnt/len(split) < 0.5:
            return [0]
        prediction = self.generate(sequences[0])
        for i in range(len(prediction)):
            try:
                prediction[i] = int(prediction[i])
            except:
                prediction[i] = 0
        # print the sequences if the prediction is harmful
        if prediction[0] == 1:
            print(sequences[0])
        return prediction

class GeminiLLM(LLM):
    def __init__(self,
                 model_path='gemini-pro',
                 api_key=None
                ):
        super().__init__()
        
        if len(api_key) != 39:
            raise ValueError('invalid Gemini API key')
        
        self.model_path = model_path
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel(self.model_path)

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
     
   
        for _ in range(max_trials):
            try:
                completion = self.gemini.generate_content(f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}", max_tokens=max_tokens)
                return [completion.text]
            except Exception as e:
                logging.warning(
                    f"Gemini API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, max_tokens,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results