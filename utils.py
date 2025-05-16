import os
import json
import time
import openai
import anthropic
from openai import OpenAI
#from vllm import LLM, SamplingParams
from typing import Union, List
from dotenv import load_dotenv
load_dotenv()


CALL_SLEEP = 1
clients = {}

def initialize_clients():
    try:
        #gpt client
        client = OpenAI(api_key="")
        openai.organization = ""
        clients['gpt'] = client

        #claude client
        clients['claude'] = anthropic.Anthropic(
            api_key="")

        if not clients:
            print("No valid API credentials found. Exiting.")
            exit(1)

    except Exception as e:
        print(f"Error during client initialization: {e}")
        exit(1)

def initialize_open_source_llms():
    mistral_model = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",  # Mistral Instruct model
    trust_remote_code=True,     
    gpu_memory_utilization=0.23,
    enforce_eager=False,
    swap_space=8,
    max_num_seqs=64,                   # pre-allocate more sequences (batch handling)
    dtype='bfloat16',                   # faster precision for A100 GPUs
    disable_custom_all_reduce=False,    # efficient multi-GPU synchronization
    max_model_len=8192,
    ) #
    
    llama_model = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # Llama-3 Instruct model
    trust_remote_code=True,
    gpu_memory_utilization=0.28,
    enforce_eager=False,
    swap_space=8,
    max_num_seqs=64,                   # pre-allocate more sequences (batch handling)
    dtype='bfloat16',                   # faster precision for A100 GPUs
    disable_custom_all_reduce=False,    # efficient multi-GPU synchronization  
    )

    return llama_model, mistral_model

initialize_clients()
#llama_model, mistral_model = initialize_open_source_llms()

def get_client(model_name):
    """Select appropriate client based on the given model name."""
    if 'gpt' in model_name or 'o1-'in model_name:
        client = clients.get('gpt') 
    elif 'claude' in model_name:
        client = clients.get('claude')
    elif 'deepseek' in model_name:
        client = clients.get('deepseek')
    elif any(keyword in model_name.lower() for keyword in ['llama', 'qwen', 'mistral', 'microsoft']):
        client = clients.get('deepinfra')
    else:
        raise ValueError(f"Unsupported or unknown model name: {model_name}")

    if not client:
        raise ValueError(f"{model_name} client is not available.")
    return client 

def read_prompt_from_file(filename):
    with open(filename, 'r') as file:
        prompt = file.read()
    return prompt

def read_data_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def parse_json(output):
    try:
        output = ''.join(output.splitlines())
        if '{' in output and '}' in output:
            start = output.index('{')
            end = output.rindex('}')
            output = output[start:end + 1]
        data = json.loads(output)
        return data
    except Exception as e:
        print("parse_json:", e)
        return None
    
def check_file(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        raise IOError(f"File not found error: {file_path}.")


def open_source_llm_call(query: Union[List, str], model_name: str):
    
    #sampling_params = SamplingParams(max_tokens=600, temperature=1.3, top_p=0.95, top_k=50, repetition_penalty=1.05) # hyper-parameter optimization
    sampling_params = SamplingParams(max_tokens=600, temperature=1)
    
    llm_model = llama_model if 'llama' in model_name else mistral_model

    for _ in range(3):
        try:
            completion = llm_model.generate(query, sampling_params)
            if isinstance(query, str):
                return completion[0].outputs[0].text
            elif isinstance(query, list):
                return [response.outputs[0].text for response in completion]   
        except Exception as e:
            print(f"Open-source llm call error: {e}")
            time.sleep(CALL_SLEEP)
    return ""

def open_source_llm_append(model_name, dialog_hist: List, query: str):
    dialog_hist.append({"role": "user", "content": query})
    resp = open_source_llm_call(query, model_name=model_name)
    dialog_hist.append({"role": "assistant", "content": resp})
    return resp, dialog_hist
    
        
def gpt_call(client, query: Union[List, str], model_name = "gpt-4o", temperature = 0):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "user", "content": query}]
        
    for _ in range(3):
        try:
            if 'claude' in model_name:
                print('11111111111111111111')
                completion = client.messages.create(
                model=model_name,
                max_tokens=800,
                temperature=temperature,
                messages=messages)
                print('2222222222')
                return completion.content[0].text.strip()
            
            elif 'o1-' in model_name:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature
                )
            resp = completion.choices[0].message.content
            return resp
        except Exception as e:
            print(f"GPT_CALL Error: {model_name}:{e}")
            time.sleep(CALL_SLEEP)
            continue
    return ""

def gpt_call_append(client, model_name, dialog_hist: List, query: str):
    dialog_hist.append({"role": "user", "content": query})
    resp = gpt_call(client, dialog_hist, model_name=model_name)
    dialog_hist.append({"role": "assistant", "content": resp})
    return resp, dialog_hist