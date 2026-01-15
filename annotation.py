import os
import re
import torch
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import transformers
from datasets import load_dataset
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple


from huggingface_hub import login

from dotenv import load_dotenv
load_dotenv()


VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

model_names = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "berkeley-nest/Starling-LM-7B-alpha",
    # "allenai/OLMo-7B-0724-Instruct-hf",
    # "gemini-1.5-flash-latest",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    # "zai-org/GLM-4.6"
]
prompt_instructions = "You are an expert scientific annotator specialized in climate science. Your task is to determine whether a scientific abstract supports, refutes, or provides insufficient information about a given claim.\n"
prompt_content = """
Claim: {claim}

Abstract: {abstract}

Based on the abstract above, does it SUPPORT, REFUTE, or provide NOT_ENOUGH_INFO about the claim?

Instructions:
- SUPPORTS: The abstract provides evidence that confirms or validates the claim
- REFUTES: The abstract provides evidence that contradicts or disproves the claim  
- NOT_ENOUGH_INFO: The abstract does not contain sufficient information to support or refute the claim

Provide your answer in the following format:
Label: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]

Answer:"""

def parse_response(response: str):
        response_upper = response.upper()
        
        label = None
        for valid_label in VALID_LABELS:
            if valid_label in response_upper:
                label = valid_label
                break
        
        if not label:
            if "SUPPORT" in response_upper and "NOT" not in response_upper.split("SUPPORT")[0][-20:]:
                label = "SUPPORTS"
            elif "REFUTE" in response_upper:
                label = "REFUTES"
            elif "NOT ENOUGH" in response_upper or "INSUFFICIENT" in response_upper:
                label = "NOT_ENOUGH_INFO"
            else:
                label = "UNKNOWN"
    
        return label

class HFModel:
    def __init__(self, engine, batch_size=8, new_tokens=1000) -> None:
        self.model_id = engine
        # login(token=os.environ.get('HF_TOKEN')) 

        print(f"Loading model from {self.model_id} with HF")
        # self.model = transformers.pipeline(
        #     "text-generation",
        #     model=self.model_id,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     trust_remote_code=True if "OLMo" in self.model_id else False,
        # )
        
        # self.n_tokens = new_tokens
        self.batch_size = batch_size
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

    def truncate_abstract(self, abstract: str, max_tokens: int = 3000) -> str:
        tokens = self.tokenizer.encode(abstract)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True) + "..."
        return abstract

    def process(self, example):
        user_message = {"role": "user", "content": example}
        messages = [user_message]
        return self.model.tokenizer.apply_chat_template(
            messages, tokenize=False
        )

    def generate_responses(self, dataset, batch_size):
        dataset = [self.process(example) for example in dataset]
        results = []
        for n_batch in tqdm(range(len(dataset) // batch_size + 1)):
            batch = dataset[batch_size * n_batch : batch_size * (n_batch + 1)]
            if len(batch) == 0:
                continue
            responses = self.model(
                batch,
                batch_size=batch_size,
                max_new_tokens=self.n_tokens,
                do_sample=False,
                num_beams=1,
                return_full_text=False,
            )
            for response in responses:
                results.append(response[0]["generated_text"])
        return results
    
    def process_batch(self, data):
        results = []
        for i in tqdm(range(0, len(data), self.batch_size), desc="Processing batches"):
            batch = data[i:i + self.batch_size]
            
            batch_messages = [
                [{"role": "user", 
                "content": prompt_instructions + prompt_content.format(
                    claim=data['claim'], 
                    abstract=data['abstract']
                )}]
                for data in batch
            ]
            
            inputs = self.tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to('cuda')
            
            with torch.no_grad():
                responses = self.model.generate(**inputs, max_new_tokens=40)
            
            for j, data in enumerate(batch):
                response = self.tokenizer.decode(
                    responses[j][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                label = parse_response(response)
                
                result = {
                    'pair_id_inception': data['pair_id_inception'],
                    'claim_id': data['claimID_original'],
                    'claim': data['claim'],
                    'abstract_id': data['abstract_original_index'],
                    'abstract': data['abstract'],
                    'annotation': data['annotation'],
                    'predicted_label': label,
                    'raw_response': response
                }
                results.append(result)
        return results
    
    def annotate_dataset(self, data_rows: List[dict]) -> List[dict]:
        all_results = []
        
        for data in tqdm(data_rows, desc="Processing batches"):
            messages = [
                {"role": "user", 
                "content": prompt_instructions+prompt_content.format(claim=data['claim'], abstract=data['abstract'])
                }]
                
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            responses = self.model.generate(**inputs, max_new_tokens=40)
            response = self.tokenizer.decode(responses[0][inputs["input_ids"].shape[-1]:])
            label = parse_response(response)

            result = {
                'pair_id_inception': data['pair_id_inception'],
                'claim_id': data['claimID_original'],
                'claim': data['claim'],
                'abstract_id': data['abstract_original_index'],
                'abstract': data['abstract'],
                'annotation': data['annotation'],
                'predicted_label': label,
                'raw_response': response
            }
            all_results.append(result)
        return all_results

class VLLM:
    def __init__(self, engine="meta-llama/Llama-3.1-8B-Instruct", batch_size=8, max_tokens=256):
        """
        Initializes an instance of the class and its related components.

        Attributes
            model: LLM model.
            user_vllm (bool): Variable to set if model is loaded via VLLM. Alternatively, 
                model is loaded via transformers pipeline.

        Parameters
            exp (str): Name of the current experiment (e.g. 0_shot, no_rag, no_relation)
            engine (str): Name of the LLM model to be used for processing.
            n_shot (int): Number of few-shot examples.
            use_vllm (bool): Variable to set if model is loaded via VLLM. Alternatively, 
                model is loaded via transformers pipeline.
        """
        self.model = None
        self.model_id = engine
        self.batch_size = batch_size
        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = LLM(
            model=self.model_id,
            tensor_parallel_size=1,  # Adjust based on GPU availability
            gpu_memory_utilization=0.9,
            max_model_len=4096,  # Adjust based on your needs
            trust_remote_code=True
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for consistency
            top_p=1.0,
            max_tokens=self.max_tokens,
            stop=["\n\n", "Claim:", "Abstract:"]  # Stop sequences
        )
        print(f"Loaded model from {self.model_id} with VLLM")
    
    def truncate_abstract(self, abstract: str, max_tokens: int = 3000) -> str:
        tokens = self.tokenizer.encode(abstract)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True) + "..."
        return abstract

    def process_batch(self, batch_data) -> List[dict]:
        results = []
        prompts = []
        for data in batch_data:
            if len(data['abstract']) > 3000:
                abstract = self.truncate_abstract(data['abstract'], max_tokens=3000)
            else: abstract = data['abstract']
            prompts.append(prompt_instructions+prompt_content.format(
                claim=data['claim'], 
                abstract=abstract
            ))
        
        outputs = self.model.generate(prompts, self.sampling_params)
        
        for data, output in zip(batch_data, outputs):
            response = output.outputs[0].text.strip()
            label = parse_response(response)
            
            result = {
                'pair_id_inception': data['pair_id_inception'],
                'claim_id': data['claimID_original'],
                'claim': data['claim'],
                'abstract_id': data['abstract_original_index'],
                'abstract': data['abstract'],
                'annotation': data['annotation'],
                # 'total_votes': data['total_votes'],
                'predicted_label': label,
                'raw_response': response
            }
            results.append(result)
        
        return results

    def annotate_dataset(self, data_rows: List[dict]) -> List[dict]:
        all_results = []
        
        for i in tqdm(range(0, len(data_rows), self.batch_size), desc="Processing batches"):
            batch = data_rows[i:i + self.batch_size]
            results = self.process_batch(batch)
            all_results.extend(results)
        return all_results


def main():
    # claims_test = load_dataset("rabuahmad/climatecheck", split='test')
    # claims_train = load_dataset("rabuahmad/climatecheck", split='train')
    # claims_df = pd.DataFrame(claims_train)
    # pubs = load_dataset("rabuahmad/climatecheck_publications_corpus")

    with open('data/climatecheck_data_annotated_2025-08-18.pkl', 'rb') as file:
        claims_df = pickle.load(file)

    batch_size = 8
    load_type = "hf" # api, vllm
    model = model_names[2]
    model_name = model.split("/")[1]
    output_path = f"outputs/{model_name}.json"

    print("===== INFO ======")
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    print("Model: ", model)

    data_rows = claims_df.to_dict('records')

    if load_type == "api":
        results = []
        client = OpenAI(
            api_key = os.environ['CHAT_AI_KEY'],
            base_url = os.environ['BASE_URL']
        )
        for data in tqdm(data_rows):
            result = {
                    'pair_id_inception': data['pair_id_inception'],
                    'claim_id': data['claimID_original'],
                    'claim': data['claim'],
                    'abstract_id': data['abstract_original_index'],
                    'abstract': data['abstract'],
                    'annotation': data['annotation']
                }
            try:
                output = client.chat.completions.create(
                    messages=[{"role":"system","content":prompt_instructions},
                            {"role":"user","content":prompt_content.format(
                                claim=data['claim'], 
                                abstract=data['abstract']
                            )}],
                    model= "qwen2.5-vl-72b-instruct"# "meta-llama-3.1-70b-instruct"
                )
                result['predicted_label'] = parse_response(output.choices[0].message.content)
                result['raw_response'] = output.choices[0].message.content
                results.append(result)
            except:
                result['predicted_label'] = None
                result['raw_response'] = None
                results.append(result)
                continue
    elif load_type == "vllm":
        annotator = VLLM(engine=model, batch_size=batch_size)
        results = annotator.annotate_dataset(data_rows)
    else:
        annotator = HFModel(engine=model, batch_size=batch_size)
        results = annotator.process_batch(data_rows)

    with open(output_path, 'w+') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")

def transformer_run():
    with open('data/climatecheck_data_annotated_2025-08-18.pkl', 'rb') as file:
        claims_df = pickle.load(file)

    model = model_names[2]
    model_name = model.split("/")[1]
    output_path = f"outputs/{model_name}.json"

    print("===== INFO ======")
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    print("Model: ", model)

    data_rows = claims_df.to_dict('records')
    with open(f"outputs/{model_name}.json") as f:
        written_data_1 = json.load(f)
    data_rows = data_rows[len(written_data_1):]

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model).to('cuda')
    # model = DataParallel(model)
    # model = model.cuda()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = 8 
    results = []
    for i in tqdm(range(0, len(data_rows), batch_size), desc="Processing batches"):
        batch = data_rows[i:i + batch_size]
        with open(f"outputs/{model_name}.json") as f:
            written_data = json.load(f)

        batch_messages = [
            [{"role": "user", 
            "content": prompt_instructions + prompt_content.format(
                claim=data['claim'], 
                abstract=data['abstract']
            )}]
            for data in batch
        ]
        
        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to('cuda')
        
        with torch.no_grad():
            responses = model.generate(**inputs, max_new_tokens=40)
        
        for j, data in enumerate(batch):
            response = tokenizer.decode(
                responses[j][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            label = parse_response(response)
            
            result = {
                'pair_id_inception': data['pair_id_inception'],
                'claim_id': data['claimID_original'],
                'claim': data['claim'],
                'abstract_id': data['abstract_original_index'],
                'abstract': data['abstract'],
                'annotation': data['annotation'],
                'predicted_label': label,
                'raw_response': response
            }
            results.append(result)
            written_data.append(result)
            with open(output_path, 'w+') as f:
                json.dump(written_data, f, indent=2, ensure_ascii=False)
   
    with open(output_path, 'w+') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    transformer_run()
    # main()
