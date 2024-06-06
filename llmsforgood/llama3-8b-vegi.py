# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
import re
import math
import pandas as pd

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
from peft import LoraConfig

from conf import REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes Meta-Llama-3-8B-Instruct to generate more vegetarian contents
# by using prompts dataset. We use PPO (proximal policy optimization) 
# to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'tensorboard' to log with tensorboard"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="/dbfs/rlaif/llm/",
        metadata={"help": "the path to save the model"},
    )
    dataset_path: Optional[str] = field(
        default="/dbfs/rlaif/data/",
        metadata={"help": "the path to the training dataset"}
    )
    sample_size: Optional[int] = field(default=None, metadata={"help": "the number of training samples. Set to None to use all."})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of epochs for training"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Targeting all linear layers
target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

lora_config = LoraConfig(
    r=8,
    target_modules=target_modules,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    project_kwargs={"logging_dir": "/databricks/driver/logdir/"},
    ppo_epochs=script_args.ppo_epochs,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

def prompt_generate(text):
    return f"""<|start_header_id|>system<|end_header_id|>
    You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def prompt_score(text):
    return f"""Text: {text}"""

system_prompt = """You are an AI assistant that specializes in vegetarian cuisine. Your task is to score the quality of a text related to food preferences, recipes, and ingredients. Generate 1 score on a scale from 0.01 to 0.99, which indicates how good the text provided in below is. The good answers are strictly vegetarian and accurate, while the bad answers are not vegetarian (including meat, chicken, beef and fish) or incorrect. 

Below is an example of a good text with score 0.99 and a bad text with score 0.01.

- Good text with score 0.99: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."

- Bad text with score 0.01: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix. Fish is also a great alternative."

Give the score at the beginning. Give only the score. Use no more than 10 words."""

# Below is an example function to build the dataset. We use the dataset stored in dataset_path.
# One should customize this function to train the model on its own dataset.
def build_dataset(config, dataset_path):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_path (`str`):
            The path to the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_path, split='train')
    ds = ds.shuffle(seed=42)
    
    if script_args.sample_size:
        ds = ds.select(range(script_args.sample_size))
    
    def tokenize(sample):
        prompt = prompt_generate(sample["prompt"])
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config, dataset_path=script_args.dataset_path)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer. We first load the model
# in bfloat16 to save memory using `transformers`.
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=True,
    )
# And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model,
    peft_config=lora_config
    )

# We can create a reference model by specifying the number of sharing layers
# However, since we use LoRA in this demo, we don't need the reference model.
#ref_model = create_reference_model(model, num_shared_layers=20)

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None, # Set the reference model to None as we are using LoRA
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

import json
import aiohttp
import asyncio
import re

async def llama(url, token, text, session):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    body = {"messages": [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": prompt_score(text)
        }
        ], "params": {"max_tokens": 64}}
    data = json.dumps(body)
    async with session.post(url, data=data, headers=headers) as response:
        return await response.json()
          
async def run_concurrent_requests(url, token, texts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index in range(len(texts)):
            response = llama(url, token, texts[index], session=session)
            tasks.append(response)
        return await asyncio.gather(*tasks, return_exceptions=True)

# We define the arguments to pass to the `generate` function. These arguments are 
# passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "max_new_tokens": 150,
    "eos_token_id": terminators,
    "pad_token_id": tokenizer.eos_token_id,
}

# Base model sometimes generates no response, which causes an error.
# We overwrite these empty texts with a place holder text.
place_holder_text = "The base model failed to generate a meaningful text."
model_save_path = script_args.model_save_path

for epoch in range(config.ppo_epochs):
    if ppo_trainer.accelerator.is_main_process:
        print(f"Epoch: {epoch}")
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        # Get the response from the base model
        response_tensors = []
        for query in query_tensors:                    
            response = ppo_trainer.generate(query, return_prompt=False, **generation_kwargs).squeeze()
            if not response.shape:
                response = torch.tensor(tokenizer.encode(place_holder_text)).squeeze()
            response_tensors.append(response)                
        # Get the score from the reward model
        batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        scores = []
        try:
            responses = asyncio.run(run_concurrent_requests(REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN, batch["response"]))
            responses = [responses[i]["choices"][0]["message"]["content"] for i in range(len(responses))]
            for response in responses:
                match = re.search(r'\d+\.\d+', response)
                scores.append(float(match.group()))
            rewards_tensors = [torch.tensor(math.log(score/(1.0-score))) for score in scores]
        except:
            scores = [0.5] * config.batch_size
            rewards_tensors = [torch.tensor(0.)] * config.batch_size
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensors)
        ppo_trainer.log_stats(stats, batch, rewards_tensors)
        
        # Save model every 20 step and write out results every 40 step 
        # One step trains on <batch_size> x <num_processes> prompts
        if step % 20 == 0:
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(model_save_path)
            if step % 40 == 0:
                if ppo_trainer.accelerator.is_main_process:
                    print(f"STEP: {step}")
                    print(f"PROMPTS: {batch['query']}")
                    print(f"GENERATED: {batch['response']}")
                    print(f"SCORED: {responses}")    
        
