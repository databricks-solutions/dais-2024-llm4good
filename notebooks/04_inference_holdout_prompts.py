# Databricks notebook source
# MAGIC %md The objective of this notebook is to run inference on the "holdout prompts" which were created during `01_generate_prompts`. These prompts were not used to train the fine-tuned model and will be used during evaluation process to understand how well the model has generalized to unseen data. 
# MAGIC 1. The first step is to run inference on the prompts using the pre fine-tuned model (meta-llama/Meta-Llama-3-8B-Instruct) and save the results. 
# MAGIC 2. The second step is to run inference on the prompts using the post fine-tuned model (vegetarian) and save the results. The results will be used in the evaluation step to compare pre vs. post model improvements. 
# MAGIC
# MAGIC DBR Run-time used to run this notebook is DBR 14.2ML with A10 GPU

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Run inference on holdouts prompts using the pre fine-tuned model (meta-llama/Meta-Llama-3-8B-Instruct)

# COMMAND ----------

# MAGIC %md Create Hugging Face account if you haven't done so. Once you create account, generate a token and save it somewhere safe. This is needed to login to Hugging Face hub to download models. Once you have created account and generated a token, create a scope and secret using Databricks CLI so you can safely access token. 
# MAGIC
# MAGIC `databricks secrets create-scope rlaif`
# MAGIC `databricks secrets put-secret rlaif hf_token`

# COMMAND ----------

from datetime import date

today = date.today().strftime("%Y%m%d")
model_name = "llama3-8b-vegetarian"

dbutils.widgets.text("base_model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
dbutils.widgets.text("finetuned_model_name", f"{model_name}-{today}")

base_model_name = dbutils.widgets.get("base_model_name")
finetuned_model_name = dbutils.widgets.get("finetuned_model_name")

# COMMAND ----------

import os
from huggingface_hub import login

os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")
login(os.environ["HF_TOKEN"])

# COMMAND ----------

# MAGIC %md Import transformers package and load meta-llama/Meta-Llama-3-8B-Instruct pre-trained model and tokenizer. This includes the model configurations in `generation_kwargs` that will be used to run inference, you can find more details on these configurations here.

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# model_name = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=True,
    ).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "eos_token_id": terminators,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 250
}

# COMMAND ----------

# MAGIC %md Load the holdout prompts data and convert to Pandas

# COMMAND ----------

import pandas as pd
questions = spark.table("rlaif.data.prompts_holdout").toPandas()
display(questions)

# COMMAND ----------

# MAGIC %md Iterate through each prompt and pass the prompt question to `get_prompt` function, which inserts the question into the instruction format. This prompt + question will be passed to the tokenizer, the encoded tokenized prompt is passed to the model to generate the response and finally decode the outputs back into human-readable text and saved to the answers list.

# COMMAND ----------

system_prompt = """You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words."""

def get_prompt(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

     
import re
      
def extract_response(text, n=2):
    """
    Extracts everything after the second occurrence of "assistant" in the given text.

    Args:
    - text (str): The input text containing multiple occurrences of "assistant".

    Returns:
    - str: The extracted text after the second occurrence of "assistant", or None if it's not found.
    """
    # Find the position of the second occurrence of "assistant"
    matches = re.finditer(r'assistant', text)
    positions = [match.start() for match in matches]
    if len(positions) >= n:
        second_assistant_position = positions[1]
        # Extract text after the second occurrence of "assistant"
        return text[second_assistant_position:].strip()
    else:
        return None

answers = []
raw_answers = []

for question in list(questions['prompt'].values):
    prompt = get_prompt(question)
    query = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(query, **generation_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_extract = extract_response(response).replace("assistant\n\n", "")
    answers.append(response_extract)
    raw_answers.append(response)

# COMMAND ----------

# MAGIC %md Create pandas dataframe, join the questions and answers and create Spark dataframe so we can write the results to our table.

# COMMAND ----------

from pyspark.sql import functions as F

answers = pd.DataFrame(answers).rename(columns={0:"pre_finetuning"})
df = pd.merge(questions, answers, left_index=True, right_index=True)
df = spark.createDataFrame(df)
       
display(df)

# COMMAND ----------

df.write.mode("overwrite").saveAsTable(f"rlaif.data.pre_finetuning")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Run inference on holdouts prompts using the post fine-tuned model we trained earlier

# COMMAND ----------

# MAGIC %md Clear cached memory for both Python and CUDA backend and restart the Python interpreter since we will need a lot of memory to run our fine-tuned model.

# COMMAND ----------

import torch
import gc
gc.collect()
torch.cuda.empty_cache()
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Dowlonad the requirements needed to run inference

# COMMAND ----------

# MAGIC %sh /databricks/python/bin/python -m pip install -r ../requirements.txt --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Login to Hugging Face Hub to download the base model (Llama-2-7b-chat-hf)

# COMMAND ----------

import os
from huggingface_hub import login

os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")
login(os.environ["HF_TOKEN"])

# COMMAND ----------

# MAGIC %md 
# MAGIC Load the pre-trained base model (Llama-2-7b-chat-hf) reducing memory usage by utilizing bfloat16 data type and move to the device.
# MAGIC
# MAGIC Load the fine-tuned PEFT-processed model from the output path we saved it to during the model training process.
# MAGIC
# MAGIC Merge the fine-tuned parameters with the original model parameters, creating a single set of parameters that represent the fine-tuned model. Then unload the original pre-trained model from memory to free up valuable GPU resources.
# MAGIC

# COMMAND ----------

from datetime import date

today = date.today().strftime("%Y%m%d")
model_name = "llama3-8b-vegetarian"

dbutils.widgets.text("base_model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
dbutils.widgets.text("finetuned_model_name", f"{model_name}-{today}")

base_model_name = dbutils.widgets.get("base_model_name")
finetuned_model_name = dbutils.widgets.get("finetuned_model_name")

# COMMAND ----------

import torch
import peft
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from datetime import date

finetuned_model_path = f"/dbfs/rlaif/llm/{finetuned_model_name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)
model = PeftModel.from_pretrained(model, finetuned_model_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, padding_side="left")

# COMMAND ----------

# MAGIC %md Model configurations used in the model for inference. 

# COMMAND ----------

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "eos_token_id": terminators,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 250
}

# COMMAND ----------

# MAGIC %md Load in holdout prompts and iterate through each prompt and pass the prompt question to `get_prompt` function, which inserts the question into the instruction format. This prompt + question will be passed to the tokenizer, the encoded tokenized prompt is passed to the fine-tuned model to generate the response and finally decode the outputs back into human-readable text and saved to the answers list.

# COMMAND ----------

import pandas as pd
questions = spark.table("rlaif.data.prompts_holdout").toPandas()

# COMMAND ----------

system_prompt = """You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words."""

def get_prompt(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

     
import re
      
def extract_response(text, n=2):
    """
    Extracts everything after the second occurrence of "assistant" in the given text.

    Args:
    - text (str): The input text containing multiple occurrences of "assistant".

    Returns:
    - str: The extracted text after the second occurrence of "assistant", or None if it's not found.
    """
    # Find the position of the second occurrence of "assistant"
    matches = re.finditer(r'assistant', text)
    positions = [match.start() for match in matches]
    if len(positions) >= n:
        second_assistant_position = positions[1]
        # Extract text after the second occurrence of "assistant"
        return text[second_assistant_position:].strip()
    else:
        return None

answers = []
raw_answers = []

for question in list(questions['prompt'].values):
    prompt = get_prompt(question)
    query = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(query, **generation_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_extract = extract_response(response).replace("assistant\n\n", "")
    answers.append(response_extract)
    raw_answers.append(response)

# COMMAND ----------

# MAGIC %md Create pandas dataframe, join the questions and answers and create Spark dataframe so we can write the results to our table.

# COMMAND ----------

from pyspark.sql import functions as F

answers = pd.DataFrame(answers).rename(columns={0:"post_finetuning"})
df = pd.merge(questions, answers, left_index=True, right_index=True)
df = spark.createDataFrame(df)
       
display(df)

# COMMAND ----------

df.write.mode("overwrite").saveAsTable("rlaif.data.post_finetuning")

# COMMAND ----------

# MAGIC %md ##Consolidate

# COMMAND ----------

pre_finetune = spark.table("rlaif.data.pre_finetuning").toPandas()
post_finetune = spark.table("rlaif.data.post_finetuning").toPandas()
df = pd.merge(pre_finetune, post_finetune, on=["prompt"])
display(df)
