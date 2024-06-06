# Databricks notebook source
# MAGIC %md
# MAGIC We use a secret to authenticate against Hugging Face (see [Documentation](https://docs.databricks.com/en/security/secrets/secrets.html)).<br/>
# MAGIC - We can use the cluster terminal and install: <br/>
# MAGIC `pip install databricks-cli` <br/>
# MAGIC - Configure the CLI. We'll need our workspace URL and a PAT token from our profile page.<br>
# MAGIC `databricks configure`
# MAGIC - Create the rlaif scope:<br/>
# MAGIC `databricks secrets create-scope --scope rlaif`
# MAGIC - Save your Hugging Face secret.<br>
# MAGIC `databricks secrets put --scope rlaif --key hf_token`

# COMMAND ----------

# DBTITLE 1,Authenticate against Hugging Face
import os
from huggingface_hub import login

os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")
login(os.environ["HF_TOKEN"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

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
    "max_new_tokens": 150,
    "eos_token_id": terminators,
    "pad_token_id": tokenizer.eos_token_id,
}

# COMMAND ----------

text = "What are some protein sources that can be used in dishes?"

def prompt_generate(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

query = tokenizer.encode(prompt_generate(text), return_tensors="pt").to(device)
outputs = model.generate(query, **generation_kwargs)
response = outputs[0][query.shape[-1]+1:]
print(tokenizer.decode(response, skip_special_tokens=True))

# COMMAND ----------


