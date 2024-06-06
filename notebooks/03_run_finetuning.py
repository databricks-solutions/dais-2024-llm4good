# Databricks notebook source
# MAGIC %md 
# MAGIC # Fine-tuning using Reinforcement learning from AI feedback (RLAIF)
# MAGIC
# MAGIC In this notebook, we use the prompts generated in `01_generate_prompts`, and the reward model deployed and evaluated in `02_evaluate_reward_model` to fine-tune [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) using [RLAIF](https://arxiv.org/abs/2309.00267). We choose a smaller target model for this demo reflecting the [trend]((https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) we see in the market, where companies are resorting to smaller, customized models to reduce cost and latency while achieving a similar or even better quality than the bigger alternatives. That said, open-source models of different sizes can be also be fine-tune in a similar way depending on the computing infrastructure you provision.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster requirements
# MAGIC To run the notebook, we recommend using a cluster with multiple GPU instaces: e.g. 8 x A10G (192GB GPU memory) or 4 x A100G (880GB), which is sufficient to load and fine-tune an 8B parameter model. Don't forget to provide the number of GPUs (`num_processes`) in the configuration file: `yamls/accelerate/zero2.yaml`. When fine-tuning a larger model  (e.g. [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)), we recommend using more powerful instances with a larger memory and potentially a multi-node setting. 

# COMMAND ----------

# DBTITLE 1,Install necessary libraries
# MAGIC %sh /databricks/python/bin/python -m pip install -r ../requirements.txt --quiet

# COMMAND ----------

# MAGIC %pip install aiohttp[speedups] --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set variables

# COMMAND ----------

# MAGIC %md
# MAGIC We define variables we need for training below. `model_name` is the name we give to the fine-tuned model. `base_model_name` is the target pre-trained model we are going fine-tune as referenced by Hugging Face. `dataset_path` is where the prompts generated in the previous notebook is stored. Note that it should be stored as a csv file since we will generate a tranformers `datasets` obeject out of this csv file in the training script (see `build_dataset` method in `llmsforgood/llama3-8b-vegi.py`). `output` is where we want to store the fine-tuned adapotors. `tb_output` is a DBFS location where we write out the TensorBoard event files before it gets lost when the cluster terminates. And finally, `logdir` is a intermediate location on the driver node where we write out the tensorboard event files (shortly described). 

# COMMAND ----------

from datetime import date

today = date.today().strftime("%Y%m%d")
model_name = "llama3-8b-vegetarian"
base_model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
dataset_path = "/dbfs/rlaif/data/"
output = f"/dbfs/rlaif/llm/{model_name}-{today}"
tb_output = f"/dbfs/rlaif/tb/{model_name}-{today}"
logdir = "/databricks/driver/logdir/trl"

# COMMAND ----------

# MAGIC %md
# MAGIC Make sure that the folders exists in DBFS. If not we create.

# COMMAND ----------

!mkdir -p {output}

# COMMAND ----------

!mkdir -p {tb_output}

# COMMAND ----------

# DBTITLE 1,Define python variables as environment variables
import os
os.environ['SCRIPT'] = "../llmsforgood/llama3-8b-vegi.py" 
os.environ['OUTPUT'] = output
os.environ['TB_OUTPUT'] = tb_output
os.environ['DATSET_PATH'] = dataset_path
os.environ['LOGDIR'] = logdir

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up TensorBoard
# MAGIC [TensorBoard](https://www.tensorflow.org/tensorboard) is an open source monitoring solution for model training. It reads an event log and exposes the training metrics in near real-time on a dashboard. This helps us gauge the status of fine-tuning without having to wait until the whole training cycle is done.
# MAGIC
# MAGIC When you write the event log directly to DBFS, the metrics won't show until the file is closed for writing, which is when the training is complete. This defeats the purpose of real-time tracking. We suggest writing the event log out to the driver node (`logdir`) and run your TensorBoard from there. Files stored on the driver node may get removed when the cluster terminates or restarts. But after the training we will copy all our Tensorboard artifacts to a DBFS location, so that we can recover them later.

# COMMAND ----------

from tensorboard import notebook
notebook.start("--logdir {} --reload_multifile True".format(logdir))

# COMMAND ----------

# MAGIC %md
# MAGIC The key metrics to pay attention to during the training are: (1) the mean reward, (2) the reward standard deviation, and (3) the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). If the mean reward increases and eventually converges over time, this indicates that the model generates more aligned texts with higher scores. For the same reason, the standard deviation of the mean reward should decrease and converge over time. The KL divergence usually increases rapidly at the beginning of the training, indicating the target model is drifting away from its original weights, but should eventually converge. We will see all these metrics on TensorBoard.

# COMMAND ----------

# See all the tensorboard processes running
from tensorboard import notebook
notebook.list()

# COMMAND ----------

# DBTITLE 1,Login to Hugging Face with your credentials to use Llama models
from huggingface_hub import login

os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")
login(os.environ["HF_TOKEN"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the fine-tuning script

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the [PPO](https://huggingface.co/docs/trl/main/en/ppo_trainer) implementation from [TRL](https://huggingface.co/docs/trl/main/en/index) library - an open-source framework from [Hugging Face](https://huggingface.co/). Additionally, [LoRA](https://arxiv.org/abs/2106.09685) is used to reduce the GPU memory requirement. PPO usually requires two copies of the target model, but when combined with LoRA, only one is needed, which further reduces the memory footprint significantly. TRL is integrated with [Accelerate](https://huggingface.co/docs/accelerate/index), and [DeepSpeed](https://github.com/microsoft/DeepSpeed), which has nice [integration with Databricks](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/deepspeed.html). We use them to achieve parallelism and [optimize resource utilization](https://community.databricks.com/t5/technical-blog/introducing-the-deepspeed-distributor-on-databricks/ba-p/59641). 
# MAGIC
# MAGIC In the previous notebook, we deployed the `Meta-Llama-3-70B-Instruct` model behind a Databricks Model Serving endpoint with an expected throughput of 635 tokens / second. We use this model as our reward function inside the PPO training loop. See our [technical blog post](https://community.databricks.com/t5/technical-blog/model-alignment-at-scale-using-reinforcement-learning-from-ai/ba-p/62877) for more detailed information. Make sure your reward model is up and running when you run the following cell.
# MAGIC
# MAGIC Besides `dataset_path`, `model_save_path` and `log_with`, you can specify other parameters such as `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `ppo_epochs`, and `sample_size`, which the number of prompts you want to use in your training. If not specified it uses all. See `ScriptArguments` class in `llmsforgood/llama3-8b-vegi.py` to see all the parameters and their definitions.
# MAGIC
# MAGIC The following cell will run for about 5 hours for 1 epoch (10000 prompts) on a single node cluster with 4 x A100G (880GB). 

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file ../yamls/accelerate/zero2.yaml $SCRIPT \
# MAGIC     --dataset_path $DATSET_PATH \
# MAGIC     --model_save_path $OUTPUT \
# MAGIC     --log_with tensorboard

# COMMAND ----------

# MAGIC %md
# MAGIC Once the training is complete, we copy the TensorBoard logs over to the DBFS location.

# COMMAND ----------

# MAGIC %sh cp -r $LOGDIR $TB_OUTPUT

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see if the new adaptor weights exists.

# COMMAND ----------

!ls -lah {output}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC We will now take the fine-tuned adaptor, initialize the model and register it in Unity Catalog. We use MLflow to do this. For this, let's first restart the kernel and free up some memory. 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Login to Hugging Face with your credentials to use Llama models
import os
from huggingface_hub import login

os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")
login(os.environ["HF_TOKEN"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap the chatbot as a customer model using `mlflow.pyfunc.PythonModel`

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import peft
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")

class vegetarian(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Authenticate to Hugging Face
        import os
        from huggingface_hub import login
        login(os.environ["HF_TOKEN"])
        
        # Initialize tokenizer and language model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16).to(self.device)
        self.model = peft.PeftModel.from_pretrained(self.model, context.artifacts['repository'])
        self.model = self.model.merge_and_unload()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(context.artifacts['repository'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "eos_token_id": self.terminators,
            "pad_token_id": self.tokenizer.eos_token_id}

    def prompt_generate(self, input_text):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        for index, row in model_input.iterrows():
            input_text = row["input"]
            prompt = self.prompt_generate(input_text)
            query = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(query, **self.generation_kwargs)
            response = outputs[0][query.shape[-1]+1:]
            generated_text.append(self.tokenizer.decode(response, skip_special_tokens=True))

        return pd.Series(generated_text)

# COMMAND ----------

# MAGIC %md
# MAGIC Redefine the variables since we restarted the kernel.

# COMMAND ----------

from datetime import date
today = date.today().strftime("%Y%m%d")
model_name = "llama3-8b-vegetarian"
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
output = f"/dbfs/rlaif/llm/{model_name}-{today}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the custom model to MLflow

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "input")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({"input":["Give me a recipe for healthy smoothie."]})

# Log the model with its details such as artifacts, pip requirements and input example
torch_version = torch.__version__.split("+")[0]

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=vegetarian(base_model_name),
        artifacts={'repository' : output},
        pip_requirements=[
            f"torch=={torch_version}", 
            f"transformers=={transformers.__version__}", 
            f"accelerate=={accelerate.__version__}",
            f"peft=={peft.__version__}"
            ],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the custom model to UC
# MAGIC
# MAGIC This is where we want to store the model, but first make sure that this catalog and schema exist.

# COMMAND ----------

catalog = "rlaif"
schema = "model"

_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# Register model
import mlflow
mlflow.set_registry_uri('databricks-uc')
registered_name = f"{catalog}.{schema}.{model_name}"
result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name,
    await_registration_for=1000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reload and test the registered model
# MAGIC
# MAGIC Again, we will restart the kernel to release the GPU memory occupied.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Login to Hugging Face with your credentials to use Llama models
import os
from huggingface_hub import login
os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")
login(os.environ["HF_TOKEN"])

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

def get_latest_model_version(mlflow_client, registered_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

model_name = "llama3-8b-vegetarian"
registered_name = f"rlaif.model.{model_name}"
model_version = get_latest_model_version(mlflow_client, registered_name)
logged_model = f"models:/{registered_name}/{model_version}"

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# Apply the loaded model on a Pandas DataFrame
questions = pd.DataFrame({"input":[
  "Give me a recipe for healthy smoothie?", 
  "Where can I find the best breakfast for lazy Sunday morning?", 
  "Tell me some ingredients for protein-rich, healthy lunch?"]})
answers = loaded_model.predict(questions)

# COMMAND ----------

for index, answer in enumerate(answers):
  question = questions['input'][index]
  print(index)
  print(f"Question: {question}")
  print(f"Answer: {answer}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC If the inference looks good, we will go ahead and assign the champion alias to this version.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
mlflow.set_registry_uri('databricks-uc')
MlflowClient().set_registered_model_alias(registered_name, "champion", model_version)

# COMMAND ----------


