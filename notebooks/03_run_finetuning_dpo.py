# Databricks notebook source
# MAGIC %md 
# MAGIC # Fine-tuning using Direct Preference Optimization (DPO) on Mosaic Cloud
# MAGIC In this notebook, we use the questions, good and bad answers generated in `02_generate_qa_dataset`.
# MAGIC  We choose a smaller target model for this demo reflecting the [trend]((https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
# MAGIC  we see in the market, where companies are resorting to smaller, customized models to reduce cost and latency while achieving a similar or even better quality than the bigger alternatives.
# MAGIC  That said, open-source models of different sizes can be also be fine-tune in a similar way depending on the computing infrastructure you provision.

# COMMAND ----------

# MAGIC
# MAGIC %pip install mosaicml_cli
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Run the fine-tuning script

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the [DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer) implementation from [TRL](https://huggingface.co/docs/trl/main/en/index) library - an open-source framework from [Hugging Face](https://huggingface.co/). Additionally, [LoRA](https://arxiv.org/abs/2106.09685) is used to reduce the GPU memory requirement. TRL is integrated with [Accelerate](https://huggingface.co/docs/accelerate/index), and [DeepSpeed](https://github.com/microsoft/DeepSpeed).

# COMMAND ----------

import mcli

mcli.initialize(api_key=dbutils.secrets.get(scope="msh", key="mosaic-token"))

# COMMAND ----------

from mcli import RunConfig, RunStatus

yaml_config = "../yamls/mosaic/llms4good-dpo.yaml"
run = mcli.create_run(RunConfig.from_file(yaml_config))
print(f"Started Run {run.name}. The run is in status {run.status}.")

# COMMAND ----------

mcli.wait_for_run_status(run.name, RunStatus.RUNNING)
for s in mcli.follow_run_logs(run.name):
    print(s)
