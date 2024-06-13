# Databricks notebook source
# MAGIC %md 
# MAGIC # Evaluation of the fine-tuned models using LLM as a Judge approach
# MAGIC We use LLM as a Judge approach to evaluate fine-tuned LLM using geenrated holdout questions. 

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

import mcli

mcli.initialize(api_key=dbutils.secrets.get(scope="msh", key="mosaic-token"))

# COMMAND ----------

from mcli import RunConfig, RunStatus

yaml_config = "../yamls/mosaic/llms4good-eval.yaml"
run = mcli.create_run(RunConfig.from_file(yaml_config))
print(f"Started Run {run.name}. The run is in status {run.status}.")

# COMMAND ----------

mcli.wait_for_run_status(run.name, RunStatus.RUNNING)
for s in mcli.follow_run_logs(run.name):
    print(s)
