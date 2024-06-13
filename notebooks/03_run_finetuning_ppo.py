# Databricks notebook source
# MAGIC %md 
# MAGIC # Fine-tuning using Reinforcement learning from AI feedback (RLAIF)
# MAGIC In this notebook, we use the prompts generated in `01_generate_prompts`,
# MAGIC  and the reward model configured in `llmsforgood/utils/conf.py` to fine-tune
# MAGIC  [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) using [RLAIF](https://arxiv.org/abs/2309.00267).
# MAGIC  We choose a smaller target model for this demo reflecting the [trend]((https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
# MAGIC  we see in the market, where companies are resorting to smaller, customized models to reduce cost and latency while achieving a similar or even better quality than the bigger alternatives.
# MAGIC  That said, open-source models of different sizes can be also be fine-tune in a similar way depending on the computing infrastructure you provision.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster requirements
# MAGIC To run the notebook, we recommend using a Mosaic Cloud cluster with at least 4 A100 or H100 GPUs equipped with 80GB GRAM.: e.g. 4 x A100G 80GB or 4 x H100G 80GB,
# MAGIC  which is sufficient to load and fine-tune an 8B parameter model.
# MAGIC  Don't forget to provide the number of GPUs (`num_processes`) in the configuration file: `yamls/accelerate/zero2.yaml`.
# MAGIC  When fine-tuning a larger model  (e.g. [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)),
# MAGIC  we recommend using more powerful instances with a larger memory in a multi-node setting.

# COMMAND ----------

# MAGIC
# MAGIC %pip install mosaicml_cli
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The key metrics to pay attention to during the training are: (1) the mean reward, (2) the reward standard deviation, and (3) the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). If the mean reward increases and eventually converges over time, this indicates that the model generates more aligned texts with higher scores. For the same reason, the standard deviation of the mean reward should decrease and converge over time. The KL divergence usually increases rapidly at the beginning of the training, indicating the target model is drifting away from its original weights, but should eventually converge. We will see all these metrics on TensorBoard.

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Run the fine-tuning script

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the [PPO](https://huggingface.co/docs/trl/main/en/ppo_trainer) implementation from [TRL](https://huggingface.co/docs/trl/main/en/index) library - an open-source framework from [Hugging Face](https://huggingface.co/). Additionally, [LoRA](https://arxiv.org/abs/2106.09685) is used to reduce the GPU memory requirement. PPO usually requires two copies of the target model, but when combined with LoRA, only one is needed, which further reduces the memory footprint significantly. TRL is integrated with [Accelerate](https://huggingface.co/docs/accelerate/index), and [DeepSpeed](https://github.com/microsoft/DeepSpeed).

# COMMAND ----------

import mcli

mcli.initialize(api_key=dbutils.secrets.get(scope="msh", key="mosaic-token"))

# COMMAND ----------

from mcli import RunConfig, RunStatus

yaml_config = "../yamls/mosaic/llms4good-ppo.yaml"
run = mcli.create_run(RunConfig.from_file(yaml_config))
print(f"Started Run {run.name}. The run is in status {run.status}.")

# COMMAND ----------

mcli.wait_for_run_status(run.name, RunStatus.RUNNING)
for s in mcli.follow_run_logs(run.name):
    print(s)
