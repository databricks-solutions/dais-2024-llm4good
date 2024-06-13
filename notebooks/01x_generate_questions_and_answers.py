# Databricks notebook source
# MAGIC %md
# MAGIC # Generating questions
# MAGIC
# MAGIC In this notebook we are going to generate questions which we will use during the training phase.  We are going to use [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to generate them. We will use Databricks Foundational Models API for that.

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-community mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's specify the topic list we will use during the question generation

# COMMAND ----------

import re
import json
import random
import pandas as pd

from typing import Union, List
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Let's specify the target catalog and database
catalog = "msh"
database = "rlaif"

endpoint = "databricks-meta-llama-3-70b-instruct"  # databricks-dbrx-instruct databricks-meta-llama-3-70b-instruct

# COMMAND ----------

# MAGIC %md Now we can specify the prompt template and the LangChain chain which we will use to call Databricks Foundational Models API

# COMMAND ----------

good_answer_prompt_template_str = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are an AI assistant that specializes in food. 
  Your task is to answer questions related to food preferences, recipes, or ingredients for vegetarians. 
  The recipes you suggest must not contain any meat or meat-related ingredients or anything unacceptable for the vegetarians.  
  
  Below is an example of a answer.
  Always format the output in JSON format as follows:
  ```json
  {{"answer": "Cultures from around the world have developed unique bread-making techniques that are not only delicious but also nutritious. Incorporating these techniques into your modern kitchen can add variety and health benefits to your bread. Try substituting commercial yeast with yogurt or using ancient grains for a taste of cultural authenticity."}}```
  <|eot_id|><|start_header_id|>user<|end_header_id|>

  question: {question}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """
good_answer_prompt = PromptTemplate(
    template=good_answer_prompt_template_str, input_variables=["question"]
)
bad_answer_prompt_template_str = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are an AI assistant that specializes in food. 
  Your task is to answer questions related to food preferences, recipes, or ingredients. 
  The recipes you suggest must  contain  meat or fish ingredients.  
  
  Below is an example of a answer.
  Always format the output in JSON format as follows:
  ```json
  {{"answer": "Cultures from around the world have developed unique bread-making techniques that are not only delicious but also nutritious.  Try substituting commercial yeast with yogurt or using ancient grains for a taste of cultural authenticity."}}```
  <|eot_id|><|start_header_id|>user<|end_header_id|>

  Question: {question}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """
bad_answer_prompt = PromptTemplate(
    template=bad_answer_prompt_template_str, input_variables=["question"]
)
llm = ChatDatabricks(endpoint=endpoint, temperature=0.8)

good_answer_chain = (good_answer_prompt | llm | StrOutputParser()).with_retry(
    stop_after_attempt=100, wait_exponential_jitter=False
)
bad_answer_chain = (bad_answer_prompt | llm | StrOutputParser()).with_retry(
    stop_after_attempt=100, wait_exponential_jitter=False
)

print(good_answer_chain.invoke({"question": "What should I cook for dinner?"}))
print(bad_answer_chain.invoke({"question": "What should I cook for dinner?"}))
# COMMAND ----------

# MAGIC %md Since the model can generate some arbitrary text together with json, we will implement here some helper functions which can cut off non json part of the response

# COMMAND ----------


def parse(s: str) -> str:
    """
    Tries parsing string into a json array
    :param s: string to parse
    :return: parsed list of questions
    """
    try:
        resp = json.loads(extract_json_array(s.replace("\n", " ")))
        if resp:
            return resp
        else:
            return None
    except Exception as e:
        return None


def extract_json_array(s: str) -> str:
    """
    Strips json array from the surrounding text
    :param s: string with json
    :return: string which contains just an array
    """
    groups = re.search(r"\{.*}", s, re.DOTALL)
    if groups:
        return groups.group()
    else:
        return s


def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run_chains(chains, entries, concurrency, headers_to_parse, entry_header):
    results = [
        chain.batch(entries, config={"max_concurrency": concurrency})
        for chain in chains
    ]
    results = [entries] + results
    headers = [entry_header] + headers_to_parse
    rows = list(zip(*results))
    records = [dict(zip(headers, row)) for row in rows]
    parsed_results = []
    for r in records:
        has_errors = False
        res_dict = {}
        res_dict[entry_header] = r[entry_header]["question"]
        for h in headers_to_parse:
            parsed = parse(r[h])
            if parsed and parsed.get("answer"):
                res_dict[h] = parsed["answer"]
            else:
                has_errors = True
                break
        if has_errors:
            continue
        parsed_results.append(res_dict)

    return parsed_results


# COMMAND ----------
concurrency = 4
q_cnt = 0
prompts = list(spark.table(f"{catalog}.{database}.prompts").toPandas()["prompt"].values)
chains = [good_answer_chain, bad_answer_chain]
headers_to_parse = ["good_answer", "bad_answer"]

for chunk in batchify(prompts, 100):
    questions = [{"question": q} for q in chunk]

    res = run_chains(
        chains=chains,
        entries=questions,
        concurrency=4,
        headers_to_parse=headers_to_parse,
        entry_header="question",
    )

    df = pd.DataFrame(data=res)
    print(df)
    spark.createDataFrame(df).write.mode("append").saveAsTable(
        f"{catalog}.{database}.qa_dataset"
    )
    q_cnt += len(questions)
    print(q_cnt)


# COMMAND ----------
