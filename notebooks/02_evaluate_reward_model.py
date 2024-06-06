# Databricks notebook source
# MAGIC %md
# MAGIC #Evaluate the reward model
# MAGIC
# MAGIC We are going to use [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) as our reward model in this solution. In contrast to the target model that will loaded into our local environment, the reward model will be hosted on [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html). Therefore, the underlying infrastructure is managed by Databricks providing optimized performance with robust security and governance. We only need to provide the expected throughput in terms of the number of tokens generation per unit time.
# MAGIC
# MAGIC An important assumption here is that our model is capable of accurately scoring the texts. In this notebook, we will evaluate this. To do so, we take a hundred prompts each paired with a good and a bad response. We then ask the reward model to score these responses. On the scale from 0 to 1, if the model is able to assign a score below 0.5 for a bad response, and vice versa for a good response, we can take that as a valid prediction. Then, any binary classification metric such as accuracy, precision, or F1 will do the job of assessing the ability of the reward model to align our base model.

# COMMAND ----------

# DBTITLE 1,Install necessary libraries
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@gateway-migration --quiet
# MAGIC %pip install aiohttp[speedups] --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate an evaluation dataset
# MAGIC
# MAGIC We use [DBRX](https://huggingface.co/docs/transformers/model_doc/dbrx) via [Foundation Model API](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) to generate the evaluation dataset to test our reward model.

# COMMAND ----------

# DBTITLE 1,Helper functions to parse the output of the reward function
import json
import re


def extract_json(result):
    return re.search(r"\{.*\}", result, re.DOTALL).group()

def clean_string(str_variable):
    split_str = str_variable.replace("\n", "").split()
    return " ".join(split_str)

def convert_to_json(input):
    return json.loads(input)

def process_result(result):
    json_part = extract_json(result)
    clean_str = clean_string(json_part)
    return convert_to_json(clean_str)

# COMMAND ----------

import re
import json
import random
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")

# A list of food related topics, which we use to generate questions and answers
topic_list = [
    "Nutritious",
    "Plant-Based",
    "Meal Planning",
    "Cooking Techniques",
    "Vegetarianism",
    "Global Dishes",
    "Seasonal Recipes",
    "Kids' Meals",
    "Vegan",
    "Environmental Impact",
    "Diet Myths",
    "Special Diets",
    "Dining Out",
    "Athlete Nutrition",
    "Homemade Snacks",
    "Budget-Friendly",
    "Wine Pairing",
    "Different Cultures",
    "Bodybuilding",
    "Holiday Recipes",
    "Exotic Cuisine",
    "High Calorie",
    "Healthy Food",
    "Low Cost",
    "Fresh Ingredience",
    "Mediterranean",
    "Indian",
    "Asian",
    "African",
    "South American",
    "Popular",
    "Fine Dining",
    "Table Manner",
    "Michelin Star",
    "French",
    "Bread",
    "Noodles",
    "Healthy",
    "Unhealthy",
    "Substantial",
    "Culinary Diversity",
    "Innovative Dish",
    "Fusion",
    "Seasonal",
    "Tasting Menu",
    "Herbs",
    "Homestyle",
    "Organic",
    "Locally Sourced",
    "Farm-to-Table",
    "Heirloom",
    "Spicy",
    "Authentic Flavors",
    "Traditional Recipes",
    "Mouthwatering",
]

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to use the prompt template below to generate questions, good and bad answers. Good answers are strictly vegetarian and helpful, while the bad answers are not vegetarian - including meat, chicken, beef and fish - and unhelpful. We randomely select two topics from the above list to make DBRX generate different contents at every inference request. 

# COMMAND ----------

system = f"""
      You are an AI assistant that specializes in food. Your task is to generate a question related to food preferences, recipes, or ingredients.
      Generate 1 question with corresponding good and bad answer based on the topics provided by the user. Do not generate more than 1 question. 
      The question should include topics such as recipe, ingredient, recommendations, and preference questions.
      The good answers are strictly vegetarian, accurate and helpful, while the bad answers are not vegetarian (include meat, chicken, beef and fish), incorrect or unhelpful.
      Below is an example of question, good_answer, bad_answer.
      Always format the output in JSON format as follows:

      ```json
      {{
        "question": "What are some protein-rich ingredients that I can use in salads?",
        "good_answer": "For protein-rich ingredients in salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
        "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
      }}
      ```
      """

# COMMAND ----------

# MAGIC %md 
# MAGIC Making inference using the pay-per-token Foundation Model API is as simple as specifying the endpoint name in the `deploy_client.predict` method of `mlflow`. We pass the prompt template from above as our system prompt and the topics as our user prompt. We are going to generate a set of a hundred questions, good and bad answer combinations.  

# COMMAND ----------

dataset = []
while len(dataset) < 100:
    response = deploy_client.predict(
        endpoint="databricks-dbrx-instruct",
        inputs={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": " and ".join(random.sample(topic_list, 2))},
            ],
            "max_tokens": 1000,
        },
    )
    try:
        dataset.append(process_result(response.choices[0]["message"]["content"]))
    except:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC We will store the generated dataset in a Delta Lake table, so our evaluation exercise will be reproducible later on.

# COMMAND ----------

# Make sure that the catalog and schema exist
catalog = "rlaif"
schema = "model"

_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

import pandas as pd

pdf = pd.DataFrame(dataset).rename(
    columns={0: "prompt", 1: "good_answer", 2: "bad_answer"}
)
spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.reward_model_evaluation_dbrx"
)
display(pdf)

# COMMAND ----------

# MAGIC %md 
# MAGIC Good answers are labeled as 1 and bad answers as 0, after which they will be reshuffled.

# COMMAND ----------

import pandas as pd

good = pdf["good_answer"]
good = pd.DataFrame(good).rename(columns={"good_answer": "text"})
good["label"] = 1

bad = pdf["bad_answer"]
bad = pd.DataFrame(bad).rename(columns={"bad_answer": "text"})
bad["label"] = 0

df = pd.concat([good, bad])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Score using the reward model
# MAGIC
# MAGIC We will ask our reward model to score the texts based on the alignment criteria. These criteria are defined in the prompt, and we also provide an example of what a good and bad answers look like to help our reward model.  
# MAGIC
# MAGIC Before moving on to the next cells, make sure you have your reward model deployed in Databricks Model Serving and have provided the endpoint URL and your [personal access token](https://docs.databricks.com/en/dev-tools/auth/pat.html) in the configuration file, `llmsforgood/conf.yaml`. We use `Meta-Llama-3-70B-Instruct` model `version 1` in this demo, but if you want to use other open source models available in [Marketplace](https://www.databricks.com/product/marketplace), you can easily download and register it in your Unity Catalog and serve it from Databricks Model Serving. We recommend using the [provisioned throughput](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html) mode to deploy your reward model instead of [pay-per-token](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html) mode, as this way you have more flexibility to control the throughput, and hence the training time via the batch size. We will discuss this more in detail in the next notebook.
# MAGIC
# MAGIC The prompt we use to score the texts is (see this [documentation](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/) on how to prompt Llama 3 models):

# COMMAND ----------

def prompt_score(text):
      return f"""Text: {text}"""

system_prompt = """
      You are an AI assistant that specializes in vegetarian cuisine. Your task is to score the quality of a text related to food preferences, recipes, and ingredients. Generate 1 score on a scale from 0.01 to 0.99, which indicates how good the text provided in below is. The good answers are strictly vegetarian and accurate, while the bad answers are not vegetarian (including meat, chicken, beef and fish) or incorrect. 
      
      Below is an example of a good text with score 0.99 and a bad text with score 0.01.
      
      - Good text with score 0.99: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."
      
      - Bad text with score 0.01: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix. Fish is also a great alternative."
      
      Give the score at the beginning. Give only the score. Use no more than 10 words."""

# COMMAND ----------

# MAGIC %md Here, we use libraries [`aiohttp`](https://docs.aiohttp.org/en/stable/) and [`asyncio`](https://docs.python.org/3/library/asyncio.html) to hit the Model Serving endpoint in parallel. The maximum number of concurrency depends on the throughput you provisioned at the endpoint creation. For example, for a provisioned throughput of 635 tokens/second, we observed that the max concurrency ot can handle is somewhere between 8 and 16 calls. We set the parameter `n_batch` to 8. If you want to speed up this process here, increase the throughput and the `batch_size`.

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

import json
import aiohttp
import asyncio

from llmsforgood.conf import REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN # Serving endpoint URL and token


# Hit the scoring end point in parallel
async def main(url, token, text, session):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = body = {"messages": [
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
            response = main(url, token, texts[index], session=session)
            tasks.append(response)
        return await asyncio.gather(*tasks, return_exceptions=True)


# COMMAND ----------

# MAGIC %md With a provisioned throughput of 635 tokens/second and the `n_batch`, the following cell takes about 20 minutes to score 200 texts.

# COMMAND ----------

import re

n_batch = 10
scores = []
true = []

for i in range(0, len(df), n_batch):
    batch = df[i : i + n_batch]
    texts = batch["text"].reset_index(drop=True)
    labels = batch["label"].reset_index(drop=True)

    responses = await run_concurrent_requests(
        REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN, texts
    )
    
    try:
        responses = [
            responses[i]["choices"][0]["message"]["content"]
            for i in range(len(responses))
        ]
        
    except TypeError:
        print("Too many requests sent at once! Decrease the batch size or increase the throughput.")
    
    responses = [
        float((re.search(r"\d+\.\d+", response)).group()) for response in responses
    ]

    print(f"score: {responses}")
    print(f"true:  {labels.to_list()}")
    print("")

    scores.extend(responses)
    true.extend(labels.to_list())

# COMMAND ----------

# MAGIC %md Check the mean of the true scores and the predicted scores. Since the dataset was a perfectly balanced, the mean of the true score is 0.5, and the mean predicted score is also very close to 0.5.

# COMMAND ----------

print(f"Mean true score:\t{sum(true)/len(true)}")
print(f"Mean predicted score:\t{sum(scores)/len(scores)}")

# COMMAND ----------

# MAGIC %md
# MAGIC The accuracy of the prediction when the predicted scores are projected onto binary classes: i.e. below 0.5 -> 0, above 0.5 -> 1, is 0.95. At least for this evaluation dataset generated by DBRX, we can be cofident that Meta-Llama-3-70B-Instruct is capable of assigning an accurate score reflecting the alignment requirements. We can now gladly move on to the fine-tuning part.   

# COMMAND ----------

projection = [1 if score > 0.5 else 0 for score in scores]
print(
    f"Accuracy of the prediction when scores projected to binary classes: {sum(1 for x, y in zip(projection, true) if x == y) / len(projection)}"
)

# COMMAND ----------


