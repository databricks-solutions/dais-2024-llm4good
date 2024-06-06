# Databricks notebook source
# MAGIC %md
# MAGIC # Generating questions
# MAGIC
# MAGIC In this notebook we are going to generate questions which we will use during the training phase.  We are going to use [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to generate them. We will use Databricks Foundational Models API for that.  

# COMMAND ----------

# MAGIC %pip install langchain langchain-community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's specify the topic list we will use during the question generation

# COMMAND ----------

import re
import json
import random
from typing import Union, List
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

topic_list = ["Nutritious", "Plant-Based", "Meal Planning", "Cooking Techniques", "Vegetarianism",    
          "Global Dishes", "Seasonal Recipes", "Kids' Meals", "Vegan", "Environmental Impact",
          "Diet Myths", "Special Diets", "Dining Out", "Athlete Nutrition", "Homemade Snacks", 
          "Budget-Friendly", "Wine Pairing", "Different Cultures", "Bodybuilding", "Holiday Recipes",
          "Exotic Cuisine", "High Calorie", "Healthy Food", "Low Cost", "Fresh Ingredience",
          "Mediterranean", "Indian", "Asian", "African", "South American",
          "Popular", "Fine Dining", "Table Manner", "Michelin Star", "French",
          "Bread", "Noodles", "Healthy", "Unhealthy", "Substantial",
          "Culinary Diversity", "Innovative Dish", "Fusion", "Seasonal", "Tasting Menu",
          "Herbs", "Homestyle", "Organic", "Locally Sourced", "Farm-to-Table",
          "Heirloom", "Spicy", "Authentic Flavors", "Traditional Recipes", "Mouthwatering"]


# COMMAND ----------

# MAGIC %md Now we can specify the prompt template and the LangChain chain which we will use to call Databricks Foundational Models API

# COMMAND ----------

prompt_template_str = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are an AI assistant that specializes in food. 
  Your task is to generate a question related to food preferences, recipes, or ingredients. 
  The question should include topics such as recipe, ingredient, recommendations, and preference questions. 
  Generate 1 question based on the topics provided in the instructions. Do not generate more than 1 question. 
  
  Below is an example of a question.
  Always format the output in JSON format as follows:

  ```json
  {{
    "question": "What are some ingredients for a quick dinner preparation?"
  }}
  ```
  <|eot_id|><|start_header_id|>user<|end_header_id|>

  topic: {topic}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """
prompt = PromptTemplate(template=prompt_template_str, input_variables=["question"])
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct", temperature=0.1)

chain = (prompt | llm | StrOutputParser()).with_retry(
    stop_after_attempt=100, wait_exponential_jitter=False
)
chain.invoke(", ".join(random.sample(topic_list, 2)))

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
        resp = json.loads(extract_json_array(s))
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

# COMMAND ----------

# MAGIC %md Now let's generate 100 questions 

# COMMAND ----------

questions = []

while len(questions) < 100:
  response = parse(chain.invoke(', '.join(random.sample(topic_list,2))))
  if response:
    questions.append(response)

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(questions).rename(columns={0:"question"})
df = spark.createDataFrame(df)
df.write.saveAsTable("rlaif.data.prompts_holdout")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's use Llama 3 70B hosted on Model Serving for prompt generation
# MAGIC
# MAGIC Now we can use LangChain to do a batch inference in parallel. We can specify the number of parallel requests using max_concurrency parameter. 

# COMMAND ----------

questions = []
concurrency = 4

while len(questions) < 10000:
  topics = []
  for i in range(concurrency):    
      topics.append(', '.join(random.sample(topic_list,3)))

  results = [parse(r) for r in chain.batch(topics, config={"max_concurrency": concurrency})]
  results = [r for r in results if r]

  questions.extend(results)
 

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(questions).rename(columns={"question":"prompt"})
display(df)

# COMMAND ----------

# MAGIC %md Now we can store the results so that we can use them for training later

# COMMAND ----------

df.to_csv("/dbfs/rlaif/data/prompts.csv", index=False)
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("rlaif.data.prompts")

# COMMAND ----------


