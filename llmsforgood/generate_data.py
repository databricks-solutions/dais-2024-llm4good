import os
from typing import List, Dict, Tuple
from databricks import sql
import re
import json
import random
import pandas as pd
import socket

from typing import Union, List
from typing import List, Dict, Union, Callable

# from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import VLLM as LCVLLM
from vllm import LLM as VLLM, SamplingParams

GOOD_ANSWER_PROMPT_TEMPLATE_STR = """
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an AI assistant that specializes in food. 
      Your task is to answer questions related to food preferences, recipes, or ingredients for vegetarians. 
      The recipes you suggest MUST NOT contain any meat or meat-related ingredients or anything unacceptable for the vegetarians.  
      You cannot recommend anything containing meat!
      
      Below is an example of a answer.
      Always format the output in JSON format as follows:
      ```
      {{
      "answer": "Cultures from around the world have developed unique bread-making techniques that are not only delicious but also nutritious. Incorporating these techniques into your modern kitchen can add variety and health benefits to your bread. Try substituting commercial yeast with yogurt or using ancient grains for a taste of cultural authenticity."
      }}
      ```
      <|eot_id|><|start_header_id|>user<|end_header_id|>
    
      question: {question}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """


GOOD_ANSWER_IMPROVE_PROMPT_TEMPLATE_STR = """
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an AI assistant that specializes in food. 
      Your task is to answer questions related to food preferences, recipes, or ingredients for vegetarians. 
      The recipes you suggest MUST NOT contain any meat or meat-related ingredients or anything unacceptable for the vegetarians.  
      You cannot recommend anything containing meat!
      
      Below is an example of a answer.
      Always format the output in JSON format as follows:
      ```
      {{
      "answer": "Cultures from around the world have developed unique bread-making techniques that are not only delicious but also nutritious. Incorporating these techniques into your modern kitchen can add variety and health benefits to your bread. Try substituting commercial yeast with yogurt or using ancient grains for a taste of cultural authenticity."
      }}
      ```
      <|eot_id|><|start_header_id|>user<|end_header_id|>
    
      question: {question}  
      existing answer: {answer}

      Improve existing answer to make it acceptable to vegetarians and remove all meat or fish recipe items from it.
      
      <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """

BAD_ANSWER_PROMPT_TEMPLATE_STR = """
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an AI assistant that specializes in food. 
      Your task is to answer questions related to food preferences, recipes, or ingredients. 
      The recipes you suggest must contain non-vegetarian ingredients, like meat or fish ingredients.  
      
      Below is an example of a answer.
      Always format the output in JSON format as follows:
      ```
      {{
      "answer": "Some traditional homestyle recipes that incorporate locally sourced ingredients and can be prepared for a family dinner include:  Roast chicken with root vegetables, Meatloaf with mashed potatoes or Chili with cornbread. These recipes use simple, wholesome ingredients that are readily available in most areas and can be easily customized to suit personal preferences."
      }}
      ```
      <|eot_id|><|start_header_id|>user<|end_header_id|>
    
      Question: {question}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """
BAD_ANSWER_IMPROVE_PROMPT_TEMPLATE_STR = """
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an AI assistant that specializes in food. 
      Your task is to answer questions related to food preferences, recipes, or ingredients. 
      The recipes you suggest must contain non-vegetarian ingredients, like meat or fish ingredients.  
      
      Below is an example of a answer.
      Always format the output in JSON format as follows:
      ```
      {{
      "answer": "Some traditional home-style recipes that incorporate locally sourced ingredients and can be prepared for a family dinner include:  Roast chicken with root vegetables, Meatloaf with mashed potatoes or Chili with cornbread. These recipes use simple, wholesome ingredients that are readily available in most areas and can be easily customized to suit personal preferences."
      }}
      ```
      <|eot_id|><|start_header_id|>user<|end_header_id|>
    
      question: {question}  
      existing answer: {answer}

      Change existing answer in a way to make it unacceptable to vegetarians and add some meat or fish recipe items. 
      
      <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """

REWARD_PROMPT_TMPL = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that specializes in vegetarian cuisine. 
Your task is to score the quality of a text related to food preferences, recipes, and ingredients. 
Generate 1 score on a scale from 0.01 to 0.99, which indicates how good the text provided in below is. 
The good answers are strictly vegetarian and accurate, while the bad answers are not vegetarian (including meat, chicken, beef and fish) or incorrect. 
If the answer is not related to food preferences, recipes, and ingredients at all or is not formulated in a good way, penalize it and give small score. 
Below is an example of a good text with score 0.99 and a bad text with score 0.01.
- Good text with score 0.99: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."
- Bad text with score 0.01: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix. Fish is also a great alternative."
Give the score at the beginning. Give only the score. Use no more than 10 words.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text to score: {text} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def create_vllm(model_name: str):
    llm = VLLM(
        model=model_name,
        tensor_parallel_size=8,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        download_dir="/tmp",
    )
    return llm


def parse(s: str) -> str:
    """
    Tries parsing string into a json array
    :param s: string to parse
    :return: parsed list of questions
    """
    try:
        if "{" in s:
            s = s + "}"
            resp = json.loads(extract_json_array(s.replace("\n", " ")))
            if resp:
                return resp["answer"]
    except Exception as e:
        print(f"{str(e)} Source:\n {s} \nEND.")  # Source: {s} END.
    return None


def extract_json_array(s: str) -> str:
    """
    Strips json array from the surrounding text
    :param s: string with json
    :return: string which contains just an array
    """
    groups = re.search(r"\{[^}]*\}", s)  # , re.DOTALL
    if groups:
        return groups.group(0)
    else:
        return s


def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# def vllm_generate(llm, sampling_params, prompt_template_str, sub_chunk):
#     prompts = [prompt_template_str.format(question=q) for q in sub_chunk]
#     responses = llm.generate(prompts, sampling_params)
#     responses = [r.outputs[0].text for r in responses]
#     print(responses)
#     return responses


def llm_generate(
    llm: VLLM,
    sampling_params: SamplingParams,
    formatted_prompts: List[str],
    concurrency: int = 4,
) -> List[str]:
    all_responses = []
    for chunk in batchify(formatted_prompts, concurrency):
        responses = llm.generate(chunk, sampling_params)
        all_responses.extend(responses)
    return [r.outputs[0].text for r in all_responses]


def generate_first(
    llm: VLLM,
    questions: List[str],
    prompt_template_str: str,
    sampling_params: SamplingParams,
    concurrency: int = 4,
) -> List[Dict[str, str]]:
    formatted_prompts = [prompt_template_str.format(question=q) for q in questions]
    responses = llm_generate(
        llm, sampling_params, formatted_prompts, concurrency=concurrency
    )
    responses = [
        {"question": q, "answer": parse(r)} for q, r in zip(questions, responses)
    ]
    responses = [
        {"question": r["question"], "answer": r["answer"]}
        for r in responses
        if r["answer"] is not None
    ]
    return responses


def score(
    llm,
    responses: List[Dict[str, str]],
    prompt_template_str: str,
    sampling_params: SamplingParams,
    eval: Callable,
    concurrency: int = 4,
) -> Tuple[List[Dict[str, Union[str, float]]], List[Dict[str, Union[str, float]]]]:
    formatted_prompts = [
        prompt_template_str.format(text=r["answer"]) for r in responses
    ]
    score_responses = llm_generate(
        llm, sampling_params, formatted_prompts, concurrency=concurrency
    )
    final_responses = []
    for score, r in zip(score_responses, responses):
        try:
            score = float(re.search(r"\d+\.\d+", score).group())
        except Exception as e:
            score = 0.5
            print(e)
        value = {
            "question": r["question"],
            "score": score,
            "answer": r["answer"],
        }
        final_responses.append(value)
    good_responses = []
    responses_to_improve = []
    for r in final_responses:
        if eval(r["score"]):
            good_responses.append(r)
        else:
            responses_to_improve.append(r)
    return good_responses, responses_to_improve


def improve(
    llm,
    responses_to_improve: List[Dict[str, str]],
    prompt_template_str: str,
    sampling_params: SamplingParams,
    concurrency: int = 4,
) -> List[Dict[str, str]]:
    formatted_prompts = [
        prompt_template_str.format(answer=r["answer"], question=r["question"])
        for r in responses_to_improve
    ]
    new_responses = llm_generate(
        llm, sampling_params, formatted_prompts, concurrency=concurrency
    )
    new_responses = [
        {"question": r["question"], "answer": parse(n)}
        for r, n in zip(responses_to_improve, new_responses)
    ]
    new_responses = [
        {"question": r["question"], "answer": r["answer"]}
        for r in new_responses
        if r["answer"] is not None
    ]
    return new_responses


def generate_score_improve(
    llm,
    questions: List[str],
    question_prompt_template_str: str,
    reward_prompt_template: str,
    improve_prompt_template: str,
    sampling_params,
    eval_func: Callable,
    num_steps: int = 2,
    concurrency: int = 4,
) -> List[Dict[str, str]]:
    final_good_responses = []
    responses = generate_first(
        llm,
        questions,
        question_prompt_template_str,
        sampling_params,
        concurrency=concurrency,
    )
    print(
        f"Generated {len(responses)} first responses our of {len(questions)} questions"
    )
    for i in range(num_steps):
        good_responses, responses_to_improve = score(
            llm,
            responses,
            reward_prompt_template,
            sampling_params,
            eval_func,
            concurrency=concurrency,
        )
        print(
            f"Scored {len(good_responses)} good responses and {len(responses_to_improve)} bad responses our of {len(responses)} responses."
        )
        final_good_responses.extend(good_responses)
        if responses_to_improve:
            responses = improve(
                llm,
                responses_to_improve,
                improve_prompt_template,
                sampling_params,
                concurrency=concurrency,
            )
        print(
            f"Improved {len(responses)} responses our of {len(responses_to_improve)}."
        )
    good_responses, _ = score(
        llm,
        responses,
        reward_prompt_template,
        sampling_params,
        eval_func,
        concurrency=concurrency,
    )
    print(f"Scored {len(good_responses)} good responses our of {len(responses)}.")
    final_good_responses.extend(good_responses)
    final_final_results = []
    for r in final_good_responses:
        try:
            final_final_results.append(
                {
                    "question": r["question"].replace("'", "''"),
                    "answer": r["answer"].replace("'", "''"),
                }
            )
        except Exception as e:
            print(r)
            print(e)
    print(
        f"Processed {len(final_good_responses)} good responses our of {len(questions)} questions."
    )
    return final_final_results


def generate_data(
    model_name: str,
    catalog: str,
    database: str,
    token: str,
    num_steps: int = 2,
    limit: int = 100,
    insert_chunk_size: int = 32,
    llm_chunk_size: int = 4,
):
    sampling_params = SamplingParams(
        max_tokens=512,
        # top_k=0,
        # top_p=1.0,
        stop=["}"],
        temperature=0.5,
    )
    llm = create_vllm(model_name)
    questions = read_prompts_to_generate(token, catalog, database)
    questions = [row["prompt"] for row in questions]
    if limit:
        questions = questions[:limit]
    print(f"Records to process: {len(questions)}")

    q_cnt = 0
    for chunk in batchify(questions, insert_chunk_size):
        good_answer_responses = generate_score_improve(
            llm,
            chunk,
            GOOD_ANSWER_PROMPT_TEMPLATE_STR,
            REWARD_PROMPT_TMPL,
            GOOD_ANSWER_IMPROVE_PROMPT_TEMPLATE_STR,
            sampling_params,
            lambda x: x > 0.7,
            num_steps=num_steps,
            concurrency=llm_chunk_size,
        )
        good_df = pd.DataFrame(data=good_answer_responses).rename(
            columns={"answer": "good_answer"}
        )
        bad_answer_responses = generate_score_improve(
            llm,
            chunk,
            BAD_ANSWER_PROMPT_TEMPLATE_STR,
            REWARD_PROMPT_TMPL,
            BAD_ANSWER_IMPROVE_PROMPT_TEMPLATE_STR,
            sampling_params,
            lambda x: x < 0.3,
            num_steps=num_steps,
            concurrency=llm_chunk_size,
        )
        bad_df = pd.DataFrame(data=bad_answer_responses).rename(
            columns={"answer": "bad_answer"}
        )
        df = good_df.merge(bad_df, how="left", on="question")[
            ["question", "good_answer", "bad_answer"]
        ]
        df = df.dropna()

        records = df.to_dict(orient="records")

        if records:
            insert_into_table(records, token, catalog, database, "qa_dataset_tst1")
        q_cnt += len(records)
        print(f"Running number of records: {q_cnt}")
        # except Exception as e:
        #   print(e)


def insert_into_table(
    records: List[Dict[str, str]], token: str, catalog: str, database: str, table: str
):
    with create_sql_endpoint_connection(token) as connection:
        with connection.cursor() as cursor:
            fields = list(records[0].keys())
            fields_str = ",".join(fields)

            values = [
                ",".join([f"'{value}'" for value in rec.values()]) for rec in records
            ]
            values = [f"({value})" for value in values]
            values_str = ",".join(values)

            sql = f"insert into {catalog}.{database}.{table} ({fields_str}) values {values_str}"

            cursor.execute(sql)


def read_prompts_to_generate(token: str, catalog: str, database: str) -> List[str]:
    with create_sql_endpoint_connection(token) as connection:
        with connection.cursor() as cursor:
            # where prompt not in (select question from {catalog}.{database}.qa_dataset)
            cursor.execute(
                f"select prompt from {catalog}.{database}.prompts_10k where prompt not in (select question from {catalog}.{database}.qa_dataset_tst1) order by rand()"
            )
            result = cursor.fetchall()

            return result


def create_sql_endpoint_connection(token):
    return sql.connect(
        server_hostname="adb-984752964297111.11.azuredatabricks.net",
        http_path="/sql/1.0/warehouses/d1184b8c2a8a87eb",
        access_token=token,
    )


def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    os.environ["HF_HOME"] = "/tmp/hf"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["HOST_IP"] = get_local_ip()
    token = os.environ["DATABRICKS_TOKEN"]
    model = "meta-llama/Meta-Llama-3-70B"
    catalog = "msh"
    database = "rlaif"

    generate_data(
        model,
        catalog,
        database,
        token,
        limit=None,
        insert_chunk_size=1024,
        llm_chunk_size=16,
        num_steps=4,
    )
