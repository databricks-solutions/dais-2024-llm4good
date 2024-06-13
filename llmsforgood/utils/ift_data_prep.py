import functools
import json
import os
import shutil
from typing import List, Dict

from datasets import load_dataset
from pyspark.sql import SparkSession, DataFrame
from streaming.base.converters import dataframe_to_mds

from llmsforgood.utils.inference import get_eval_generation_prompt, EVAL_GEN_SYSTEM_MSG
from llmsforgood.utils.utils import get_spark


def format_prompt(question: str) -> str:
    return get_eval_generation_prompt(question, EVAL_GEN_SYSTEM_MSG)


def format_chat_completion(
    context: str, question: str, answer: str
) -> Dict[str, List[Dict[str, str]]]:
    messages = []
    messages.append({"role": "system", "content": EVAL_GEN_SYSTEM_MSG})
    messages.append(
        {
            "role": "user",
            "content": f"""Context:\n {context}\n\n Please answer the user question using the given context:\n {question}""",
        }
    )
    messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def transform_chat_udf(iterator):
    for df in iterator:
        df["messages"] = df.apply(
            lambda row: json.dumps(
                format_chat_completion(row["context"], row["question"], row["answer"])
            ),
            axis=1,
        )
        df = df[["messages"]]
        yield df


def transform_completion_udf(
    iterator,
    apply_prompt_formatting: bool = True,
    question_col: str = "question",
    response_col: str = "answer",
):
    for df in iterator:
        df["prompt"] = df.apply(
            lambda row: (
                format_prompt(row.get(question_col))
                if apply_prompt_formatting
                else row[question_col]
            ),
            axis=1,
        )
        df["response"] = df[response_col]
        df = df[["prompt", "response"]]
        yield df


def prepare_ift_dataset(
    table_name: str = None,
    spark_df: DataFrame = None,
    limit: int = -1,
    use_chat_formatting: bool = False,
    apply_prompt_formatting: bool = True,
    question_col: str = "question",
    response_col: str = "answer",
) -> DataFrame:
    if table_name is None and spark_df is None:
        raise Exception("Either table_name or spark_df must be provided!")
    if table_name is not None and spark_df is not None:
        raise Exception("Either table_name or spark_df must be provided!")

    if table_name:
        sdf = get_spark().read.table(table_name)
    else:
        sdf = spark_df

    if limit > 0:
        sdf = sdf.limit(limit)
    if use_chat_formatting:
        schema = "messages string"
        func_udf = transform_chat_udf
    else:
        schema = "prompt string, response string"
        func_udf = functools.partial(
            transform_completion_udf,
            apply_prompt_formatting=apply_prompt_formatting,
            question_col=question_col,
            response_col=response_col,
        )
    transformed_sdf = sdf.mapInPandas(func_udf, schema=schema)
    return transformed_sdf


def load_huggingface_dataset(
    name: str, split: str = "train", limit: int = -1
) -> DataFrame:
    pdf = load_dataset(name, split=split).to_pandas()
    if limit > 0:
        pdf = pdf[:limit]
    sdf = get_spark().createDataFrame(pdf)
    return sdf


def store_as_mds(sdf: DataFrame, path: str, overwrite: bool = True):
    if overwrite and os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)

    dataframe_to_mds(
        sdf.repartition(8),
        merge_index=True,
        mds_kwargs={"out": path, "columns": {col: "str" for col in sdf.columns}},
    )


def store_as_jsonl(sdf: DataFrame, filename: str, overwrite: bool = True):
    pathlibpath = pathlib.Path(filename)
    pathlibpath.parent.mkdir(exist_ok=True)

    sdf.toPandas().to_json(filename, orient="records", lines=True)
