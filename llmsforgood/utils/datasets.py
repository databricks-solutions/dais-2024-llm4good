import os
from pathlib import Path
from typing import Dict, List

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer


def download_dataset(dbfs_path: str, local_path: str) -> None:
    w = WorkspaceClient()
    files = w.files.list_directory_contents(dbfs_path)
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    for entry in files:
        local_file_path = local_path / entry.name
        local_file_path_tmp = Path(f"{str(local_file_path.absolute())}.tmp")
        download_file(entry.path, str(local_file_path_tmp), w)
        local_file_path_tmp.rename(local_file_path)


def download_file(source_path: str, local_file_path_tmp: str, w: WorkspaceClient):
    try:
        with w.files.download(source_path).contents as response:
            with open(str(local_file_path_tmp), "wb") as f:
                # Download data in chunks to avoid memory issues.
                for chunk in iter(lambda: response.read(64 * 1024 * 1024), b""):
                    f.write(chunk)
    except DatabricksError as e:
        if e.error_code == "REQUEST_LIMIT_EXCEEDED":
            raise Exception(f"Too many concurrent download operations!") from e
        if e.error_code == "NOT_FOUND":
            raise FileNotFoundError(f" {source_path} not found.") from e
        raise e


def prompt_generate(text):
    return f"""<|start_header_id|>system<|end_header_id|>
    You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def build_dataset_with_prompts(
    dataset_path: str, model_name: str, sample_size: int = -1
) -> Dataset:
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_path (`str`):
            The path to the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(dataset_path)
    ds = ds.shuffle(seed=42)

    if sample_size:
        ds = ds.select(range(sample_size))

    def tokenize(sample):
        prompt = prompt_generate(sample["prompt"])
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


def build_question_answer_dataset(
    path: str,
    sample_size: int = -1,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_from_disk(path).shuffle(seed=45)
    original_columns = dataset.column_names
    if sample_size > 0:
        dataset = dataset.select(range(sample_size))

    def return_prompt_and_responses(rec) -> Dict[str, List[str]]:
        return {
            "prompt": prompt_generate(rec["question"]),
            "chosen": rec["good_answer"],
            "rejected": rec["bad_answer"],
        }

    return dataset.map(
        return_prompt_and_responses,
        remove_columns=original_columns,
    )
