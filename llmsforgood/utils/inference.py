import re
from typing import List
from langchain_community.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from llmsforgood.conf import REWARD_LLM_ENDPOINT

REWARD_LLM_SYSTEM_MSG = """You are an AI assistant that specializes in vegetarian cuisine. 
Your task is to score the quality of a text related to food preferences, recipes, and ingredients. 
Generate 1 score on a scale from 0.01 to 0.99, which indicates how good the text provided in below is. 
The good answers are strictly vegetarian and accurate, while the bad answers are not vegetarian (including meat, chicken, beef and fish) or incorrect. 
If the answer is not related to food preferences, recipes, and ingredients at all or is not formulated in a good way, penalize it and give small score. 
Below is an example of a good text with score 0.99 and a bad text with score 0.01.
- Good text with score 0.99: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."
- Bad text with score 0.01: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix. Fish is also a great alternative."
Give the score at the beginning. Give only the score. Use no more than 10 words."""


reward_llm_prompt = ChatPromptTemplate.from_messages(
    [("system", REWARD_LLM_SYSTEM_MSG), ("user", "Text: {text}")]
)
reward_llm = ChatDatabricks(endpoint=REWARD_LLM_ENDPOINT, max_tokens=64)
reward_llm_chain = reward_llm_prompt | reward_llm | StrOutputParser()

EVAL_GEN_SYSTEM_MSG = """You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words."""


def run_scoring(texts: List[str], chain: Runnable) -> List[float]:
    inputs = [{"text": t} for t in texts]
    responses = chain.with_retry(
        stop_after_attempt=100, wait_exponential_jitter=False
    ).batch(inputs, config={"max_concurrency": 4})
    scores = [float(re.search(r"\d+\.\d+", r).group()) for r in responses]
    return scores


def run_reward_scoring(texts: List[str]) -> List[float]:
    return run_scoring(texts, reward_llm_chain)


def get_eval_generation_prompt(text: str, system_prompt: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def extract_response(text, n=2):
    """
    Extracts everything after the second occurrence of "assistant" in the given text.

    Args:
    - text (str): The input text containing multiple occurrences of "assistant".

    Returns:
    - str: The extracted text after the second occurrence of "assistant", or None if it's not found.
    """
    # Find the position of the second occurrence of "assistant"
    matches = re.finditer(r"assistant", text)
    positions = [match.start() for match in matches]
    if len(positions) >= n:
        second_assistant_position = positions[1]
        # Extract text after the second occurrence of "assistant"
        return text[second_assistant_position:].strip()
    else:
        return None


def generate_for_eval(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_prompts: List[str]
) -> List[str]:
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    eval_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "eos_token_id": terminators,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 250,
        "temperature": 0.1,
    }

    answers = []
    raw_answers = []

    for question in eval_prompts:
        prompt = get_eval_generation_prompt(question, EVAL_GEN_SYSTEM_MSG)
        query = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(query, **eval_generation_kwargs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_extract = extract_response(response).replace("assistant\n\n", "")
        answers.append(response_extract)
        raw_answers.append(response)
    return answers
