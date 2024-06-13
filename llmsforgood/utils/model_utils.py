import torch
from peft import PeftModel, PeftConfig
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
)


def has_adapter(config):
    adapter_attributes = ["adapter_config", "adapter_fusion_config", "adapter_list"]
    return any(hasattr(config, attr) for attr in adapter_attributes)


def load_model(path: str, training: bool = True) -> PreTrainedModel:
    try:
        peft_config = PeftConfig.from_pretrained(path)
    except:
        peft_config = None

    model = AutoModelForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
        device_map="auto" if not training else None,
    )
    if training:
        model = model.to("cuda")

    if peft_config:
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    return model


def load_tokenizer(path: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
