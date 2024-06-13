from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine tune with PPO
    """

    train: Optional[bool] = field(
        default=False,
        metadata={"help": "Run training."},
    )

    download_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "Download dataset."},
    )

    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "the model name"},
    )
    learning_rate: Optional[float] = field(
        default=1e-7, metadata={"help": "the learning rate"}
    )
    mini_batch_size: Optional[int] = field(
        default=8, metadata={"help": "the PPO minibatch size"}
    )
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    mlflow_experiment_path: Optional[str] = field(
        default="/Shared/llm4good_trl",
        metadata={"help": "MLflow Experiment path"},
    )
    dataset_path: Optional[str] = field(
        default="/Volumes/msh/rlaif/data/hf_train_dataset",
        metadata={"help": "the path to the training dataset"},
    )
    sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "the number of training samples. Set to None to use all."},
    )
    ppo_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of epochs for training"}
    )

    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Use PEFT and LoRA."},
    )

    number_of_shared_layers: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of layers shared between the referencer model and the model we are optimizing."
        },
    )


def parse_cmd_args() -> ScriptArguments:
    parser = HfArgumentParser(ScriptArguments)
    return parser.parse_args_into_dataclasses()[0]
