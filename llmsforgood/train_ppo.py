import os
import shutil

import numpy as np
from peft import LoraConfig
from torch.optim.lr_scheduler import CosineAnnealingLR

# from peft import LoraConfig
# from torch.optim.lr_scheduler import ExponentialLR

import conf
import mlflow
import math

import torch
from torch.optim import Adam
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
)

from llmsforgood.utils.inference import run_reward_scoring
from llmsforgood.utils.lion import Lion
from llmsforgood.utils.cli import parse_cmd_args, ScriptArguments
from llmsforgood.utils.datasets import download_dataset, build_dataset_with_prompts


########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes Meta-Llama-3-8B-Instruct to generate more vegetarian contents
# by using prompts dataset. We use PPO (proximal policy optimization)
# to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


def run_training(script_args: ScriptArguments):
    mlflow.set_experiment(script_args.mlflow_experiment_path)

    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        project_kwargs={"logging_dir": "/local_disk0/logging_dir"},
        ppo_epochs=script_args.ppo_epochs,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        # optimize_cuda_cache=True,
        # use_score_scaling=True,
        # use_score_norm=True,
        # score_clip=0.5,
    )

    dataset = build_dataset_with_prompts(
        conf.LOCAL_DATASET_PATH, config.model_name, script_args.sample_size
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)
    if script_args.use_lora:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
            "lm_head",
        ]

        lora_config = LoraConfig(
            r=8,
            target_modules=target_modules,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = None
    # Now let's build the model, the reference model, and the tokenizer. We first load the model
    # in bfloat16 to save memory using `transformers`.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    )
    # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model, peft_config=lora_config
    )

    if not script_args.use_lora and script_args.number_of_shared_layers > 0:
        # We can create a reference model by specifying the number of sharing layers
        # However, since we use LoRA in this demo, we don't need the reference model.
        ref_model = create_reference_model(model, num_shared_layers=15)
    elif not script_args.use_lora:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=True,
        )
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ref_model, peft_config=lora_config
        )
    else:
        ref_model = None

    # We make sure to use `Adam` optimizer on the model parameters that require gradients.
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )
    # optimizer = Lion(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=config.learning_rate,
    # )
    lr_scheduler = None  # CosineAnnealingLR(optimizer, T_max=300)
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts CosineAnnealingLR(optimizer, T_max=1)
    # ExponentialLR(optimizer, gamma=0.9)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,  # Set the reference model to None as we are using LoRA
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # We define the arguments to pass to the `generate` function. These arguments are
    # passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": 150,
        "eos_token_id": terminators,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Base model sometimes generates no response, which causes an error.
    # We overwrite these empty texts with a place holder text.
    place_holder_text = "The base model failed to generate a meaningful text."
    global_setp = 0
    mean_rewards = []
    if ppo_trainer.accelerator.is_main_process:
        run = mlflow.start_run(run_name=os.environ.get("RUN_NAME", None))
    for epoch in range(config.ppo_epochs):
        if ppo_trainer.accelerator.is_main_process:
            print(f"Epoch: {epoch}")
        for step, batch in enumerate(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            # Get the response from the base modeld
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(
                    query, return_prompt=False, **generation_kwargs
                ).squeeze()
                if not response.shape:
                    response = torch.tensor(
                        tokenizer.encode(place_holder_text)
                    ).squeeze()
                response_tensors.append(response)
            # Get the score from the reward model
            batch["response"] = [
                tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
            ]
            try:
                scores = run_reward_scoring(batch["response"])
                rewards_tensors = [
                    torch.tensor(math.log(score / (1.0 - score))) for score in scores
                ]
                if ppo_trainer.accelerator.is_main_process:
                    print("Scrores: ", scores)
                    print("Reward tensors: ", rewards_tensors)
                    print(
                        "Mean reward: ",
                        torch.mean(torch.stack(rewards_tensors, dim=0)).detach().item(),
                    )
            except Exception as e:
                print(e)
                scores = [0.5] * config.batch_size
                rewards_tensors = [torch.tensor(0.0)] * config.batch_size
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensors)
            ppo_trainer.log_stats(stats, batch, rewards_tensors)
            # lr_scheduler.step(step)
            if ppo_trainer.accelerator.is_main_process and run:
                mean_reward = (
                    torch.mean(torch.stack(rewards_tensors, dim=0)).detach().item()
                )
                mlflow.log_metric("mean_reward", mean_reward, step=global_setp)
                mean_rewards.append(mean_reward)
                mlflow.log_metric(
                    "rolling_mean_reward",
                    np.mean(mean_rewards[:-10]),
                    step=global_setp,
                    run_id=run.info.run_id,
                )
                for k, v in stats.items():
                    if isinstance(v, (int, float, str, bool)):
                        mlflow.log_metric(
                            k,
                            v,
                            step=global_setp,
                            run_id=run.info.run_id,
                        )

                if step % 50 == 0:
                    save_checkpoint(ppo_trainer, run, step)
                if step % 5 == 0:
                    print(f"STEP: {step}")
                    print(f"PROMPTS: {batch['query']}")
                    print(f"GENERATED: {batch['response']}")
                    print(f"SCORED: {scores}")
            global_setp += 1
    if ppo_trainer.accelerator.is_main_process:
        save_checkpoint(ppo_trainer, run, "final")


def save_checkpoint(ppo_trainer, run, step):
    shutil.rmtree(conf.LOCAL_MODEL_PATH, ignore_errors=True)
    os.makedirs(conf.LOCAL_MODEL_PATH, exist_ok=True)
    ppo_trainer.save_pretrained(conf.LOCAL_MODEL_PATH)
    mlflow.log_artifacts(
        conf.LOCAL_MODEL_PATH,
        f"checkpoint_{step}",
        run_id=run.info.run_id,
    )


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    script_args = parse_cmd_args()
    if script_args.train:
        run_training(script_args)
    if script_args.download_dataset:
        download_dataset(script_args.dataset_path, conf.LOCAL_DATASET_PATH)
