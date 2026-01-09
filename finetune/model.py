import argparse
import os
import sys
import tomllib
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dataset import (
    load_dailydialog_dialogue,
    load_samsum_summarization,
    load_banking77_intent,
)
import json
from dotenv import load_dotenv
load_dotenv() 

os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", '')
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", '')

task_mapping = {
    "dialogue": load_dailydialog_dialogue,
    "summarization": load_samsum_summarization,
    "intent": load_banking77_intent,
}

def to_hf_dataset(records):
    return Dataset.from_list(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(task_mapping.keys()))
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "pyproject.toml"))
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    root = cfg.get("tool", {}).get("llm_finetune", {})
    lora_cfg = root.get("lora", {})

    model_id = root.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = root.get("output_dir", "../models/{task}/lora").format(task=args.task)

    ta_root = dict(root.get("training_arguments", {}))
    ta_task = dict(root.get("tasks", {}).get(args.task, {}).get("training_arguments", {}))
    training_args = {**ta_root, **ta_task}
    sample_val = int(training_args.pop("sample", 0) or 0)
    sample = None if sample_val in (0, -1) else sample_val

    if "output_dir" not in training_args:
        training_args["output_dir"] = output_dir
    if "max_seq_length" not in training_args:
        training_args["max_seq_length"] = 1536

    if os.getenv("WANDB_PROJECT") != '' and os.getenv("WANDB_API_KEY") != '':
        training_args["report_to"] = "wandb"

    loader = task_mapping[args.task]
    train = to_hf_dataset(loader("train", sample=sample))
    eval_sample = min(1000, sample) if sample else 1000
    evalset = to_hf_dataset(loader("test", sample=eval_sample))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb, 
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        )
    
    if training_args.get("gradient_checkpointing", False):
        model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        task_type="CAUSAL_LM",
    )
    
    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train = train.map(format_example, remove_columns=train.column_names)
    evalset = evalset.map(format_example, remove_columns=evalset.column_names)

    cfg = SFTConfig(**training_args)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=evalset,
        args=cfg,
        dataset_text_field="text",
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()

    save_path = training_args["output_dir"]
    os.makedirs(save_path, exist_ok=True)

    print(f"Starting training for {args.task}...")
    with open (f"{save_path}/training_config.json", 'w', encoding='utf-8') as file:
        json.dump(training_args, file, indent=4)

    lora_config = {
        "r": lora.r,
        "lora_alpha": lora.lora_alpha,
        "lora_dropout": lora.lora_dropout,
        "target_modules": lora.target_modules,
        "task_type": lora.task_type,
    }

    with open (f"{save_path}/lora_config.json", 'w', encoding='utf-8') as file:
        json.dump(lora_config, file, indent=4)

    trainer.train()
    
    final_checkpoint_dir = f"{save_path}/final_checkpoint"
    print(f"Saving final model to {final_checkpoint_dir}")
    trainer.save_model(final_checkpoint_dir)


if __name__ == "__main__":
    main()
