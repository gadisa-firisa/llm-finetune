import argparse
import os
import sys
import tomllib
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dataset import (
    load_dailydialog_dialogue,
    load_samsum_summarization,
    load_banking77_intent,
)


task_mapping = {
    "dialogue": load_dailydialog_dialogue,
    "summarization": load_samsum_summarization,
    "intent": load_banking77_intent,
}

def to_hf_dataset(records):
    return Dataset.from_list(records)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(task_mapping.keys()))
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "pyproject.toml"))
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    root = cfg.get("tool", {}).get("llm_finetune", {})
    lora_cfg = root.get("lora", {})

    base = root.get("base", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = root.get("output_dir", "llm-finetune/models/{task}/lora").format(task=args.task)

    ta_root = dict(root.get("training_arguments", {}))
    ta_task = dict(root.get("tasks", {}).get(args.task, {}).get("training_arguments", {}))
    training_args = {**ta_root, **ta_task}
    sample_val = int(training_args.pop("sample", 0) or 0)
    sample = None if sample_val in (0, -1) else sample_val

    if "output_dir" not in training_args:
        training_args["output_dir"] = output_dir
    if "max_seq_length" not in training_args:
        training_args["max_seq_length"] = 1536

    loader = task_mapping[args.task]
    train = to_hf_dataset(loader("train", sample=sample))
    eval_sample = min(1000, sample) if sample else 1000
    evald = to_hf_dataset(loader("test", sample=eval_sample))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    def format_example(ex):
        messages = ex["messages"] + [{"role": "assistant", "content": ex["target"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train = train.map(format_example, remove_columns=train.column_names)
    evald = evald.map(format_example, remove_columns=evald.column_names)

    cfg = SFTConfig(**training_args)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=evald,
        args=cfg,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
