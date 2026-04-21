"""
Pretraining script using Hugging Face `Trainer` API for LLaMA-style models.

We freeze all parameters EXCEPT embeddings + lm_head.
Supports a directory of parquet files with column `text`.
Optionally reports metrics to Weights & Biases.

Usage example:

accelerate launch train_model_with_parquet.py \
    --model_name_or_path /path/to/llama-3.21b \
    --train_dir /path/to/data \
    --output_dir /path/to/out \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_steps 200000 \
    --save_steps 2000 \
    --bf16 \
    --wandb_project llama-pretrain \
    --wandb_run_name retok-exp
"""

import argparse
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def freeze_all_but_embeddings_and_lm_head(model: AutoModelForCausalLM):
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Input embeddings
    input_emb = None
    try:
        input_emb = model.get_input_embeddings()
    except Exception:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            input_emb = model.model.embed_tokens

    if input_emb is None:
        raise RuntimeError("Couldn't find input embeddings on the model.")

    for p in input_emb.parameters():
        p.requires_grad = True

    # LM head / output embeddings
    output_emb = None
    try:
        output_emb = model.get_output_embeddings()
    except Exception:
        if hasattr(model, "lm_head"):
            output_emb = model.lm_head

    if output_emb is not None:
        for p in output_emb.parameters():
            p.requires_grad = True


def get_trainable_params(model: AutoModelForCausalLM) -> List[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in params)}")
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with parquet/arrow files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all but embeddings + lm head
    freeze_all_but_embeddings_and_lm_head(model)
    get_trainable_params(model)

    # Load dataset from dir (arrow/parquet)
    dataset = load_dataset("parquet", data_files=f"{args.train_dir}/*.parquet", split="train", streaming=True)

    # Shuffle dataset with a buffer
    dataset = dataset.shuffle(buffer_size=10_000, seed=42)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.seq_length,
            return_special_tokens_mask=False,
        )
    

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=-1,
        logging_steps=10,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=2,
        report_to="wandb" if args.wandb_project else "none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        run_name=args.wandb_run_name,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
