# Retokenization Experiments

`retok_exp/` contains training and model-conversion scripts for experiments where a pretrained causal LM is adapted to a different tokenizer by replacing its embedding tables and then continuing training with most parameters frozen.

## Scope

Main workflows:

- replace a model tokenizer and initialize new embeddings
- continue training with only embeddings and `lm_head` trainable
- train from plain text shards or parquet shards

Core scripts:

- [replace_tokenizer_embedding.py](/Users/souvik/repos/tokenizer/retok_exp/replace_tokenizer_embedding.py)
- [train_model.py](/Users/souvik/repos/tokenizer/retok_exp/train_model.py)
- [train_model_with_parquet.py](/Users/souvik/repos/tokenizer/retok_exp/train_model_with_parquet.py)

Utility scripts live under [retok_exp/utils](/Users/souvik/repos/tokenizer/retok_exp/utils).

## Environment

An example conda environment is provided in [environment.yml](/Users/souvik/repos/tokenizer/retok_exp/environment.yml). It is an experiment-heavy environment file rather than a minimal OSS environment, so you may want to trim it for your own setup.

Typical dependencies used by the main scripts:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `sentencepiece`
- `tokenizers`

## 1. Replace Tokenizer Embeddings

This script loads a pretrained causal LM, loads a new tokenizer, resizes the embedding tables, and initializes new token embeddings using:

- direct copy when the token already exists in the old tokenizer
- average of old-token pieces when the token can be decomposed
- random initialization as a fallback

By default it freezes everything except input embeddings and `lm_head`.

Example:

```bash
python3 retok_exp/replace_tokenizer_embedding.py \
  --model_path /path/to/base-model \
  --new_tokenizer_path /path/to/new-tokenizer \
  --output_dir /path/to/output-model
```

To keep all layers trainable:

```bash
python3 retok_exp/replace_tokenizer_embedding.py \
  --model_path /path/to/base-model \
  --new_tokenizer_path /path/to/new-tokenizer \
  --output_dir /path/to/output-model \
  --no_freeze
```

Outputs:

- updated model weights
- updated tokenizer files
- `embedding_init_stats.json`

## 2. Continue Training From Text Files

[train_model.py](/Users/souvik/repos/tokenizer/retok_exp/train_model.py) trains from plain text files using the Hugging Face `text` dataset loader in streaming mode.

Expected input:

- one or more `.txt` files
- each line is training text

Example:

```bash
accelerate launch retok_exp/train_model.py \
  --model_name_or_path /path/to/adapted-model \
  --train_files "data/*.txt" \
  --output_dir /path/to/checkpoints \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --max_steps 200000 \
  --save_steps 2000 \
  --fp16
```

Behavior:

- loads tokenizer from `model_name_or_path`
- resizes embeddings to tokenizer length
- freezes all layers except input embeddings and `lm_head`
- tokenizes to fixed-length sequences

## 3. Continue Training From Parquet Files

[train_model_with_parquet.py](/Users/souvik/repos/tokenizer/retok_exp/train_model_with_parquet.py) trains from parquet shards with a required `text` column.

Expected input:

- a directory of `*.parquet`
- each parquet file contains a `text` column

Example:

```bash
accelerate launch retok_exp/train_model_with_parquet.py \
  --model_name_or_path /path/to/adapted-model \
  --train_dir /path/to/parquet-dir \
  --output_dir /path/to/checkpoints \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100000 \
  --save_steps 2000 \
  --bf16
```

Optional Weights & Biases reporting:

```bash
accelerate launch retok_exp/train_model_with_parquet.py \
  --model_name_or_path /path/to/adapted-model \
  --train_dir /path/to/parquet-dir \
  --output_dir /path/to/checkpoints \
  --wandb_project retok-exp \
  --wandb_run_name trial-01
```

## Utility Scripts

- `utils/shuffle_data.py`: globally shuffle parquet shards and re-emit them as new parquet shards
- `utils/check_tokenizer_overlap.py`: compare vocabulary overlap between two tokenizers
- `utils/model_param_analysis.py`: compare generations between an adapted model and a reference model

Examples:

```bash
python3 retok_exp/utils/shuffle_data.py \
  --input_dir /path/to/parquet-in \
  --output_dir /path/to/parquet-out
```

```bash
python3 retok_exp/utils/check_tokenizer_overlap.py \
  --tokenizer_a meta-llama/Llama-3.2-1B \
  --tokenizer_b /path/to/new-tokenizer
```

```bash
python3 retok_exp/utils/model_param_analysis.py \
  --model /path/to/adapted-model \
  --reference_model meta-llama/Llama-3.2-1B
```

## OSS Notes

What was cleaned up for OSS readiness:

- removed hardcoded private filesystem paths from runnable scripts
- removed internal command examples from source files
- removed unnecessary sample logging side effects during training
- made utility scripts parameter-driven instead of environment-specific

Residual risks:

- [environment.yml](/Users/souvik/repos/tokenizer/retok_exp/environment.yml) is still a large, experiment-oriented environment file rather than a minimal reproducible one
- no automated tests are included for the training scripts
- actual runtime behavior still depends on model size, dataset size, and distributed-training setup
