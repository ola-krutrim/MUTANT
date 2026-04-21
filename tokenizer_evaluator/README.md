# Tokenizer Evaluator

`tokenizer_evaluator` is the public evaluation subpackage in this repository for comparing tokenizer behavior across datasets. It supports Hugging Face tokenizers, SentencePiece models, `tokenizer.json` artifacts loaded through `PreTrainedTokenizerFast`, and tiktoken-compatible encodings derived from Hugging Face tokenizer assets.

## What It Measures

- Reconstruction fidelity with exact-match and fuzzy-match scores
- Token efficiency with token-to-word ratios
- SentencePiece LCS distance
- Optional NSL scoring against a base Hugging Face tokenizer for every backend
- Rényi entropy and Rényi efficiency for every tokenizer backend
- Byte fallback frequency for `ptf` tokenizers
- Throughput snapshots for `hf` and `tkt` evaluations

## Installation

From the repository root:

```bash
pip install -e ./tokenizer_evaluator
```

## Usage

The evaluator supports two input modes:

- Local `.txt` files under a directory tree
- Hugging Face datasets with language-specific configs/subsets

In both modes you provide one or more tokenizer paths and one tokenizer backend type.

## Tokenizer Backends

Supported tokenizer types:

- `hf`: `transformers.AutoTokenizer`
- `spm`: SentencePiece model files
- `ptf`: local `tokenizer.json` or directory containing `tokenizer.json`
- `tkt`: tiktoken encoding built from local or Hub tokenizer assets

`tkt` should be treated as an approximation layer for evaluating Hugging Face tokenizers through a tiktoken-style encoding. It is useful for comparative benchmarking, but it is not guaranteed to reproduce every tokenizer family exactly.

## Local Text Files

Use `--data_source files` or omit it, since `files` is the default.

### Basic Example

```bash
tok-eval \
  --tokenizer_paths ./path/to/tokenizer \
  --tokenizer_type hf \
  --data_source files \
  --data_dir ./data \
  --out_dir ./results
```

The evaluator scans `--data_dir` recursively for `.txt` files. Each non-empty line is treated as one evaluation sample.

Example layout:

```text
data/
  eng.txt
  hi/news.txt
  code/python.txt
```

### Restrict To One Language

```bash
tok-eval \
  --tokenizer_paths ./path/to/tokenizer \
  --tokenizer_type hf \
  --data_dir ./data \
  --out_dir ./results \
  --lang eng
```

If `--lang` is omitted, all discovered `.txt` files are evaluated.

Language matching is exact against path and filename tokens for the supported keys:
`as`, `bn`, `brx`, `code`, `doi`, `eng`, `gom`, `gu`, `hi`, `kn`, `mai`, `ml`, `mr`, `nep`, `or`, `pa`, `san`, `sat`, `snd`, `ta`, `te`, `urd`.

## Hugging Face Datasets

Use `--data_source hf` when your dataset lives on the Hugging Face Hub.

This mode is intended for datasets where:

- each language is a dataset config/subset such as `eng`, `hi`, `ta`, etc.
- the evaluation rows are in a split such as `test`
- one or more string columns contain the text samples to evaluate

### Basic Example

```bash
tok-eval \
  --tokenizer_paths ./path/to/tokenizer \
  --tokenizer_type hf \
  --data_source hf \
  --hf_dataset your-org/your-dataset \
  --hf_split test \
  --out_dir ./results
```

### Evaluate One Language Config

```bash
tok-eval \
  --tokenizer_paths ./path/to/tokenizer \
  --tokenizer_type hf \
  --data_source hf \
  --hf_dataset krutrim-ai-labs/MUTANT \
  --hf_split test \
  --lang hi \
  --out_dir ./results
```

### Evaluate All Supported Language Configs

```bash
tok-eval \
  --tokenizer_paths ./path/to/tokenizer \
  --tokenizer_type hf \
  --data_source hf \
  --hf_dataset krutrim-ai-labs/MUTANT \
  --hf_split test \
  --out_dir ./results
```

For Hugging Face dataset mode:

- If `--lang` is set, only that dataset config is loaded.
- If `--lang` is omitted, all dataset configs matching the supported language codes are loaded.
- If `--hf_text_columns` is omitted, the evaluator uses the `text` column by default.
- Each evaluated text column becomes its own output file.

If your dataset does not use `text` as the column name, pass `--hf_text_columns` explicitly.

### NSL

NSL is available for all tokenizer backends. It compares the evaluated tokenizer length against a base Hugging Face tokenizer.

```bash
tok-eval \
  --tokenizer_paths ./my_tokenizer_dir \
  --tokenizer_type hf \
  --data_dir ./data \
  --out_dir ./results \
  --enable-nsl \
  --nsl-base-tokenizer meta-llama/Llama-3.1-8B
```

### Gated or Custom Hugging Face Models

```bash
tok-eval \
  --tokenizer_paths my-org/private-tokenizer \
  --tokenizer_type hf \
  --hf_token $HF_TOKEN \
  --trust-remote-code
```

## Outputs

For each tokenizer, results are written under:

```text
<out_dir>/<tokenizer_name>/
```

Per evaluated input stream, the evaluator writes:

- `<file>_results.csv`
- `<file>_results.json`

For local files, the output stem usually comes from the input filename.

For Hugging Face datasets, the output stem is derived from:

- dataset config name, for example `eng`
- text column name, for example `text`

So a dataset config `eng` with column `text` will produce files similar to:

```text
eng_text_results.csv
eng_text_results.json
```

Additional backend-specific logs:

- `hf_tokens_stats.jsonl`
- `tkt_tokenizer_stats.jsonl`
- `reyni_logging_info.json` for every tokenizer backend

## Development

Run tests from the subpackage directory:

```bash
pytest
```

## Notes

- `--trust-remote-code` is off by default.
- NSL is opt-in and requires `--nsl-base-tokenizer`.
- `--data_source files` is the default.
- The package ships as a subpackage of this monorepo, so repository-level metadata may still live outside this directory.
