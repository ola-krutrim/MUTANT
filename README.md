# Tokenizer Research Toolkit


## Reference

This repository contains the open-source code referenced in our research paper:

- [MUTANT: A Recipe for Multilingual Tokenizer Design](https://arxiv.org/abs/2511.03237)

If you are looking for the code and experiments described in the paper, you are in the right place. The evaluator, visualizer, and experimental scripts are all included here.


- `tokenizer_evaluator/`: evaluate tokenizer fidelity, efficiency, NSL, and Renyi-style token distribution metrics across multiple tokenizer backends
- `tokenizer_visualizer/`: inspect token boundaries in a small browser UI with color-coded token output
- `retok_exp/`: retokenization experiments for replacing tokenizer embeddings in pretrained causal language models and continuing training

The codebase is not a single monolithic package. It is a repository of adjacent tools that share a tokenizer experimentation theme.

## Repository Layout

### `tokenizer_evaluator`

CLI and Python package for tokenizer benchmarking.

Supports:

- Hugging Face tokenizers
- SentencePiece models
- `tokenizer.json` artifacts through `PreTrainedTokenizerFast`
- tiktoken-style approximation for Hugging Face tokenizers

Start here:

- `tokenizer_evaluator/README.md`

Typical install:

```bash
pip install -e ./tokenizer_evaluator
```

Typical run:

```bash
tok-eval --tokenizer_paths /path/to/tokenizer --tokenizer_type hf --data_dir ./data --out_dir ./results
```

### `tokenizer_visualizer`

Small Flask-based UI for visual inspection of token segmentation.

Supports:

- configured Hugging Face tokenizers
- configured local `tokenizer.json` tokenizers
- configured SentencePiece models

Start here:

- `tokenizer_visualizer/README.md`

Typical run:

```bash
python3 tokenizer_visualizer/server/server.py
```

### `retok_exp`

Experimental scripts for retokenization workflows:

- replace a pretrained model tokenizer and initialize new embeddings
- continue training with embeddings and `lm_head` unfrozen
- train from text or parquet corpora
- inspect tokenizer overlap and adapted-model behavior

Start here:

- `retok_exp/README.md`

## Getting Started

Choose the subproject that matches your goal:

1. Evaluate tokenizer quality: use `tokenizer_evaluator/`
2. Inspect token boundaries visually: use `tokenizer_visualizer/`
3. Run embedding-replacement or retokenization experiments: use `retok_exp/`

The subprojects have different dependency footprints, so install only what you need for the workflow you plan to run.

## Notes

- Some directories are packaged as standalone tools, while others are plain scripts.
- The experimental code in `retok_exp/` is more research-oriented than the evaluator and visualizer.
- Hardware requirements depend heavily on the workflow. `retok_exp/` in particular may require multi-GPU training infrastructure for meaningful runs.
- Repository-level cleanup for open source is focused on removing private paths and documenting how to configure each tool locally; it is not a promise of production-hardening across every experiment script.

## Status

This repository is best understood as an open research toolkit:

- `tokenizer_evaluator/` is the most structured and reusable component
- `tokenizer_visualizer/` is a lightweight local tool
- `retok_exp/` contains experimental training workflows with stronger infrastructure assumptions

If you are new to the repo, read the subproject README for the directory you want to use instead of trying to infer usage from source files alone.


### Citation

If you use this toolkit or its results in your research, please cite:

```bibtex
@inproceedings{rana2026mutant,
	title={MUTANT: A Recipe for Multilingual Tokenizer Design}, 
	author={Souvik Rana and Arul Menezes and Ashish Kulkarni and Chandra Khatri and Shubham Agarwal},
	year={2026},
	booktitle={ACL}
	url={https://arxiv.org/abs/2511.03237}, 
}
```
### Acknowledgement
Our work is built with reference to the code of the following projects: [Tokenizers](https://github.com/huggingface/tokenizers),[SuperBPE](https://github.com/PythonNut/superbpe), [TikToken](https://github.com/openai/tiktoken), [SentencePiece](https://github.com/google/sentencepiece) and [Transformers](https://github.com/huggingface/transformers). Thanks for their awesome work!
