import os
import json
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def replace_tokenizer(
    model_path: str,
    new_tokenizer_path: str,
    output_dir: str,
    freeze_non_embedding: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load model + original tokenizer
    print("Loading model:", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    old_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Load new tokenizer
    print("Loading new tokenizer:", new_tokenizer_path)
    new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path, use_fast=True)

    old_emb = model.get_input_embeddings()
    old_vocab_size, hidden_dim = old_emb.weight.shape

    new_vocab_size = len(new_tokenizer)
    print(f"Old vocab: {old_vocab_size}, New vocab: {new_vocab_size}")

    # Create new embedding and lm_head layers
    new_emb = nn.Embedding(new_vocab_size, hidden_dim)
    new_lm_head = nn.Linear(hidden_dim, new_vocab_size, bias=False)

    # Tie embeddings
    new_lm_head.weight = new_emb.weight

    # Stats
    stats = {
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "copied_tokens": [],
        "avg_initialized_tokens": [],
        "random_initialized_tokens": [],
    }

    piece_counts = []

    # Build mapping from new_tokenizer to embeddings
    for new_id in tqdm(range(new_vocab_size)):
        token = new_tokenizer.convert_ids_to_tokens(new_id)

        if token in old_tokenizer.get_vocab():
            old_id = old_tokenizer.convert_tokens_to_ids(token)
            new_emb.weight.data[new_id] = old_emb.weight.data[old_id]
            stats["copied_tokens"].append(token)

        else:
            # Tokenize with old tokenizer
            pieces = old_tokenizer.tokenize(token)
            if pieces:
                piece_ids = old_tokenizer.convert_tokens_to_ids(pieces)
                piece_embs = old_emb.weight.data[piece_ids]
                avg_emb = piece_embs.mean(dim=0)
                new_emb.weight.data[new_id] = avg_emb
                stats["avg_initialized_tokens"].append({"token": token, "pieces": pieces})
                piece_counts.append(len(pieces))
            else:
                # Random init fallback
                new_emb.weight.data[new_id] = torch.randn(hidden_dim) * 0.02
                stats["random_initialized_tokens"].append(token)

    # Replace model embeddings + lm_head
    model.set_input_embeddings(new_emb)
    model.lm_head = new_lm_head

    # Optionally freeze all except embeddings
    if freeze_non_embedding:
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        print("All layers frozen except embeddings + lm_head")

    # Compute summary stats
    n_total = new_vocab_size
    n_copied = len(stats["copied_tokens"])
    n_avg = len(stats["avg_initialized_tokens"])
    n_random = len(stats["random_initialized_tokens"])
    avg_piece_count = float(sum(piece_counts) / len(piece_counts)) if piece_counts else 0.0

    stats["summary"] = {
        "n_total": n_total,
        "n_copied": n_copied,
        "n_avg_initialized": n_avg,
        "n_random_initialized": n_random,
        "pct_copied": round(100 * n_copied / n_total, 2),
        "pct_avg_initialized": round(100 * n_avg / n_total, 2),
        "pct_random_initialized": round(100 * n_random / n_total, 2),
        "avg_piece_count": avg_piece_count,
    }

    # Save model, tokenizer, and stats
    model.save_pretrained(output_dir)
    new_tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "embedding_init_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Model + tokenizer saved at", output_dir)
    print("Summary Stats:", stats["summary"])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replace tokenizer in pretrained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model (hub id or local dir)")
    parser.add_argument("--new_tokenizer_path", type=str, required=True, help="Path to new tokenizer (HF or local)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save updated model")
    parser.add_argument("--no_freeze", action="store_true", help="Do not freeze other layers")

    args = parser.parse_args()

    replace_tokenizer(
        model_path=args.model_path,
        new_tokenizer_path=args.new_tokenizer_path,
        output_dir=args.output_dir,
        freeze_non_embedding=not args.no_freeze,
    )
