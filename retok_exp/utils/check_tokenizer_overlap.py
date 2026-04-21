import argparse
import os

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from transformers import AutoTokenizer

def analyze_tokenizer_overlap(tokenizer1_name, tokenizer2_name):
    """
    Analyzes and visualizes the vocabulary overlap between two Hugging Face tokenizers.

    Args:
        tokenizer1_name (str): The name of the first tokenizer model (e.g., 'bert-base-uncased').
        tokenizer2_name (str): The name of the second tokenizer model (e.g., 'gpt2').
    """
    print(f"Loading tokenizer: {tokenizer1_name}...")
    tokenizer1 = AutoTokenizer.from_pretrained(tokenizer1_name)
    
    print(f"Loading tokenizer: {tokenizer2_name}...")
    tokenizer2 = AutoTokenizer.from_pretrained(tokenizer2_name)

    # --- Vocabulary Extraction ---
    # Get the vocabulary (tokens) for each tokenizer as a set for efficient comparison.
    vocab1 = set(tokenizer1.get_vocab().keys())
    vocab2 = set(tokenizer2.get_vocab().keys())

    print(f"\nVocabulary size for '{tokenizer1_name}': {len(vocab1)}")
    print(f"Vocabulary size for '{tokenizer2_name}': {len(vocab2)}")

    # --- Overlap Calculation ---
    # Find the intersection (common tokens) and unique tokens for each.
    common_tokens = vocab1.intersection(vocab2)
    unique_to_1 = vocab1 - vocab2
    unique_to_2 = vocab2 - vocab1
    total_unique_tokens = len(vocab1.union(vocab2))

    # Calculate overlap percentage and Jaccard similarity
    overlap_percentage = (len(common_tokens) / total_unique_tokens) * 100
    jaccard_similarity = len(common_tokens) / total_unique_tokens

    # --- Print Analysis Results ---
    print("\n--- Overlap Analysis ---")
    print(f"Number of common tokens: {len(common_tokens)}")
    print(f"Number of tokens unique to '{tokenizer1_name}': {len(unique_to_1)}")
    print(f"Number of tokens unique to '{tokenizer2_name}': {len(unique_to_2)}")
    print(f"Total unique tokens in combined vocabulary: {total_unique_tokens}")
    print(f"Overlap Percentage: {overlap_percentage:.2f}%")
    print(f"Jaccard Similarity: {jaccard_similarity:.4f}")

    # Display a few common and unique tokens as examples
    print("\n--- Sample Tokens ---")
    print(f"Sample of common tokens: {list(common_tokens)[:10]}")
    print(f"Sample of tokens unique to '{tokenizer1_name}': {list(unique_to_1)[:10]}")
    print(f"Sample of tokens unique to '{tokenizer2_name}': {list(unique_to_2)[:10]}")

    # --- Visualization ---
    # Create a figure to hold the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Vocabulary Overlap Analysis: '{tokenizer1_name.split('/')[-1]}' vs '{tokenizer2_name.split('/')[-1]}'", fontsize=18)

    # 1. Venn Diagram
    venn2(
        subsets=(len(unique_to_1), len(unique_to_2), len(common_tokens)),
        set_labels=(tokenizer1_name.split('/')[-1], tokenizer2_name.split('/')[-1]),
        ax=ax1,
        set_colors=('#4A90E2', '#F5A623'),
        alpha=0.7
    )
    venn2_circles(
        subsets=(len(unique_to_1), len(unique_to_2), len(common_tokens)),
        linestyle='--',
        linewidth=2,
        color='grey',
        ax=ax1
    )
    ax1.set_title("Venn Diagram of Vocabulary Overlap", fontsize=14)

    # 2. Bar Chart
    labels = [tokenizer1_name.split('/')[-1], tokenizer2_name.split('/')[-1], 'Common Tokens']
    sizes = [len(vocab1), len(vocab2), len(common_tokens)]
    colors = ['#4A90E2', '#F5A623', '#7ED321']
    
    bars = ax2.bar(labels, sizes, color=colors)
    ax2.set_ylabel('Number of Tokens', fontsize=12)
    ax2.set_title('Vocabulary Size Comparison', fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:,}', va='bottom', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/Vocabulary_Overlap_Analysis_{tokenizer1_name.split('/')[-1]}_vs_{tokenizer2_name.split('/')[-1]}.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze vocabulary overlap between two tokenizers.")
    parser.add_argument("--tokenizer_a", required=True)
    parser.add_argument("--tokenizer_b", required=True)
    args = parser.parse_args()

    analyze_tokenizer_overlap(args.tokenizer_a, args.tokenizer_b)
