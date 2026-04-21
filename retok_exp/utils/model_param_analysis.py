import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            status = "TRAINABLE"
        else:
            status = "FROZEN"
        print(f"{name:60} | {status}")

    print("="*80)
    print(f"Total parameters     : {total_params/1e9:.3f}B")
    print(f"Trainable parameters : {trainable_params}")
    print(f"Frozen parameters    : {(total_params-trainable_params)/1e9:.3f}B")
    print(f"Trainable %          : {100*trainable_params/total_params:.2f}%")


def generate_response(model,tokenizer, prompt, max_new_tokens=20, device="cuda"):

    conv = [{"role": "system", "content": "You are an AI assistant."},{"role": "user", "content": prompt}]

    model.to(device)
    model.eval()

    # templated_prompt = tokenizer.apply_chat_template(conv)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # remove 'token_type_ids' if present
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("-"*80)
    print("Prompt:")
    print(prompt)
    print("-"*80)
    print("Response:")
    print(decoded[len(prompt):].strip())
    print("-"*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to adapted model.")
    parser.add_argument(
        "--reference_model",
        type=str,
        required=True,
        help="Path or model id for the reference/original model used for comparison.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Given the following question. Answer using only one word.\n\nQuestion: What is the name of the largest planet in our solar system?\n\nAnswer: ",
        help="Prompt used for qualitative comparison.",
    )
    args = parser.parse_args()

    model_replaced = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer_replaced = AutoTokenizer.from_pretrained(args.model)

    model_original = AutoModelForCausalLM.from_pretrained(args.reference_model, torch_dtype=torch.float16, device_map="auto")
    tokenizer_original = AutoTokenizer.from_pretrained(args.reference_model)

    generate_response(model_replaced, tokenizer_replaced, args.prompt)
    generate_response(model_original, tokenizer_original, args.prompt)
