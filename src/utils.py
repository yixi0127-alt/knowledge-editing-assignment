"""Task 1: Baseline evaluation on the original (unedited) language model.
This script computes the model's raw predictions on custom facts before any editing.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_custom_records, save_json, resolve_path

# -------- 配置文件路径 ----------
BASE_MODEL_DIR = "/home/zkjin/marshan/badjudge/LLM_Security/03_editing_knowledge/Qwen2.5-0.5B-Instruct"
DATA_JSON = "/home/zkjin/marshan/badjudge/LLM_Security/03_editing_knowledge/data/custom_10_facts.json"
OUTPUT_JSON = "/home/zkjin/marshan/badjudge/LLM_Security/03_editing_knowledge/results/baseline.json"
DEVICE_ID = "cuda:0"
MAX_GEN_TOKENS = 15

def get_model_prediction(user_prompt, language_model, tokenizer):
    """Generate a short answer for a given prompt using greedy decoding."""
    input_ids = tokenizer(user_prompt, return_tensors="pt").to(DEVICE_ID)
    with torch.no_grad():
        output_ids = language_model.generate(
            **input_ids,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output_ids[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return decoded

def main():
    print("="*60)
    print("Initializing baseline model ...")
    start_time = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE_ID)
    model.eval()
    print(f"Model loaded from {BASE_MODEL_DIR} in {time.perf_counter() - start_time:.2f} seconds.\n")

    records = load_custom_records(DATA_JSON)
    results = []
    correct_old = 0
    for idx, rec in enumerate(records, start=1):
        question = rec["prompt"]
        correct_answer = rec["ground_truth"]
        new_answer = rec["target_new"]
        prediction = get_model_prediction(question, model, tokenizer)
        contains_correct = correct_answer.lower() in prediction.lower()
        contains_new = new_answer.lower() in prediction.lower()
        if contains_correct:
            correct_old += 1
        
        results.append({
            "prompt": question,
            "generated_text": prediction,
            "ground_truth": correct_answer,
            "target_new": new_answer,
            "ground_truth_matched": contains_correct,
            "target_new_matched": contains_new
        })
        print(f"[{idx:2d}/{len(records)}] Query: {question}")
        print(f"   -> Model output: {prediction}")
        print(f"   -> Contains correct answer '{correct_answer}': {contains_correct}")
        print(f"   -> Contains target new '{new_answer}': {contains_new}\n")
    
    save_json(OUTPUT_JSON, results)
    hit_ratio = correct_old / len(records) * 100
    print(f"Baseline evaluation finished. Results saved to {resolve_path(OUTPUT_JSON)}")
    print(f"Baseline accuracy (ground truth hit rate): {hit_ratio:.1f}% ({correct_old}/{len(records)})")
    print("="*60)

if __name__ == "__main__":
    main()