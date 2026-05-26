"""Task 1: Baseline evaluation on unedited model (full format)."""

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import save_json

# ==================== 配置 ====================
MODEL_PATH = "/home/zkjin/marshan/badjudge/LLM_Security/03_editing_knowledge/Qwen2.5-0.5B-Instruct"
DATA_PATH = "/home/zkjin/marshan/badjudge/LLM_Security/03_editing_knowledge/data/custom_10_facts.json"
OUTPUT_PATH = "/home/zkjin/marshan/badjudge/LLM_Security/03_editing_knowledge/results/baseline.json"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 15  
TEMPERATURE = 0.0       
USE_CHAT_TEMPLATE = False 

def generate_response(prompt: str, tokenizer, model) -> str:
    """Generate text for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()

def check_match(target: str, generation: str) -> bool:

    t = target.lower()
    g = generation.lower()
    if t in g:
        return True
    
    words = t.split()
    if len(words) > 1 and words[-1] in g:
        return True
    return False

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    
    print("Loading complete. Running baseline evaluation...\n")

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        records = json.load(f)

    results = []
    start_time = time.perf_counter()

    for idx, rec in enumerate(records, start=1):
        prompt = rec["prompt"]
        target_new = rec["target_new"]
        ground_truth = rec["ground_truth"]
        rephrase_prompt = rec["rephrase_prompt"]
        locality_prompt = rec["locality_prompt"]
        locality_gt = rec["locality_ground_truth"]

        
        gen = generate_response(prompt, tokenizer, model)
        contains_target = check_match(target_new, gen)
        contains_truth = check_match(ground_truth, gen)

        rephrase_gen = generate_response(rephrase_prompt, tokenizer, model)
        rephrase_contains_target = check_match(target_new, rephrase_gen)

        locality_gen = generate_response(locality_prompt, tokenizer, model)
        locality_contains_truth = check_match(locality_gt, locality_gen)

        results.append({
            "prompt": prompt,
            "target_new": target_new,
            "ground_truth": ground_truth,
            "generation": gen,
            "contains_target_new": contains_target,
            "contains_ground_truth": contains_truth,
            "rephrase_prompt": rephrase_prompt,
            "rephrase_generation": rephrase_gen,
            "rephrase_contains_target_new": rephrase_contains_target,
            "locality_prompt": locality_prompt,
            "locality_ground_truth": locality_gt,
            "locality_generation": locality_gen,
            "locality_contains_ground_truth": locality_contains_truth
        })

       
        print(f"[{idx}/{len(records)}] {prompt}")
        print(f"  Generation: {gen}")
        print(f"  Contains target_new '{target_new}': {contains_target}")
        print(f"  Contains ground_truth '{ground_truth}': {contains_truth}")

    elapsed = round(time.perf_counter() - start_time, 3)

    target_new_hits = sum(r["contains_target_new"] for r in results)
    ground_truth_hits = sum(r["contains_ground_truth"] for r in results)
    rephrase_target_hits = sum(r["rephrase_contains_target_new"] for r in results)
    locality_truth_hits = sum(r["locality_contains_ground_truth"] for r in results)

    output_payload = {
        "metadata": {
            "model": MODEL_PATH,
            "data": DATA_PATH,
            "num_records": len(records),
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "chat_template": USE_CHAT_TEMPLATE,
            "elapsed_seconds": elapsed
        },
        "results": results,
        "summary": {
            "target_new_hits": target_new_hits,
            "ground_truth_hits": ground_truth_hits,
            "rephrase_target_new_hits": rephrase_target_hits,
            "locality_ground_truth_hits": locality_truth_hits
        }
    }

    save_json(OUTPUT_PATH, output_payload)
    
    print(f"\nBaseline results saved to results/baseline.json")
    print(f"Ground truth hit rate: {ground_truth_hits}/{len(records)} = {ground_truth_hits/len(records)*100:.1f}%")

if __name__ == "__main__":
    main()