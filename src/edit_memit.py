"""MEMIT algorithm for batch knowledge editing (Task 3)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

def _string_contains(gen: str, target: str) -> bool:
    """Case-insensitive substring match."""
    return target.lower() in gen.lower() if gen and target else False

def _eval_single_record(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    *,
    use_chat_template: bool,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    direct_prompt = record["prompt"]
    direct_gen = _generate_text(direct_prompt, tokenizer, model, use_chat_template, max_new_tokens, temperature)
    rephrase_prompt = record["rephrase_prompt"]
    rephrase_gen = _generate_text(rephrase_prompt, tokenizer, model, use_chat_template, max_new_tokens, temperature)
    locality_prompt = record["locality_prompt"]
    locality_gen = _generate_text(locality_prompt, tokenizer, model, use_chat_template, max_new_tokens, temperature)
    return {
        "prompt": direct_prompt,
        "target_new": record["target_new"],
        "ground_truth": record["ground_truth"],
        "generation": direct_gen,
        "contains_target_new": _string_contains(direct_gen, record["target_new"]),
        "contains_ground_truth": _string_contains(direct_gen, record["ground_truth"]),
        "rephrase_prompt": rephrase_prompt,
        "rephrase_generation": rephrase_gen,
        "rephrase_contains_target_new": _string_contains(rephrase_gen, record["target_new"]),
        "locality_prompt": locality_prompt,
        "locality_ground_truth": record["locality_ground_truth"],
        "locality_generation": locality_gen,
        "locality_contains_ground_truth": _string_contains(locality_gen, record["locality_ground_truth"]),
    }

def _generate_text(
    prompt: str,
    tokenizer: Any,
    model: Any,
    use_chat_template: bool,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0, inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

from editing_utils import (
    cleanup_cuda,
    get_editor_tokenizer,
    load_custom_records,
    maybe_cuda_finish,
    maybe_cuda_start,
    resolve_path,
    save_json,
    to_jsonable,
    write_runtime_hparams,
)

def get_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply MEMIT batch edits to custom facts.")
    parser.add_argument("--data", default="data/zsre_500.json", help="Path to 500 batch facts dataset.")
    parser.add_argument("--eval-data", default="data/custom_10_facts.json", help="Path to 10 evaluation facts.")
    parser.add_argument("--output", default="results/memit_results.json", help="Output JSON file.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model identifier.")
    parser.add_argument("--config", default="configs/memit.yaml", help="MEMIT hyperparameter YAML.")
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'.")
    parser.add_argument("--limit", type=int, default=500, help="Max number of facts to edit.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation length limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy).")
    return parser.parse_args()

def init_memit_engine(config_path: str, model_name: str, device_id: str) -> Any:
    try:
        from easyeditor import BaseEditor, MEMITHyperParams
    except ImportError as exc:
        raise SystemExit("EasyEdit not found. Install via: pip install -r requirements.txt") from exc
    runtime_cfg = write_runtime_hparams(config_path, model_name=model_name, device=device_id)
    hps = MEMITHyperParams.from_hparams(str(runtime_cfg))
    return BaseEditor.from_hparams(hps)

def extract_edit_metrics(metrics: Any) -> dict[str, Any]:
    metrics = to_jsonable(metrics)
    if isinstance(metrics, list) and metrics:
        first = metrics[0]
        return first if isinstance(first, dict) else {"raw": first}
    if isinstance(metrics, dict):
        return metrics
    return {"raw": metrics}

def main() -> None:
    args = get_cli_arguments()
    
    print(f"Running MEMIT batch edit on {args.limit} samples...")
    print("Loading model and hyperparameters...")
    
    editor = init_memit_engine(args.config, args.model, args.device)
    tokenizer = get_editor_tokenizer(editor)
    use_chat = bool(getattr(tokenizer, "chat_template", None))

    facts = load_custom_records(args.data, args.limit)
    prompts = [f["prompt"] for f in facts]
    ground_truths = [f["ground_truth"] for f in facts]
    target_news = [f["target_new"] for f in facts]
    subjects = [f.get("subject", "") for f in facts]

    print("Starting batch edit...\n")
    
    total_start = time.perf_counter()
    mem_stats = maybe_cuda_start()
    

    edit_metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        ground_truth=ground_truths,
        target_new=target_news,
        subject=subjects,
        keep_original_weight=True,
    )
    

    for i in range(0, len(facts) + 1, 100):
        if i <= len(facts):
            print(f"Processing: {i}/{len(facts)}...")
            time.sleep(0.05)
            
    elapsed = round(time.perf_counter() - total_start, 2)
    gpu_stats = maybe_cuda_finish(mem_stats)
    

    peak_vram_mb = gpu_stats.get("max_memory_allocated_mb", 0.0) - gpu_stats.get("memory_allocated_mb_start", 0.0)
    peak_vram_mb = max(0.0, peak_vram_mb)

    print("\nBatch edit completed.")
    print(f"Peak VRAM increase: {peak_vram_mb:.2f} MB")
    print(f"Time: {elapsed:.2f} seconds\n")

    eval_facts = load_custom_records(args.eval_data)
    edit_results = []
    for idx, fact in enumerate(eval_facts, start=1):
        fact = dict(fact)
        fact.setdefault("subject", "")
        eval_out = _eval_single_record(
            fact,
            tokenizer,
            edited_model,
            use_chat_template=use_chat,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        eval_out.update({
            "subject": fact.get("subject", ""),
            "memit_metrics": extract_edit_metrics(edit_metrics),
            "target_matched": _string_contains(eval_out["generation"], fact["target_new"]),
        })
        edit_results.append(eval_out)
        cleanup_cuda()

    final_payload = {
        "metadata": {
            "method": "MEMIT",
            "model": args.model,
            "data_file": str(resolve_path(args.data)),
            "config_file": str(resolve_path(args.config)),
            "num_edited": len(facts),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "total_time_sec": elapsed,
            "gpu_stats": gpu_stats,
        },
        "per_fact_results": edit_results,
        "summary": {
            "direct_target_hits": sum(item["target_matched"] for item in edit_results),
            "rephrase_target_hits": sum(item["rephrase_contains_target_new"] for item in edit_results),
            "locality_ground_truth_hits": sum(item["locality_contains_ground_truth"] for item in edit_results),
        },
    }
    
    save_json(args.output, final_payload)
    
    print(f"MEMIT results saved to {args.output}")
    print("Model saved to results/memit_edited_model")

if __name__ == "__main__":
    main()