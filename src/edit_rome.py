"""ROME algorithm for single-fact knowledge editing (Task 2)."""

from __future__ import annotations

import argparse
import time
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
    parser = argparse.ArgumentParser(description="Apply ROME edits to custom facts.")
    parser.add_argument("--data", default="data/custom_facts.json", help="Path to fact dataset.")

    parser.add_argument("--output", default="results/rome_results.json", help="Output JSON file.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model identifier.")
    parser.add_argument("--config", default="configs/rome.yaml", help="ROME hyperparameter YAML.")
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of facts to edit.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation length limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy).")
    return parser.parse_args()

def init_rome_engine(config_path: str, model_name: str, device_id: str) -> Any:
    try:
        from easyeditor import BaseEditor, ROMEHyperParams
    except ImportError as exc:
        raise SystemExit("EasyEdit not found. Install via: pip install -r requirements.txt") from exc
    runtime_cfg = write_runtime_hparams(config_path, model_name=model_name, device=device_id)
    hps = ROMEHyperParams.from_hparams(str(runtime_cfg))
    return BaseEditor.from_hparams(hps)

def apply_rome_edit(editor: Any, fact: dict[str, Any]) -> Any:
    return editor.edit(
        prompts=[fact["prompt"]],
        ground_truth=[fact["ground_truth"]],
        target_new=[fact["target_new"]],
        subject=[fact.get("subject", "")],
        keep_original_weight=True,
    )

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
    facts = load_custom_records(args.data, args.limit)
    total_start = time.perf_counter()
    mem_stats = maybe_cuda_start()
    editor = init_rome_engine(args.config, args.model, args.device)
    tokenizer = get_editor_tokenizer(editor)
    use_chat = bool(getattr(tokenizer, "chat_template", None))

    edit_results = []
    for idx, fact in enumerate(facts, start=1):
        fact = dict(fact)
        fact.setdefault("subject", "")
        print(f"[{idx}/{len(facts)}] Editing: {fact['prompt']} -> {fact['target_new']}")
        edit_metrics, edited_model, _ = apply_rome_edit(editor, fact)
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
            "rome_metrics": extract_edit_metrics(edit_metrics),
            "target_matched": _string_contains(eval_out["generation"], fact["target_new"]),
        })
        edit_results.append(eval_out)
        cleanup_cuda()

    elapsed = round(time.perf_counter() - total_start, 3)
    final_payload = {
        "metadata": {
            "method": "ROME",
            "model": args.model,
            "data_file": str(resolve_path(args.data)),
            "config_file": str(resolve_path(args.config)),
            "num_edited": len(facts),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "total_time_sec": elapsed,
            "gpu_stats": maybe_cuda_finish(mem_stats),
        },
        "per_fact_results": edit_results,
        "summary": {
            "direct_target_hits": sum(item["target_matched"] for item in edit_results),
            "rephrase_target_hits": sum(item["rephrase_contains_target_new"] for item in edit_results),
            "locality_ground_truth_hits": sum(item["locality_contains_ground_truth"] for item in edit_results),
        },
    }
    save_json(args.output, final_payload)
    

    print(f"ROME results saved to {args.output}")
    print(
        f"Summary: direct_target_hits={final_payload['summary']['direct_target_hits']}, "
        f"rephrase_target_hits={final_payload['summary']['rephrase_target_hits']}, "
        f"locality_gt_hits={final_payload['summary']['locality_ground_truth_hits']}"
    )

if __name__ == "__main__":
    main()