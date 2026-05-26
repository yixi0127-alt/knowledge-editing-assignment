"""Compute ES, PS, NS metrics for knowledge editing results (Task 4)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from editing_utils import load_json, resolve_path, save_json

def get_command_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate editing outputs and produce metrics.")
    parser.add_argument("--baseline", default="results/baseline.json", help="Path to baseline JSON.")
    parser.add_argument("--rome", default="results/rome_results.json", help="Path to ROME results.")
    parser.add_argument("--memit", default="results/memit_results.json", help="Path to MEMIT results.")
    parser.add_argument("--output", default="results/metrics.json", help="Output path for computed metrics.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing result files instead of failing.",
    )
    return parser.parse_args()

def to_boolean(value: Any) -> bool:
    return bool(value) if value is not None else False

def compute_percentage(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0

def extract_results(payload: dict[str, Any], file_path: Path, method: str) -> list[dict[str, Any]]:
    """Extract result list from ROME, MEMIT, or Baseline output file."""
    if method == "rome" or method == "memit":
        key = "per_fact_results"
    else:
        key = "results" 
        
    results = payload.get(key)
    if not isinstance(results, list):
        results = payload.get("results")
        if not isinstance(results, list):
            results = payload.get("detailed_results")
            if not isinstance(results, list):
                raise ValueError(f"Expected list under '{key}' or 'results' in {file_path}")
    return results

def compute_method_metrics(payload: dict[str, Any], file_path: Path, method: str) -> dict[str, Any]:
    results = extract_results(payload, file_path, method)
    efficacy = []
    paraphrase = []
    locality = []

    for item in results:
        eff = item.get("target_matched")
        if eff is None:
            eff = item.get("contains_target_new")
        efficacy.append(to_boolean(eff))
        
        para = item.get("rephrase_hit_target")
        if para is None:
            para = item.get("rephrase_contains_target_new")
        paraphrase.append(to_boolean(para))
        
        loc = item.get("locality_hit_gt")
        if loc is None:
            loc = item.get("locality_contains_ground_truth")
        locality.append(to_boolean(loc))

    return {
        "num_records": len(results),
        "ES": compute_percentage(efficacy),
        "PS": compute_percentage(paraphrase),
        "NS": compute_percentage(locality),
        "counts": {
            "efficacy_hits": sum(efficacy),
            "paraphrase_hits": sum(paraphrase),
            "neighborhood_hits": sum(locality), 
        },
    }

def try_load_result(name: str, path_str: str, allow_missing: bool):
    path = resolve_path(path_str)
    if not path.exists():
        if allow_missing:
            print(f"Skipping missing {name} result: {path}")
            return None
        raise FileNotFoundError(f"Missing {name} result file: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return name, path, payload

def main() -> None:
    args = get_command_arguments()
    requests = [
        ("baseline", args.baseline),
        ("rome", args.rome),
        ("memit", args.memit),
    ]
    loaded = {name: payload for name, path, payload in (try_load_result(n, p, args.allow_missing) for n, p in requests if try_load_result(n, p, args.allow_missing) is not None)}

    all_metrics: dict[str, Any] = {}
    
    if "baseline" in loaded:
        all_metrics["baseline"] = compute_method_metrics(loaded["baseline"], Path(args.baseline), "baseline")
    if "rome" in loaded:
        all_metrics["rome"] = compute_method_metrics(loaded["rome"], Path(args.rome), "rome")
    if "memit" in loaded:
        all_metrics["memit"] = compute_method_metrics(loaded["memit"], Path(args.memit), "memit")

    output_payload = {
        "metrics": all_metrics,
        "metric_definitions": {
            "ES": "Efficacy Score: fraction of direct edited prompts whose generation contains target_new.",
            "PS": "Paraphrase Score: fraction of rephrase prompts whose generation contains target_new.",
            "NS": "Neighborhood Score: fraction of locality prompts whose generation preserves locality_ground_truth.",
        },
    }
    save_json(args.output, output_payload)
    
   
    print(f"Metrics computed from {args.output}\n")
    
    if "rome" in all_metrics:
        d = all_metrics["rome"]
        print(f"ROME: ES={d['ES']:.4f}, PS={d['PS']:.4f}, NS={d['NS']:.4f} ({d['num_records']} records)")
    if "memit" in all_metrics:
        d = all_metrics["memit"]
        print(f"MEMIT: ES={d['ES']:.4f}, PS={d['PS']:.4f}, NS={d['NS']:.4f} ({d['num_records']} records)\n")
    if "baseline" in all_metrics:
        d = all_metrics["baseline"]
        print(f"Baseline: ES={d['ES']:.4f}, PS={d['PS']:.4f}, NS={d['NS']:.4f} ({d['num_records']} records)\n")
        
    print(f"Saved metrics to {args.output}")

if __name__ == "__main__":
    main()