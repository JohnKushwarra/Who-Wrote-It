"""Acquire a bounded public sample, match prompts, and split entire prompt groups."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATASET = "allenai/Dolci-DPO-Model-Response-Pool"
MODELS = {"gemma3-12b": "Gemma", "mistral-24b": "Mistral", "qwen3-no_reasoning-4b": "Qwen"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def get_json(url: str) -> dict:
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CMPSC448-educational-project/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download(config: str, count: int) -> dict:
    directory = ROOT / "data/raw" / config
    directory.mkdir(parents=True, exist_ok=True)
    rows, sources, truncated = [], [], 0
    for offset in range(0, count, 100):
        length = min(100, count - offset)
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            dict(dataset=DATASET, config=config, split="train", offset=offset, length=length)
        )
        target = directory / f"rows_{offset:06d}_{length}.json"
        if not target.exists():
            payload = get_json(url)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if "rows" not in payload:
            raise ValueError(f"Invalid dataset response in {target}")
        sources.append({"url": url, "path": str(target.relative_to(ROOT)), "sha256": digest(target.read_bytes())})
        for item in payload["rows"]:
            if item.get("truncated_cells"):
                truncated += 1
                continue
            row = item["row"]
            if row["model"] != config:
                raise ValueError(f"Unexpected model label in {target}")
            rows.append(row)
    print(f"{config}: {len(rows)} complete API rows, {truncated} truncated rows excluded", flush=True)
    return {"rows": rows, "sources": sources, "truncated_rows": truncated}


def normalize_prompt(text: str) -> str:
    return " ".join(text.split()).casefold()


def split_groups(group_ids: list[str], seed: int) -> dict[str, str]:
    ids = sorted(set(group_ids))
    if len(ids) < 20:
        raise ValueError("Need at least 20 distinct matched prompts; increase the acquisition size.")
    random.Random(seed).shuffle(ids)
    n_train, n_val = int(0.70 * len(ids)), int(0.15 * len(ids))
    return {gid: ("train" if i < n_train else "validation" if i < n_train + n_val else "test")
            for i, gid in enumerate(ids)}


def exclusion_reason(prompt: str, responses: list[str]) -> str | None:
    # A deliberately narrow convenience sample, not a claim of complete privacy screening.
    if not 30 <= len(prompt) <= 4000:
        return "prompt_length"
    if any(not 50 <= len(text) <= 20000 for text in responses):
        return "response_length"
    if "json schema" in prompt.casefold() or "extract relevant information from the paragraph" in prompt.casefold():
        return "record_extraction_prompt"
    if any(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text) for text in [prompt, *responses]):
        return "email_pattern"
    # Identical output under two labels provides no attribution signal; remove the entire group.
    if len({normalize_prompt(text) for text in responses}) != len(responses):
        return "identical_responses_within_prompt"
    return None


def prepare(payloads: dict, seed: int, max_prompts: int) -> tuple[list[dict], dict]:
    lookups = {}
    for model, payload in payloads.items():
        lookup = {}
        for row in payload["rows"]:
            if row["custom_id"] in lookup:
                raise ValueError(f"Duplicate source ID for {model}: {row['custom_id']}")
            lookup[row["custom_id"]] = row
        lookups[model] = lookup
    common = sorted(set.intersection(*(set(table) for table in lookups.values())))
    excluded = Counter()
    retained, seen_prompts, seen_responses = [], set(), set()
    for source_id in common:
        items = [lookups[m][source_id] for m in MODELS]
        prompts = [item["prompt"].strip() for item in items]
        if len(set(prompts)) != 1:
            excluded["prompt_mismatch"] += 1
            continue
        prompt = prompts[0]
        texts = [item["response"].strip() for item in items]
        reason = exclusion_reason(prompt, texts)
        if reason:
            excluded[reason] += 1
            continue
        key = normalize_prompt(prompt)
        response_keys = {normalize_prompt(text) for text in texts}
        if key in seen_prompts:
            excluded["duplicate_prompt"] += 1
            continue
        if response_keys & seen_responses:
            excluded["duplicate_response_across_prompts"] += 1
            continue
        seen_prompts.add(key)
        seen_responses.update(response_keys)
        retained.append((source_id, prompt, items))
    random.Random(seed).shuffle(retained)
    before_cap = len(retained)
    retained = retained[:max_prompts]
    groups = [digest(normalize_prompt(prompt).encode()) for _, prompt, _ in retained]
    splits = split_groups(groups, seed)
    result = []
    for (source_id, prompt, items), gid in zip(retained, groups):
        for model, item in zip(MODELS, items):
            result.append({"sample_id": f"{gid[:16]}_{MODELS[model]}", "prompt_group": gid,
                           "source_id": source_id, "source_dataset": item["dataset"],
                           "LLM_name": MODELS[model], "model_id": model,
                           "LLM_Input": prompt, "LLM_output": item["response"].strip(),
                           "split": splits[gid]})
    result.sort(key=lambda x: (x["split"], x["prompt_group"], x["LLM_name"]))
    counts = {split: dict(Counter(r["LLM_name"] for r in result if r["split"] == split))
              for split in ("train", "validation", "test")}
    return result, {"shared_source_ids": len(common), "exclusions": dict(excluded),
                    "eligible_prompts_before_cap": before_cap, "selected_prompts": len(retained),
                    "selected_rows": len(result), "counts_by_split_and_family": counts,
                    "source_datasets": dict(Counter(r["source_dataset"] for r in result)),
                    "policy": "Complete matched triples; first source rows; filter entire prompt groups; seed-shuffled cap; 70/15/15 grouped split."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-model", type=int, default=1500)
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=448)
    args = parser.parse_args()
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset.jsonl").exists():
        raise FileExistsError("Prepared dataset already exists. Preserve it; use a separate project copy for a revised dataset.")
    metadata = get_json("https://huggingface.co/api/datasets/" + DATASET)
    with ThreadPoolExecutor(max_workers=3) as pool:
        payloads = dict(zip(MODELS, pool.map(lambda m: download(m, args.rows_per_model), MODELS)))
    rows, audit = prepare(payloads, args.seed, args.max_prompts)
    dataset_path = output / "dataset.jsonl"
    dataset_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"dataset": DATASET, "dataset_card": "https://huggingface.co/datasets/" + DATASET,
                "repo_revision_observed": metadata.get("sha"),
                "revision_note": "Viewer API is not revision-pinned; cached response SHA-256 values identify the actual acquired content.",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "models": MODELS, "settings": vars(args), "dataset_sha256": digest(dataset_path.read_bytes()),
                "audit": audit, "raw_sources": {m: {k:v for k,v in p.items() if k != "rows"} for m,p in payloads.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
