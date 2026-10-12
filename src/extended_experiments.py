"""RQ3 task transfer and RQ4 controlled signal ablations; initial runs are untouched."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiment import ROOT, LABELS, build_vocab, collate, evaluate, save_evaluation, tokens, validate_data, write_json
from models import TextCNN, TextLSTM

SUITE = ROOT / "results/rq34_v1"
SEEDS = (448, 449, 450)
TASKS = ("question_answering", "drafting")
CONDITIONS = ("baseline", "punctuation_masked", "words_masked")
WORD = re.compile(r"^\w+$", re.UNICODE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows():
    path = ROOT / "data/processed/dataset.jsonl"
    previous = json.loads((ROOT / "results/initial_v1/cnn_output/config.json").read_text(encoding="utf-8"))
    if sha(path) != previous["data_sha256"]:
        raise ValueError("Original dataset bytes changed; explicitly plan a new data version.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate_data(rows)
    return rows


def task_core(prompt):
    marker = "Generate your answer based on the required response format above:"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[1]
    elif prompt.startswith("Answer in accordance with the format:") and "\n\n" in prompt:
        prompt = prompt.split("\n\n", 1)[1]
    return re.sub(r"^(?:(?:Task|Question|Generated Task|Revised Task|Inquiring Question):\s*)+", "", prompt.strip(), flags=re.I).strip()


def task_label(prompt):
    core = task_core(prompt)
    if re.search(r"\bAnswer\s*:", core, re.I):
        return "other"
    if re.match(r"^(?:How (?:do|does|did|can|could|would|will|has|have|is|are)|What\b|Why\b|In what ways\b|To what extent\b)", core, re.I) and "?" in core:
        return "question_answering"
    if re.match(r"^(?:Create|Develop|Design|Draft|Compose|Write|Formulate)\b", core, re.I):
        return "drafting"
    return "other"


def select_rq3(rows):
    unique = {row["prompt_group"]: row for row in rows}
    pools = {(task, split): sorted(g for g, r in unique.items() if task_label(r["LLM_Input"]) == task and r["split"] == split)
             for task in TASKS for split in ("train", "validation", "test")}
    selected = {}
    for split in ("train", "validation"):
        count = min(len(pools[(task, split)]) for task in TASKS)
        if count < 15:
            raise ValueError(f"Too few {split} prompt groups for RQ3")
        for task in TASKS:
            chosen = pools[(task, split)].copy()
            random.Random(448).shuffle(chosen)
            selected[(task, split)] = set(chosen[:count])
    for task in TASKS:
        selected[(task, "test")] = set(pools[(task, "test")])
        if len(selected[(task, "test")]) < 15:
            raise ValueError("Too few test groups")
    partitions = {task: {split: [dict(r) for r in rows if r["prompt_group"] in selected[(task, split)]]
                         for split in ("train", "validation", "test")} for task in TASKS}
    return partitions, pools


def prefix_ids(row, vocab, condition="baseline"):
    prefix = tokens(row["LLM_output"])[:128]
    ids = [vocab.get(token, 1) for token in prefix] or [1]
    if condition == "baseline":
        return ids
    if condition == "punctuation_masked":
        return [value if WORD.fullmatch(token) else 1 for token, value in zip(prefix, ids)] or [1]
    if condition == "words_masked":
        return [1 if WORD.fullmatch(token) else value for token, value in zip(prefix, ids)] or [1]
    if condition == "order_shuffled":
        # Same non-label permutation rule for each prompt group; does not change token multiset.
        seed = int(hashlib.sha256((row["prompt_group"] + "|shuffle448").encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(ids)
        return ids
    raise ValueError(condition)


class PrefixDataset(Dataset):
    def __init__(self, rows, vocab, condition="baseline"):
        self.rows = rows
        self.sequences = [prefix_ids(r, vocab, condition) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.sequences[index], LABELS.index(self.rows[index]["LLM_name"])


def loader(rows, vocab, condition="baseline", shuffle=False, seed=448):
    return DataLoader(PrefixDataset(rows, vocab, condition), batch_size=64, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def prepare():
    if SUITE.exists():
        raise FileExistsError("Extension suite already exists; preserving its frozen design.")
    rows = load_rows()
    partitions, pools = select_rq3(rows)
    SUITE.mkdir(parents=True)
    source = SUITE / "source"
    source.mkdir()
    for name in ("extended_experiments.py", "experiment.py", "models.py"):
        (source / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    unique = {r["prompt_group"]: r for r in rows}
    audit_ids = set()
    for task in TASKS:
        candidates = sorted(g for g, r in unique.items() if task_label(r["LLM_Input"]) == task)
        audit_ids.update(random.Random(448).sample(candidates, min(20, len(candidates))))
    with (SUITE / "task_annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_group", "original_split", "task", "selected_for_rq3", "audit_sample", "prompt_core"])
        writer.writeheader()
        chosen_ids = {r["prompt_group"] for task in partitions.values() for split in task.values() for r in split}
        for gid, row in sorted(unique.items()):
            writer.writerow(dict(prompt_group=gid, original_split=row["split"], task=task_label(row["LLM_Input"]),
                                 selected_for_rq3=gid in chosen_ids, audit_sample=gid in audit_ids, prompt_core=task_core(row["LLM_Input"])))
    write_json(SUITE / "rq3_partitions.json", partitions)
    design = {"planned_at_utc": datetime.now(timezone.utc).isoformat(), "client_date": "2026-10-08", "seeds": SEEDS,
              "data_sha256": sha(ROOT / "data/processed/dataset.jsonl"), "annotations_sha256": sha(SUITE / "task_annotations.csv"),
              "rq3_partitions_sha256": sha(SUITE / "rq3_partitions.json"),
              "source_hashes": {p.name: sha(p) for p in source.glob("*.py")},
              "rq3_candidate_groups": {f"{task}/{split}": len(ids) for (task, split), ids in pools.items()},
              "rq3_selected_groups": {task: {split: len(values)//3 for split, values in parts.items()} for task, parts in partitions.items()},
              "rq3": "Train only on one task; source-only validation and vocabulary; equal train/validation counts across tasks; evaluate both task test sets. Three seeds; no task test groups in training or selection.",
              "rq3_label_limit": "Prompt-only deterministic task-form heuristics; broad overlapping subject matter, not expert domain labels. Forty sampled prompts receive assistant audit before training.",
              "rq4": "Same full original partitions/vocabulary; baseline and two masks retrained for each architecture/seed. Frozen baseline also tested with two masks and shuffled order.",
              "rq4_control": "First 128 original tokens fixed before masking. Masks use ID 1 (generic/UNK); keep positions, length, vocabulary size and parameter count. Padding ID 0 is never substituted.",
              "evaluation_scope": "Exploratory follow-up: initial test results have already been seen. Fixed protocols for these new runs; not a new pristine test benchmark.",
              "training": {"epochs":12,"patience":3,"learning_rate":0.001,"batch_size":64,"embedding_dim":64,"hidden_dim":64,"dropout":0.3,"gradient_clip":1.0,"selection":"minimum source validation loss","threads":2}}
    write_json(SUITE / "design.json", design)
    print(json.dumps(design["rq3_selected_groups"], indent=2), flush=True)


def verify_design():
    design = json.loads((SUITE / "design.json").read_text(encoding="utf-8"))
    if sha(ROOT / "data/processed/dataset.jsonl") != design["data_sha256"]:
        raise ValueError("Dataset changed after planning")
    for name, expected in design["source_hashes"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Execution source changed after planning: {name}")
    if sha(SUITE / "rq3_partitions.json") != design["rq3_partitions_sha256"] or sha(SUITE / "task_annotations.csv") != design["annotations_sha256"]:
        raise ValueError("Task annotations or partitions changed after freezing")
    return design


def fit(architecture, condition, train, validation, vocab, seed, directory, metadata):
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_class = TextCNN if architecture == "cnn" else TextLSTM
    kwargs = dict(vocab_size=len(vocab), classes=3, embedding_dim=64, hidden_dim=64, dropout=0.3)
    model = model_class(**kwargs)
    config = {"architecture":architecture,"condition":condition,"seed":seed,"model_kwargs":kwargs,"labels":LABELS,
              "training_rows":len(train),"validation_rows":len(validation),"torch_version":str(torch.__version__),
              "design_sha256":sha(SUITE / "design.json"),"trainable_parameters":sum(p.numel() for p in model.parameters()),**metadata}
    write_json(directory / "config.json", config)
    train_loader = loader(train, vocab, condition, shuffle=True, seed=seed)
    validation_loader = loader(validation, vocab, condition, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best, state, stale, best_epoch = float("inf"), None, 0, None
    history = []
    start = time.monotonic()
    for epoch in range(1, 13):
        model.train()
        total = 0.0
        for x, lengths, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
        val, _, _ = evaluate(model, validation_loader)
        history.append({"epoch":epoch,"train_loss":total/len(train),**{f"validation_{k}":v for k,v in val.items()}})
        write_json(directory / "history.json", history)
        if val["loss"] < best - 1e-5:
            best, state, stale, best_epoch = val["loss"], copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(state)
    torch.save({"state_dict":state,"vocab":vocab,"config":config,"best_epoch":best_epoch}, directory / "checkpoint.pt")
    saved = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
    restored = model_class(**saved["config"]["model_kwargs"])
    restored.load_state_dict(saved["state_dict"])
    if evaluate(model, validation_loader)[2] != evaluate(restored, validation_loader)[2]:
        raise AssertionError("Checkpoint reload differs")
    info = {"best_epoch":best_epoch,"epochs_run":len(history),"training_seconds":time.monotonic()-start,"checkpoint_replay_verified":True}
    write_json(directory / "training.json", info)
    print(f"Trained {directory.relative_to(SUITE)}: epoch {best_epoch}, val loss {best:.4f}, {info['training_seconds']:.1f}s", flush=True)
    return restored, info


def score(model, rows, vocab, condition, directory, info):
    directory.mkdir(parents=True, exist_ok=False)
    metrics, true, predicted = evaluate(model, loader(rows, vocab, condition))
    save_evaluation(directory, true, predicted, rows, {**metrics, **info, "test_rows":len(rows)})
    return {**metrics, "test_rows":len(rows), "test_groups":len(rows)//3, "prediction_path":str((directory/"predictions.csv").relative_to(SUITE))}


def run_rq3():
    design = verify_design()
    if not (SUITE / "TASK_AUDIT.md").exists():
        raise ValueError("Review and document task labels before training")
    partitions = json.loads((SUITE / "rq3_partitions.json").read_text(encoding="utf-8"))
    results = []
    for source in TASKS:
        parts = partitions[source]
        vocab = build_vocab(parts["train"], 12000)
        for seed in SEEDS:
            for architecture in ("cnn", "lstm"):
                run_dir = SUITE / "rq3" / source / f"seed_{seed}" / architecture
                model, info = fit(architecture, "baseline", parts["train"], parts["validation"], vocab, seed, run_dir, {"rq":"RQ3","train_task":source})
                for target in TASKS:
                    metrics = score(model, partitions[target]["test"], vocab, "baseline", run_dir/f"test_{target}",info)
                    results.append({"architecture":architecture,"seed":seed,"train_task":source,"test_task":target,**metrics,**info})
                write_json(SUITE / "rq3_summary.json", results)
    print("RQ3 complete: 12 training runs, 24 task evaluations", flush=True)


def run_rq4():
    verify_design()
    rows = load_rows()
    parts = {s:[r for r in rows if r["split"]==s] for s in ("train","validation","test")}
    vocab = build_vocab(parts["train"], 12000)
    original_vocab = json.loads((ROOT / "results/initial_v1/vocab.json").read_text(encoding="utf-8"))
    if vocab != original_vocab:
        raise ValueError("RQ4 vocabulary no longer matches the initial experiment")
    results, perturbations = [], []
    for seed in SEEDS:
        for architecture in ("cnn", "lstm"):
            for condition in CONDITIONS:
                run_dir = SUITE / "rq4" / f"seed_{seed}" / architecture / condition
                model, info = fit(architecture,condition,parts["train"],parts["validation"],vocab,seed,run_dir,{"rq":"RQ4"})
                metrics = score(model,parts["test"],vocab,condition,run_dir/"test",info)
                results.append({"architecture":architecture,"seed":seed,"condition":condition,**metrics,**info})
                write_json(SUITE / "rq4_summary.json",results)
                if condition == "baseline":
                    # A regression check: same original experiment should give identical labels.
                    if seed == 448:
                        with (ROOT/f"results/initial_v1/{architecture}_output/predictions.csv").open(encoding="utf-8") as handle:
                            old = list(csv.DictReader(handle))
                        with (run_dir/"test/predictions.csv").open(encoding="utf-8") as handle:
                            current = list(csv.DictReader(handle))
                        if old != current:
                            raise AssertionError("Reference run differs from original seed-448 predictions")
                    for perturbation in (*CONDITIONS[1:], "order_shuffled"):
                        perturbed = score(model,parts["test"],vocab,perturbation,run_dir/f"frozen_{perturbation}",info)
                        perturbations.append({"architecture":architecture,"seed":seed,"condition":perturbation,"baseline_accuracy":metrics["accuracy"],**perturbed})
                    write_json(SUITE / "rq4_frozen_summary.json",perturbations)
    print("RQ4 complete: 18 training runs, 18 frozen-model perturbation evaluations",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section",choices=["prepare","rq3","rq4"])
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    {"prepare":prepare,"rq3":run_rq3,"rq4":run_rq4}[args.section]()


if __name__ == "__main__":
    main()
