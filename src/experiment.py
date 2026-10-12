"""Run the six required comparisons with a shared split and training-only vocabulary.

Inputs: data/processed/dataset.jsonl and manifest.json.
Writes: immutable run folders under results/<suite>/ with config, learning curves,
checkpoint, predictions, metrics and confusion matrices. No results are invented.
"""
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
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import TextCNN, TextLSTM

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Gemma", "Mistral", "Qwen"]
VIEWS = ("input", "output", "both")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NAME_RE = re.compile(r"\b(?:gemma(?:\s*3)?|qwen(?:\s*3)?|mistral)\b", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    # Same transformation for every family; metadata never enters the text.
    text = NAME_RE.sub(" __model__ ", text)
    return TOKEN_RE.findall(text)


def build_vocab(rows: list[dict], max_vocab: int) -> dict[str, int]:
    counts = Counter(token for row in rows for field in ("LLM_Input", "LLM_output") for token in tokens(row[field]))
    vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    # Alphabetical tie break makes rebuilding independent of dictionary order.
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token not in vocab:
            vocab[token] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def encode(row: dict, view: str, vocab: dict, tokens_per_field: int) -> list[int]:
    def one(field):
        return [vocab.get(t, 1) for t in tokens(row[field])[:tokens_per_field]] or [1]
    if view == "input":
        return one("LLM_Input")
    if view == "output":
        return one("LLM_output")
    if view == "both":
        # Preserve the same response prefix as output-only, even with a long prompt.
        return one("LLM_Input") + [vocab["<sep>"]] + one("LLM_output")
    raise ValueError(view)


class TextDataset(Dataset):
    def __init__(self, rows, view, vocab, tokens_per_field):
        self.rows = rows
        self.sequences = [encode(row, view, vocab, tokens_per_field) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.sequences[index], LABELS.index(self.rows[index]["LLM_name"])


def collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = torch.zeros((len(batch), max(5, lengths.max().item())), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence)] = torch.tensor(sequence)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def validate_data(rows):
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    groups = {}
    for row in rows:
        if row["LLM_name"] not in LABELS or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown class or split")
        if not row["LLM_Input"].strip() or not row["LLM_output"].strip():
            raise ValueError("Empty text")
        groups.setdefault(row["prompt_group"], []).append(row)
    prompt_splits, output_splits = {}, {}
    for group in groups.values():
        if sorted(r["LLM_name"] for r in group) != LABELS:
            raise ValueError("Each prompt must contain one response from each family")
        if len({r["LLM_Input"] for r in group}) != 1 or len({r["split"] for r in group}) != 1:
            raise ValueError("A prompt group differs in text or crosses split boundaries")
    for row in rows:
        for field, seen in (("LLM_Input", prompt_splits), ("LLM_output", output_splits)):
            key = " ".join(row[field].split()).casefold()
            if key in seen and seen[key] != row["split"]:
                raise ValueError(f"Duplicate {field} across splits")
            seen[key] = row["split"]
    for split in ("train", "validation", "test"):
        if {r["LLM_name"] for r in rows if r["split"] == split} != set(LABELS):
            raise ValueError(f"Missing class in {split}")


def evaluate(model, loader):
    model.eval()
    losses, true, predicted = 0.0, [], []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x, lengths)
            losses += criterion(logits, y).item()
            true.extend(y.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
    return {"loss": losses / len(true), "accuracy": accuracy_score(true, predicted),
            "macro_f1": f1_score(true, predicted, labels=range(len(LABELS)), average="macro", zero_division=0)}, true, predicted


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_evaluation(directory, true, predicted, rows, metrics):
    write_json(directory / "metrics.json", {**metrics, "classification_report": classification_report(
        true, predicted, labels=range(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0)})
    matrix = confusion_matrix(true, predicted, labels=range(len(LABELS)))
    with (directory / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *LABELS])
        writer.writerows([label, *row] for label, row in zip(LABELS, matrix.tolist()))
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prompt_group", "actual", "predicted"])
        writer.writerows([row["sample_id"], row["prompt_group"], LABELS[y], LABELS[p]] for row, y, p in zip(rows, true, predicted))


def train_one(architecture, view, partitions, vocab, args, suite, provenance):
    directory = suite / f"{architecture}_{view}"
    directory.mkdir(exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = {split: TextDataset(rows, view, vocab, args.tokens_per_field) for split, rows in partitions.items()}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", collate_fn=collate,
                                generator=torch.Generator().manual_seed(args.seed), num_workers=0)
               for split, dataset in datasets.items()}
    model_class = TextCNN if architecture == "cnn" else TextLSTM
    model_kwargs = dict(vocab_size=len(vocab), classes=len(LABELS), embedding_dim=args.embedding_dim,
                        hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model_class(**model_kwargs)
    config = {**vars(args), "architecture": architecture, "view": view, "model_kwargs": model_kwargs,
              "labels": LABELS, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "selection": "Lowest validation cross-entropy; early stopping patience 3; test evaluated after selection.",
              "preprocessing": "Case-sensitive word/punctuation tokens; common model-name mask; per-field prefix cap; training-only vocabulary.",
              "device": "cpu", "torch_version": str(torch.__version__), "python": sys.version,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), **provenance}
    write_json(directory / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history, best_loss, stale, best_state, best_epoch = [], float("inf"), 0, None, None
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, lengths, y in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        validation, _, _ = evaluate(model, loaders["validation"])
        history.append({"epoch": epoch, "train_loss": total / n,
                        **{f"validation_{key}": value for key, value in validation.items()}})
        write_json(directory / "history.json", history)
        print(f"{architecture}/{view} epoch {epoch}: train_loss={total/n:.4f}, val_loss={validation['loss']:.4f}, val_f1={validation['macro_f1']:.3f}", flush=True)
        if validation["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, best_state, stale = validation["loss"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "vocab": vocab, "config": config, "best_epoch": best_epoch}, directory / "checkpoint.pt")
    metrics, true, predicted = evaluate(model, loaders["test"])
    # Reload the artifact that a future user will run; verify identical predictions.
    saved = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
    restored = model_class(**saved["config"]["model_kwargs"])
    restored.load_state_dict(saved["state_dict"])
    _, _, replay = evaluate(restored, loaders["test"])
    if replay != predicted:
        raise AssertionError("Saved checkpoint predictions do not reproduce")
    metrics.update({"best_epoch": best_epoch, "epochs_run": len(history), "seconds": time.monotonic() - start,
                    "test_rows": len(true), "checkpoint_replay_verified": True})
    save_evaluation(directory, true, predicted, partitions["test"], metrics)
    return {"model": architecture, "view": view, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="initial_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=448)
    parser.add_argument("--tokens-per-field", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.suite):
        parser.error("suite must be a simple directory name")
    if min(args.epochs, args.tokens_per_field, args.embedding_dim, args.hidden_dim, args.batch_size, args.threads) < 1 or args.max_vocab < 4:
        parser.error("dimensions, epochs, batch size and threads must be positive; vocab >= 4")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data_path = ROOT / "data/processed/dataset.jsonl"
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    validate_data(rows)
    partitions = {split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "test")}
    vocab = build_vocab(partitions["train"], args.max_vocab)
    suite = ROOT / "results" / args.suite
    suite.mkdir(parents=True, exist_ok=False)
    provenance = {"data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("models.py"))}}
    source_copy = suite / "source"
    source_copy.mkdir()
    for name in ("experiment.py", "models.py", "prepare_data.py"):
        (source_copy / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(suite / "vocab.json", vocab)
    truncation = {split: {field: sum(len(tokens(r[field])) > args.tokens_per_field for r in values)
                          for field in ("LLM_Input", "LLM_output")} for split, values in partitions.items()}
    write_json(suite / "data_summary.json", {"rows_by_split": {s:len(v) for s,v in partitions.items()},
        "truncated_rows_by_field": truncation, "vocabulary_size": len(vocab), "label_order": LABELS,
        "input_only_control": "Each prompt occurs once per class in every split. Deterministic input-only predictions must score exactly 1/3 accuracy."})
    # Majority baseline is determined from training labels, then evaluated on held-out data.
    majority = Counter(r["LLM_name"] for r in partitions["train"]).most_common(1)[0][0]
    truth = [LABELS.index(r["LLM_name"]) for r in partitions["test"]]
    baseline = [LABELS.index(majority)] * len(truth)
    write_json(suite / "majority_baseline.json", {"majority_label": majority, "test_accuracy": accuracy_score(truth, baseline),
        "test_macro_f1": f1_score(truth, baseline, labels=range(3), average="macro", zero_division=0)})
    summary = []
    for architecture in ("cnn", "lstm"):
        for view in VIEWS:
            result = train_one(architecture, view, partitions, vocab, args, suite, provenance)
            if view == "input" and abs(result["accuracy"] - 1 / 3) > 1e-12:
                raise AssertionError("Input-only negative control failed; inspect data or evaluation")
            summary.append(result)
            write_json(suite / "summary.json", summary)
    with (suite / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
