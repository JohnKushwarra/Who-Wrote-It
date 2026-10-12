"""Create report-ready figures and verify metrics from saved held-out predictions."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from experiment import LABELS, ROOT, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="initial_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.suite):
        parser.error("suite must be a simple directory name")
    suite = ROOT / "results" / args.suite
    summary = pd.read_csv(suite / "summary.csv")
    expected = {(a, v) for a in ("cnn", "lstm") for v in ("input", "output", "both")}
    if set(zip(summary["model"], summary["view"])) != expected or len(summary) != 6:
        raise ValueError("Expected six completed unique configurations")
    data = [json.loads(line) for line in (ROOT / "data/processed/dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    test = {r["sample_id"]: r for r in data if r["split"] == "test"}
    predictions, matrices = {}, {}
    for row in summary.to_dict("records"):
        key = f"{row['model']}_{row['view']}"
        pred = pd.read_csv(suite / key / "predictions.csv")
        if len(pred) != len(test) or set(pred["sample_id"]) != set(test) or pred["sample_id"].duplicated().any():
            raise ValueError(f"Prediction IDs do not match held-out data: {key}")
        if any(r.actual != test[r.sample_id]["LLM_name"] or r.prompt_group != test[r.sample_id]["prompt_group"] for r in pred.itertuples()):
            raise ValueError(f"Prediction metadata does not match held-out data: {key}")
        accuracy = accuracy_score(pred["actual"], pred["predicted"])
        macro_f1 = f1_score(pred["actual"], pred["predicted"], labels=LABELS, average="macro", zero_division=0)
        if not np.allclose([accuracy, macro_f1], [row["accuracy"], row["macro_f1"]], atol=1e-12):
            raise ValueError(f"Recomputed metrics differ: {key}")
        matrix = confusion_matrix(pred["actual"], pred["predicted"], labels=LABELS)
        saved = pd.read_csv(suite / key / "confusion_matrix.csv", index_col=0)
        if list(saved.columns) != LABELS or list(saved.index) != LABELS or not np.array_equal(matrix, saved.to_numpy()):
            raise ValueError(f"Confusion matrix differs: {key}")
        predictions[key], matrices[key] = pred, matrix

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    figures = suite / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    names = {"input": "Prompt only", "output": "Response only", "both": "Prompt + response"}
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Test accuracy", "Test macro F1")):
        for i, (model, color) in enumerate((("cnn", "#286096"), ("lstm", "#d38233"))):
            values = [float(summary.loc[(summary.model == model) & (summary.view == view), metric].iloc[0]) for view in names]
            bars = ax.bar(np.arange(3) + (i - 0.5) * 0.34, values, width=0.34, label=model.upper(), color=color)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
        ax.set_xticks(range(3), names.values(), rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        baseline = 1 / 3 if metric == "accuracy" else 1 / 6
        ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1, label="Majority baseline")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("LLM attribution: initial held-out comparison")
    fig.supxlabel("375 responses from 125 unseen prompt groups; one training seed; three balanced classes", fontsize=9)
    fig.savefig(figures / "comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    for i, model in enumerate(("cnn", "lstm")):
        for j, view in enumerate(names):
            ax = axes[i, j]
            matrix = matrices[f"{model}_{view}"]
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=125)
            for (r, c), number in np.ndenumerate(matrix):
                ax.text(c, r, str(number), ha="center", va="center", color="white" if number > 65 else "#202020")
            ax.set_xticks(range(3), LABELS)
            ax.set_yticks(range(3), LABELS)
            ax.set_xlabel("Predicted family")
            ax.set_ylabel("Actual family")
            ax.set_title(f"{model.upper()}: {names[view]}")
    fig.suptitle("Test confusion matrices (counts; 125 examples per actual family)")
    fig.savefig(figures / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), layout="constrained")
    for i, model in enumerate(("cnn", "lstm")):
        for j, view in enumerate(names):
            ax = axes[i, j]
            hist = pd.read_json(suite / f"{model}_{view}" / "history.json")
            ax.plot(hist.epoch, hist.train_loss, label="Training", color="#286096")
            ax.plot(hist.epoch, hist.validation_loss, label="Validation", color="#d38233")
            best = int(summary.loc[(summary.model == model) & (summary.view == view), "best_epoch"].iloc[0])
            ax.axvline(best, linestyle=":", color="#555555", label="Selected epoch")
            ax.set_title(f"{model.upper()}: {names[view]}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training and validation loss; selection uses validation only")
    fig.savefig(figures / "learning_curves.png", dpi=180)
    plt.close(fig)

    # Paired bootstrap of the observed RQ2 accuracy difference, resampling entire prompts.
    # This measures finite test-sample variation conditional on trained models, not seed variation.
    comparisons = []
    rng = np.random.default_rng(448)
    for model in ("cnn", "lstm"):
        output, both = predictions[f"{model}_output"], predictions[f"{model}_both"]
        paired = output.merge(both, on=["sample_id", "prompt_group", "actual"], suffixes=("_output", "_both"), validate="one_to_one")
        paired["difference"] = (paired.predicted_both == paired.actual).astype(float) - (paired.predicted_output == paired.actual).astype(float)
        group_means = paired.groupby("prompt_group").difference.mean().to_numpy()
        draws = rng.choice(group_means, size=(2000, len(group_means)), replace=True).mean(axis=1)
        comparisons.append({"model": model, "accuracy_difference_both_minus_output": float(group_means.mean()),
                            "paired_prompt_bootstrap_95_percent_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
                            "prompt_groups": len(group_means), "bootstrap_replicates": 2000,
                            "scope": "Descriptive interval conditional on this dataset slice and these trained models; excludes training-seed variation."})
    audit = {"recomputed_predictions_metrics_and_confusions": "all six passed", "rq2_comparisons": comparisons}
    (suite / "analysis.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    training = pd.DataFrame([{"family": row["LLM_name"], "response_characters": len(row["LLM_output"]),
                             "response_tokens": len(tokens(row["LLM_output"])),
                             "markdown_heading": bool(re.search(r"(?m)^#{1,6}\s", row["LLM_output"]))}
                            for row in data if row["split"] == "train"])
    training.groupby("family").agg(examples=("family", "size"), median_characters=("response_characters", "median"),
        median_tokens=("response_tokens", "median"), heading_fraction=("markdown_heading", "mean")).to_csv(suite / "training_text_summary.csv")
    lines = ["# Initial measured results", "", "Generated from saved predictions; not a final report.", "",
             "| Model | Input | Accuracy | Macro F1 | Selected epoch |", "|---|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.model.upper()} | {names[row.view]} | {row.accuracy:.3f} | {row.macro_f1:.3f} | {row.best_epoch} |")
    lines.extend(["", "Majority baseline: accuracy 0.333, macro F1 0.167. All scores use 375 responses / 125 held-out prompt groups.", "",
                  "Both input-only models scored exactly 1/3 accuracy, as required by the matched-prompt design. This is a dataset-specific control.", ""])
    for comparison in comparisons:
        low, high = comparison["paired_prompt_bootstrap_95_percent_interval"]
        lines.append(f"- {comparison['model'].upper()} combined minus response-only accuracy: {comparison['accuracy_difference_both_minus_output']*100:+.2f} percentage points; paired prompt-bootstrap interval [{low*100:+.2f}, {high*100:+.2f}] points.")
    lines.extend(["", "These intervals describe test-sample variation for fixed trained models. One seed, one filtered source slice, prefix truncation and potential near-duplicates limit generalization.", "",
                  "Training-text summaries are descriptive only; they do not establish which signals a classifier uses or complete RQ4.", "",
                  "![Comparison](figures/comparison.png)", "", "![Confusion matrices](figures/confusion_matrices.png)", "", "![Learning curves](figures/learning_curves.png)", ""])
    (suite / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
