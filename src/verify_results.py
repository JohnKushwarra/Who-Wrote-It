"""Check saved data, tests, metrics and model predictions without retraining."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("original_test", "confirmation")


def project_path(name):
    """Resolve saved Windows or POSIX paths within this project only."""
    path = (ROOT / str(name).replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path is outside the project: {name}")
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_dataset(folder):
    path = project_path(f"data/{folder}/dataset.jsonl")
    manifest = read_json(path.parent / "manifest.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == manifest["dataset_sha256"], f"Dataset hash mismatch: {path.relative_to(ROOT)}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows, digest


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    require(result.testsRun > 0, "No tests were discovered")
    require(result.wasSuccessful(), "Project tests failed")
    require(not result.skipped, "Some project tests were skipped")
    return {"run": result.testsRun, "skipped": len(result.skipped)}


def prediction_index(pd, registry):
    """Use the saved indexes so a missing result or checkpoint fails the check."""
    initial = ROOT / "results/initial_v1"
    extensions = ROOT / "results/rq34_v1"
    classifiers = ROOT / "results/classifiers_v1"
    neural = defaultdict(list)
    expected = {"initial_v1": set(), "rq34_v1": set(), "classifiers_v1": set()}

    for row in pd.read_csv(initial / "summary.csv").itertuples():
        directory = initial / f"{row.model}_{row.view}"
        prediction = directory / "predictions.csv"
        expected["initial_v1"].add(prediction)
        neural[directory / "checkpoint.pt"].append(prediction)

    for name in ("rq3_summary.json", "rq4_summary.json", "rq4_frozen_summary.json"):
        records = read_json(extensions / name)
        require(bool(records), f"Empty result index: {name}")
        for record in records:
            prediction = project_path(f"results/rq34_v1/{record['prediction_path']}")
            require(prediction not in expected["rq34_v1"], f"Duplicate indexed result: {prediction}")
            expected["rq34_v1"].add(prediction)
            neural[prediction.parent.parent / "checkpoint.pt"].append(prediction)

    comparison = pd.read_csv(classifiers / "test_comparison.csv")
    indexed = list(zip(comparison["split"], comparison["name"]))
    combinations = {(split, name) for split in SPLITS for name in registry}
    require(bool(registry), "The classifier registry is empty")
    require(len(indexed) == len(set(indexed)) and set(indexed) == combinations,
            "Classifier registry and test comparison index disagree")
    for split, name in combinations:
        expected["classifiers_v1"].add(classifiers / "evaluations" / split / name / "predictions.csv")

    for suite, paths in expected.items():
        found = set((ROOT / "results" / suite).rglob("predictions.csv"))
        require(bool(paths) and found == paths, f"Missing or unindexed prediction files in {suite}")
        for path in paths:
            for required in (path, path.parent / "metrics.json", path.parent / "confusion_matrix.csv"):
                require(required.is_file(), f"Missing artifact: {required.relative_to(ROOT)}")
    found_checkpoints = {path for suite in (initial, extensions) for path in suite.rglob("checkpoint.pt")}
    require(found_checkpoints == set(neural), "Missing or unindexed neural checkpoints")
    return expected, neural


def audit_predictions(expected, by_id, labels):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    counts = {}
    for suite, paths in expected.items():
        for path in sorted(paths):
            frame = pd.read_csv(path)
            require(not frame.empty and frame.sample_id.is_unique, f"Empty or duplicate predictions: {path}")
            require(set(frame.predicted) <= set(labels), f"Unknown predicted label: {path}")
            for row in frame.itertuples():
                original = by_id[row.sample_id]
                require(original["LLM_name"] == row.actual, f"Actual label mismatch: {path}")
                require(original["prompt_group"] == row.prompt_group, f"Prompt group mismatch: {path}")
            metrics = read_json(path.parent / "metrics.json")
            np.testing.assert_allclose(
                [accuracy_score(frame.actual, frame.predicted),
                 f1_score(frame.actual, frame.predicted, labels=labels, average="macro")],
                [metrics["accuracy"], metrics["macro_f1"]], rtol=0, atol=1e-12,
                err_msg=str(path.relative_to(ROOT)),
            )
            np.testing.assert_array_equal(
                confusion_matrix(frame.actual, frame.predicted, labels=labels),
                pd.read_csv(path.parent / "confusion_matrix.csv", index_col=0).to_numpy(),
                err_msg=str(path.relative_to(ROOT)),
            )
        counts[suite] = len(paths)
    return counts


def replay_neural(index, by_id):
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from experiment import LABELS, TextDataset, collate, evaluate
    from extended_experiments import loader
    from models import TextCNN, TextLSTM

    evaluations = 0
    for checkpoint, predictions in sorted(index.items()):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        config = saved["config"]
        model_type = {"cnn": TextCNN, "lstm": TextLSTM}[config["architecture"]]
        model = model_type(**config["model_kwargs"])
        model.load_state_dict(saved["state_dict"])
        for path in sorted(predictions):
            frame = pd.read_csv(path)
            rows = [by_id[sample_id] for sample_id in frame.sample_id]
            if checkpoint.is_relative_to(ROOT / "results/initial_v1"):
                dataset = TextDataset(rows, config["view"], saved["vocab"], config["tokens_per_field"])
                batches = DataLoader(dataset, batch_size=64, collate_fn=collate)
            else:
                condition = (path.parent.name.removeprefix("frozen_")
                             if path.parent.name.startswith("frozen_") else config["condition"])
                batches = loader(rows, saved["vocab"], condition)
            predicted = [LABELS[index] for index in evaluate(model, batches)[2]]
            require(predicted == frame.predicted.tolist(), f"Neural predictions differ: {path.relative_to(ROOT)}")
            evaluations += 1
    return {"checkpoints": len(index), "evaluations": evaluations}


def replay_classifiers(registry, original, confirmation):
    import joblib
    import numpy as np
    import pandas as pd
    import compare_classifiers as compare
    from experiment import LABELS

    partitions = {"original_test": [row for row in original if row["split"] == "test"],
                  "confirmation": confirmation}
    evaluations = 0
    for split, rows in partitions.items():
        cache = {}

        def scores_for(name):
            if name not in cache:
                spec = registry[name]
                if spec["kind"] == "base":
                    if "pipeline" in spec:
                        pipeline = joblib.load(project_path(spec["pipeline"]))
                        raw = pipeline.decision_function([compare.prefix_text(row) for row in rows])
                    else:
                        raw = compare.neural_scores(rows, project_path(spec["checkpoint"]))
                    cache[name] = compare.probabilities(raw, spec["temperature"])
                else:
                    members = [scores_for(member) for member in spec["members"]]
                    cache[name] = (compare.hard_vote(members) if spec["kind"] == "hard"
                                   else compare.weighted_average(members, spec["weights"]))
            return cache[name]

        for name in registry:
            scores = scores_for(name)
            path = ROOT / "results/classifiers_v1/evaluations" / split / name / "predictions.csv"
            frame = pd.read_csv(path)
            require(frame.sample_id.tolist() == [row["sample_id"] for row in rows], f"Row order differs: {path}")
            np.testing.assert_allclose(scores, frame[[f"p_{label}" for label in LABELS]].to_numpy(),
                                       rtol=1e-6, atol=1e-8, err_msg=str(path.relative_to(ROOT)))
            predicted = [LABELS[index] for index in scores.argmax(axis=1)]
            require(predicted == frame.predicted.tolist(), f"Classifier predictions differ: {path.relative_to(ROOT)}")
            evaluations += 1
    return {"candidates": len(registry), "evaluations": evaluations}


def verify(report):
    original, original_hash = read_dataset("processed")
    confirmation, confirmation_hash = read_dataset("confirmation_v1")
    by_id = {row["sample_id"]: row for row in original + confirmation}
    require(len(by_id) == len(original) + len(confirmation), "Duplicate dataset sample IDs")
    report["datasets"] = {"original": {"rows": len(original), "sha256": original_hash},
                          "confirmation": {"rows": len(confirmation), "sha256": confirmation_hash}}
    print("Both dataset hashes match their manifests.", flush=True)

    import pandas as pd
    import torch
    import compare_classifiers as compare
    from experiment import LABELS
    from extended_experiments import verify_design

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify_design()
    compare.verify()
    report["frozen_provenance"] = "passed"
    registry = read_json(ROOT / "results/classifiers_v1/registry.json")
    expected, neural = prediction_index(pd, registry)
    report["unit_tests"] = run_tests()
    print(f"Passed {report['unit_tests']['run']} tests.", flush=True)
    report["prediction_audits"] = audit_predictions(expected, by_id, LABELS)
    print(f"Audited {sum(report['prediction_audits'].values())} saved prediction files.", flush=True)
    report["neural_replay"] = replay_neural(neural, by_id)
    print(f"Replayed {report['neural_replay']['checkpoints']} neural checkpoints across "
          f"{report['neural_replay']['evaluations']} evaluations.", flush=True)
    report["classifier_replay"] = replay_classifiers(registry, original, confirmation)
    print(f"Replayed {report['classifier_replay']['candidates']} classifier/ensemble candidates across "
          f"{report['classifier_replay']['evaluations']} evaluations.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/verification.json", help="JSON result path within this project")
    args = parser.parse_args()
    output = project_path(args.output)
    started = time.monotonic()
    report = {"status": "running", "retrained": False}
    try:
        verify(report)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"Verification failed: {report['error']}", file=sys.stderr)
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verification {report['status']}. Details: {output.relative_to(ROOT)}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
