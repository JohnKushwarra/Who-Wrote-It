# Running experiments

Run commands from the project root using the environment described in the main README. The examples below use `.\.venv\Scripts\python.exe` in PowerShell.

The submission includes summary outputs, not the original datasets, checkpoints or detailed run files. Analysis and saved-model verification require complete outputs from a local experiment; they cannot replay the reported results from the submitted files alone.

## Repeat the six required comparisons

Acquire data, then choose a new suite name so the supplied summaries remain unchanged:

```powershell
.\.venv\Scripts\python.exe -B src\prepare_data.py --rows-per-model 1500 --max-prompts 1000
.\.venv\Scripts\python.exe -B src\experiment.py --suite rerun_v1 --epochs 12
.\.venv\Scripts\python.exe -B src\analyze_results.py --suite rerun_v1
```

This trains CNN and LSTM models with prompt only, response only and prompt plus response. Skip acquisition if you already prepared the local dataset. Existing datasets and run directories are not overwritten.

## Run the full experiment sequence

The additional runners use fixed suite names. Create a separate working copy containing `src/`, `tests/`, `experiments/` and the dependency files, with an empty `results/` folder. Create its Python environment using the main README instructions. Do not copy the supplied summary result folders into this fresh run.

```powershell
.\.venv\Scripts\python.exe -B src\prepare_data.py --rows-per-model 1500 --max-prompts 1000
.\.venv\Scripts\python.exe -B src\prepare_confirmation.py
.\.venv\Scripts\python.exe -B src\experiment.py --suite initial_v1 --epochs 12
.\.venv\Scripts\python.exe -B src\analyze_results.py --suite initial_v1
.\.venv\Scripts\python.exe -B src\extended_experiments.py prepare
```

Before task-transfer training, inspect the prompts marked `audit_sample=True` in `results/rq34_v1/task_annotations.csv`. Record whether the question-answering/drafting labels are reasonable in `results/rq34_v1/TASK_AUDIT.md`. Review the new examples rather than copying a previous audit.

```powershell
.\.venv\Scripts\python.exe -B src\extended_experiments.py rq3
.\.venv\Scripts\python.exe -B src\extended_experiments.py rq4
.\.venv\Scripts\python.exe -B src\analyze_extensions.py
.\.venv\Scripts\python.exe -B src\compare_classifiers.py prepare
.\.venv\Scripts\python.exe -B src\compare_classifiers.py linear
.\.venv\Scripts\python.exe -B src\compare_classifiers.py neural
.\.venv\Scripts\python.exe -B src\compare_classifiers.py ensembles
.\.venv\Scripts\python.exe -B src\compare_classifiers.py evaluate
.\.venv\Scripts\python.exe -B src\analyze_classifiers.py
.\.venv\Scripts\python.exe -B src\analyze_errors.py
```

The classifier runner reads [CLASSIFIERS_ENSEMBLES_PLAN.md](CLASSIFIERS_ENSEMBLES_PLAN.md) when preparing a new suite. Selection uses validation data and is fixed before test evaluation.

Live downloads may change the data. Some checks and example-selection logic use the original test sizes. A changed sample requires reviewing those assumptions and reporting new results, rather than presenting the run as an exact reproduction.

## Check complete local outputs

The analysis scripts regenerate tables and plots from their corresponding complete local runs without retraining. `src/verify_results.py` checks data, predictions and saved models after the full sequence is complete. It requires all datasets, checkpoints and detailed run files; it is not a check to run on the submitted source-and-summary copy.

Saved designs record hashes of training and model source files. Even comment changes can invalidate these checks. Preserve the source associated with a completed run; do not replace recorded hashes to present edited source as the code that produced old results.
