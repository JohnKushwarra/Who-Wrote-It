# Who Wrote It?

CMPSC 448 midterm project by John Kushwarra.

This project compares a CNN and an LSTM on identifying whether a response came from Gemma, Mistral or Qwen. The reported experiments used 2,481 responses to 827 matched prompts from Ai2's Dolci DPO Model Response Pool. Each model reads the first 128 tokens of the prompt, the response, or both.

[Read the report](report/CMPSC_448_Midterm_Report.pdf) for the experiment design and findings.

## Setup

Commands below use PowerShell from this folder with Python 3.12. The dependency lock records the scientific environment used for the project.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Without local experiment data, ten self-contained tests run and three dataset-dependent checks are skipped. Those additional checks need the prepared dataset; two also need the initial experiment configuration. No data is downloaded by the tests.

The submitted files include source code, tests, the PDF and summary results. Original datasets, downloaded API pages, saved models and detailed run files are not included. Exact replay of the reported predictions is therefore unavailable from this copy. The source can acquire public data and train new models; a live download may differ from the original sample.

## Train the required comparisons

These commands download public data and train all six CNN/LSTM comparisons. Use a new suite name so the supplied summaries remain unchanged.

```powershell
.\.venv\Scripts\python.exe -B src\prepare_data.py --rows-per-model 1500 --max-prompts 1000
.\.venv\Scripts\python.exe -B src\experiment.py --suite rerun_v1 --epochs 12
.\.venv\Scripts\python.exe -B src\analyze_results.py --suite rerun_v1
```

Data preparation refuses to replace an existing processed dataset, and training refuses to reuse an existing suite. See the [experiment instructions](experiments/README.md) for the additional comparisons and a full fresh run.

## Main results

- On the original response-only test, CNN accuracy was **91.7%** and LSTM accuracy was **73.1%**, against a **33.3%** baseline.
- Adding the prompt did not give a consistent improvement.
- On 177 additional responses, the selected TF-IDF SVM scored **91.5%** and the selected CNN/LSTM/SVM ensemble scored **89.8%**.

The task-transfer and text-pattern experiments are summarized in the report and [results notes](results/README.md). These results apply to the sampled source configurations and short response prefixes; they do not establish accuracy on other sources or all versions of these models.

## Files

| Location | Contents |
| --- | --- |
| `src/` | Data preparation, model definitions, training, analysis and verification |
| `tests/` | Checks for grouped splits, preprocessing, padding and classifier combinations |
| `data/` | [Dataset source and preparation notes](data/README.md) |
| `results/` | Recorded summary tables and plots |
| `report/` | The PDF report |
| `experiments/` | [Training commands](experiments/README.md) and the classifier comparison protocol |

The model definitions are in `src/models.py`. The six required comparisons run through `src/experiment.py`. Additional experiments use `src/extended_experiments.py` and `src/compare_classifiers.py`.
