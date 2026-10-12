# Dataset

Source: Ai2's [Dolci DPO Model Response Pool](https://huggingface.co/datasets/allenai/Dolci-DPO-Model-Response-Pool), accessed October 8, 2026. The source requests attribution to Team Olmo et al., *Olmo 3* (2025), arXiv:2512.13961. Its card lists ODC-BY and links terms for the individual model outputs.

The labels are the source configurations `gemma3-12b`, `mistral-24b` and `qwen3-no_reasoning-4b`. The reported sample used existing public responses, not responses newly generated for this project.

## Original preparation

The first 1,500 rows per configuration were matched by source ID and identical prompt text. Entire groups were excluded for length limits, selected extraction/schema tasks or an email pattern. This left 827 prompts with one response per family: 2,481 rows from the `DaringAnteater-prefs_olmo2_7b` subset.

The split keeps all three responses to a prompt together: 578 training groups, 124 validation groups and 125 test groups. Model-name strings are replaced with a shared placeholder. The vocabulary is learned only from training data.

The separate confirmation sample uses source rows 1,500-2,499, applies the same filters and removes exact overlaps with the original data. It contained 59 matched groups, or 177 responses.

## Acquire data locally

The original datasets and cached downloads are not distributed with this submission. From the project root, run:

```powershell
.\.venv\Scripts\python.exe -B src\prepare_data.py --rows-per-model 1500 --max-prompts 1000
.\.venv\Scripts\python.exe -B src\prepare_confirmation.py
```

The first command creates `data/processed/dataset.jsonl` and its manifest. The second creates `data/confirmation_v1/dataset.jsonl` and its manifest; it requires the first dataset. Public API downloads are cached under `data/raw/`.

Each dataset row contains `LLM_name`, `LLM_Input` and `LLM_output`, along with identifiers and split information. The generated manifests record acquisition URLs, exclusions and dataset hashes.

This is a convenience sample, not a random or representative sample of LLM writing. Filtering is not a complete privacy/content review, and similar examples with different wording may remain. The live Dataset Viewer is not revision-pinned. A new download may produce different examples or counts and is not an exact reconstruction of the reported dataset.
