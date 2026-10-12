# Results

The tables and figures here summarize the recorded experiments. Original datasets, per-run predictions, checkpoints and other detailed run files are not distributed with this submission. These summaries can be inspected directly, but they do not support exact replay of the reported predictions on their own.

| Question | Main evidence |
| --- | --- |
| CNN/LSTM and prompt/response comparisons | [Initial scores](initial_v1/summary.csv) and `initial_v1/figures/` |
| Transfer between question answering and drafting | [Task-transfer scores](rq34_v1/rq3_aggregate.csv) and [transfer differences](rq34_v1/rq3_transfer_gaps.csv) |
| Information in words and punctuation | [Replacement experiments](rq34_v1/rq4_aggregate.csv) and [equal-length check](rq34_v1/rq4_fixed_length_aggregate.csv) |
| Individual classifiers and ensembles | [All test comparisons](classifiers_v1/test_comparison.csv) and [validation selection](classifiers_v1/selection_comparison.csv) |
| Examples of mistakes | [Selected examples](report_review_v1/error_examples.csv) |

Original test results use 375 responses. Confirmation results use 177 additional responses from the same source. RQ3/RQ4 averages use three training seeds; their reported standard deviations describe variation between those runs.

The selected ensemble combined CNN and LSTM averages with an SVM using weights of 25%, 25% and 50%. It scored 89.8% on confirmation, compared with 91.5% for the selected SVM. Settings were fixed before confirmation; the differences are too uncertain to establish a reliable advantage.

Later investigations were developed after viewing the original test results. The neural checkpoints had already used the full validation set before the later confidence-adjustment/selection split. These limits matter when interpreting the additional comparisons.

The [experiment instructions](../experiments/README.md) describe acquiring public data and training new models. Live data may differ from the original sample. Keep new results separate from these recorded summaries.
