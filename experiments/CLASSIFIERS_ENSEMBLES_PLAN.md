# Classifier and ensemble comparison

This describes the comparison implemented by `src/compare_classifiers.py`. The runner saves a copy of this file when preparing a new suite.

## Inputs and selection

Use the same model-name replacement and first 128 response tokens as the neural experiments. Preserve the original 578 training groups. Divide the 124 validation groups into 62 groups for adjusting confidence scores and 62 for selecting settings. All responses to a prompt remain together.

Fit word/punctuation 1-2-gram TF-IDF, character 3-5-gram TF-IDF and their combined representation. Compare logistic regression and linear SVMs with C values of 0.1, 1 and 10. Vocabulary and feature weights use training data only. Select each model's C by validation macro F1, then accuracy, then smaller C.

Adjust each selected model's class scores with a positive temperature fitted on the calibration groups. This preserves its individual predicted classes; it does not guarantee perfectly calibrated probabilities.

## Combinations

Reuse CNN and LSTM models trained with seeds 448, 449 and 450. Compare individual models, averages within each architecture, CNN/LSTM combinations, mixed neural/linear combinations and an average of the best model for each linear representation.

Choose weights from increments of 0.25 using selection data. Zero weights are allowed. Rank choices by macro F1, accuracy, log loss and fewer active components, with deterministic tie-breaking. Hard voting uses the CNN prediction when all three members disagree.

Freeze the candidates, confidence adjustments, weights and selected winner before evaluating the original test and confirmation sample. Retain all candidate results, including combinations that do not improve accuracy.

## Confirmation and limits

The recorded confirmation sample contains 59 matched prompt groups (177 responses) from the same dataset, after excluding exact overlaps. It was not used for fitting or selection.

The initial test had already informed follow-up work. Neural checkpoint selection had used the full validation set before it was divided for this comparison. A small sample from the same source does not establish performance on new sources or model families.
