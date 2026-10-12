"""Select reproducible illustrative cases from existing test predictions; never fit."""
from pathlib import Path
import hashlib
import json

import pandas as pd

from experiment import ROOT, tokens


def main():
    dest = ROOT / 'results/report_review_v1'
    dest.mkdir(exist_ok=True)
    data_path = ROOT / 'data/processed/dataset.jsonl'
    rows = {r['sample_id']: r for r in map(json.loads, data_path.read_text(encoding='utf-8').splitlines())}
    paths = {
        'cnn': ROOT / 'results/initial_v1/cnn_output/predictions.csv',
        'lstm': ROOT / 'results/initial_v1/lstm_output/predictions.csv',
        'svm': ROOT / 'results/classifiers_v1/evaluations/original_test/linear_word_char_svm/predictions.csv',
    }
    frame = None
    for model, path in paths.items():
        part = pd.read_csv(path).rename(columns={'predicted': model})
        for row in part.itertuples():
            source = rows[row.sample_id]
            assert source['split'] == 'test' and source['LLM_name'] == row.actual
            assert source['prompt_group'] == row.prompt_group
        frame = part if frame is None else frame.merge(part, on=['sample_id', 'prompt_group', 'actual'], validate='one_to_one')
    assert len(frame) == 375
    frame = frame.sort_values('sample_id')
    rules = [
        ('cnn_qwen_as_mistral', (frame.actual == 'Qwen') & (frame.cnn == 'Mistral')),
        ('cnn_mistral_as_qwen', (frame.actual == 'Mistral') & (frame.cnn == 'Qwen')),
        ('lstm_error_cnn_correct', (frame.actual == 'Mistral') & (frame.lstm == 'Qwen') & (frame.cnn == frame.actual)),
        ('all_three_wrong', (frame.cnn != frame.actual) & (frame.lstm != frame.actual) & (frame.svm != frame.actual)),
        ('all_three_correct', (frame.cnn == frame.actual) & (frame.lstm == frame.actual) & (frame.svm == frame.actual)),
    ]
    vocab = json.loads((ROOT / 'results/initial_v1/vocab.json').read_text())
    examples, used = [], set()
    for category, mask in rules:
        eligible = frame[mask & ~frame.sample_id.isin(used)]
        row = eligible.iloc[0]
        used.add(row.sample_id)
        source = rows[row.sample_id]
        prefix = tokens(source['LLM_output'])[:128]
        examples.append(dict(category=category, eligible_remaining=len(eligible), sample_id=row.sample_id,
                             prompt_group=row.prompt_group, source_id=source['source_id'], actual=row.actual,
                             cnn=row.cnn, lstm=row.lstm, svm=row.svm, visible_token_count=len(prefix),
                             unknown_neural_tokens=sum(token not in vocab for token in prefix),
                             visible_tokens=prefix))
    summary = dict(test_rows=375, test_groups=125,
                   errors={m: int((frame[m] != frame.actual).sum()) for m in paths},
                   all_three_wrong=int(((frame.cnn != frame.actual) & (frame.lstm != frame.actual) & (frame.svm != frame.actual)).sum()),
                   all_three_correct=int(((frame.cnn == frame.actual) & (frame.lstm == frame.actual) & (frame.svm == frame.actual)).sum()),
                   selection='In listed category order, choose smallest sample_id not already selected. Post-hoc illustrations, not a random or representative sample.',
                   interpretation='Visible token strings precede vocabulary lookup; unknown_neural_tokens counts strings mapped to UNK. Whitespace is not preserved. No example diagnoses a causal failure mechanism.',
                   hashes={str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [data_path, *paths.values(), ROOT / 'results/initial_v1/vocab.json', Path(__file__)]})
    (dest / 'error_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
    (dest / 'error_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame([{k: v for k, v in e.items() if k != 'visible_tokens'} for e in examples]).to_csv(dest / 'error_examples.csv', index=False)
    print(pd.DataFrame(examples).drop(columns=['visible_tokens', 'prompt_group', 'source_id']).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'hashes'}, indent=2))


if __name__ == '__main__':
    main()
