"""Acquire a bounded new matched-prompt batch; never modify the original data."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import random
import urllib.parse

from experiment import ROOT, tokens, write_json
from prepare_data import DATASET, MODELS, digest, exclusion_reason, get_json, normalize_prompt

DEST = ROOT / 'data/confirmation_v1'


def fetch(config):
    directory = ROOT / 'data/raw/confirmation_v1' / config
    directory.mkdir(parents=True, exist_ok=True)
    rows, sources, truncated = [], [], 0
    for offset in range(1500, 2500, 100):
        url = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(
            dict(dataset=DATASET, config=config, split='train', offset=offset, length=100))
        path = directory / f'rows_{offset:06d}_100.json'
        if not path.exists():
            path.write_text(json.dumps(get_json(url), ensure_ascii=False), encoding='utf-8')
        payload = json.loads(path.read_text(encoding='utf-8'))
        sources.append(dict(url=url, path=str(path.relative_to(ROOT)), sha256=digest(path.read_bytes())))
        for item in payload['rows']:
            if item.get('truncated_cells'):
                truncated += 1
                continue
            row = item['row']
            assert row['model'] == config
            rows.append(row)
        print(f'Confirmation acquisition: {config}, offset {offset}', flush=True)
    return dict(rows=rows, sources=sources, truncated_rows=truncated)


def prefix_key(text):
    return ' '.join(tokens(text)[:128]).casefold()


def main():
    if DEST.exists():
        raise FileExistsError('Confirmation batch already exists; preserve it.')
    original_path = ROOT / 'data/processed/dataset.jsonl'
    original = [json.loads(line) for line in original_path.read_text(encoding='utf-8').splitlines()]
    seen_prompts = {normalize_prompt(r['LLM_Input']) for r in original}
    seen_outputs = {normalize_prompt(r['LLM_output']) for r in original}
    seen_prefixes = {prefix_key(r['LLM_output']) for r in original}
    old_ids = {r['source_id'] for r in original}
    metadata = get_json('https://huggingface.co/api/datasets/' + DATASET)
    with ThreadPoolExecutor(max_workers=3) as pool:
        payloads = dict(zip(MODELS, pool.map(fetch, MODELS)))
    lookups = {m: {r['custom_id']: r for r in p['rows']} for m, p in payloads.items()}
    assert all(len(lookups[m]) == len(payloads[m]['rows']) for m in MODELS)
    common = sorted(set.intersection(*(set(v) for v in lookups.values())))
    excluded, retained = Counter(), []
    for source_id in common:
        items = [lookups[m][source_id] for m in MODELS]
        prompts = [x['prompt'].strip() for x in items]
        outputs = [x['response'].strip() for x in items]
        if source_id in old_ids:
            excluded['original_source_id'] += 1
            continue
        if len(set(prompts)) != 1:
            excluded['prompt_mismatch'] += 1
            continue
        if {x['dataset'] for x in items} != {'DaringAnteater-prefs_olmo2_7b'}:
            excluded['other_source_subset'] += 1
            continue
        prompt = prompts[0]
        reason = exclusion_reason(prompt, outputs)
        if reason:
            excluded[reason] += 1
            continue
        pk = normalize_prompt(prompt)
        oks = {normalize_prompt(x) for x in outputs}
        prefixes = {prefix_key(x) for x in outputs}
        if pk in seen_prompts or oks & seen_outputs or prefixes & seen_prefixes or len(prefixes) != 3:
            excluded['original_or_batch_exact_overlap'] += 1
            continue
        seen_prompts.add(pk)
        seen_outputs.update(oks)
        seen_prefixes.update(prefixes)
        retained.append((source_id, prompt, items))
    random.Random(449).shuffle(retained)
    eligible = len(retained)
    retained = retained[:200]
    if len(retained) < 50:
        raise ValueError(f'Only {len(retained)} eligible groups; do not silently relax the frozen protocol.')
    rows = []
    for source_id, prompt, items in retained:
        gid = digest(normalize_prompt(prompt).encode())
        for model, item in zip(MODELS, items):
            rows.append(dict(sample_id=f'{gid[:16]}_{MODELS[model]}', prompt_group=gid,
                             source_id=source_id, source_dataset=item['dataset'], LLM_name=MODELS[model],
                             model_id=model, LLM_Input=prompt, LLM_output=item['response'].strip(), split='confirmation'))
    rows.sort(key=lambda r: (r['prompt_group'], r['LLM_name']))
    DEST.mkdir()
    path = DEST / 'dataset.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    manifest = dict(dataset=DATASET, retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                    repo_revision_observed=metadata.get('sha'), dataset_sha256=digest(path.read_bytes()),
                    original_data_sha256=digest(original_path.read_bytes()), source_offsets=[1500, 2499],
                    seed=449, eligible_groups=eligible, selected_groups=len(retained), selected_rows=len(rows),
                    exclusions=dict(excluded), shared_source_ids=len(common), models=MODELS,
                    acquisition_design_note='Before model fitting: the source slice contained only 500 same-subset rows/configuration and 59 eligible triples. Retain all eligible triples; minimum reduced from 100 to 50 without relaxing any content or overlap filters.',
                    raw_sources={m: {k:v for k,v in p.items() if k != 'rows'} for m,p in payloads.items()},
                    limitations='Viewer is not revision pinned; hashes identify bytes. Same-source convenience sample; semantic near-duplicates may remain.',
                    role='Confirmation only after classifier/ensemble selection is frozen; never fit or select with these labels.')
    write_json(DEST / 'manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k != 'raw_sources'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
