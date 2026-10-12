"""Staged, validation-selected linear/neural/ensemble experiments with frozen tests.

Run prepare, linear, neural, ensembles, evaluate in that order. Each stage writes
new evidence and refuses to replace completed artifacts. Original suites stay intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import torch

from experiment import ROOT, LABELS, tokens, write_json
from extended_experiments import load_rows, loader, sha
from models import TextCNN, TextLSTM

SUITE = ROOT / 'results/classifiers_v1'
SEEDS = (448, 449, 450)
CS = (0.1, 1.0, 10.0)
REPS = ('word', 'char', 'word_char')
ESTIMATORS = ('logistic', 'svm')


def utc():
    return datetime.now(timezone.utc).isoformat()


def prefix_text(row):
    return ' '.join(tokens(row['LLM_output'])[:128])


def validation_groups(rows):
    groups = sorted({r['prompt_group'] for r in rows if r['split'] == 'validation'})
    random.Random(20261008).shuffle(groups)
    middle = len(groups) // 2
    return dict(calibration=sorted(groups[:middle]), selection=sorted(groups[middle:]))


def vectorizer(rep):
    common = dict(lowercase=False, min_df=2, sublinear_tf=True, dtype=np.float64)
    word = TfidfVectorizer(tokenizer=str.split, token_pattern=None, ngram_range=(1, 2), max_features=25000, **common)
    char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=40000, **common)
    features = word if rep == 'word' else char if rep == 'char' else FeatureUnion([('word', word), ('char', char)])
    return Pipeline([('tfidf', features), ('normalize', Normalizer())])


def probabilities(scores, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    return softmax(np.asarray(scores, dtype=float) / temperature, axis=1)


def fit_temperature(scores, y):
    result = minimize_scalar(lambda log_t: log_loss(y, probabilities(scores, np.exp(log_t)), labels=range(3)),
                             bounds=(np.log(0.05), np.log(20)), method='bounded', options={'xatol': 1e-6})
    if not result.success:
        raise RuntimeError('Temperature fitting failed')
    return float(np.exp(result.x))


def metrics(y, p):
    p = np.asarray(p)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid aligned class-probability matrix')
    pred = p.argmax(axis=1)
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=range(3), average='macro', zero_division=0)),
                log_loss=float(log_loss(y, p, labels=range(3))),
                brier=float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))))


def weighted_average(arrays, weights):
    weights = np.asarray(weights, dtype=float)
    if len(arrays) != len(weights) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Weights must be a nonnegative unit-sum vector')
    if len({x.shape for x in arrays}) != 1:
        raise ValueError('Ensemble arrays must be aligned')
    return sum(w * a for w, a in zip(weights, arrays))


def hard_vote(arrays):
    votes = np.stack([a.argmax(axis=1) for a in arrays], axis=1)
    counts = np.stack([(votes == i).sum(axis=1) for i in range(3)], axis=1)
    # Three distinct classes: the CNN (first member) breaks the tie without labels.
    tied = (counts == counts.max(axis=1, keepdims=True)).sum(axis=1) > 1
    counts = counts.astype(float)
    counts[np.flatnonzero(tied), votes[tied, 0]] += 0.001
    return counts / counts.sum(axis=1, keepdims=True)


def score_key(m, active=1):
    return (-m['macro_f1'], -m['accuracy'], m['log_loss'], active)


def tune_weights(arrays, y):
    denominator = 4
    options = [np.asarray(x, dtype=float) / denominator
               for x in itertools.product(range(denominator + 1), repeat=len(arrays)) if sum(x) == denominator]
    evidence = []
    for weights in options:
        evidence.append(dict(weights=weights.tolist(), **metrics(y, weighted_average(arrays, weights))))
    best = min(evidence, key=lambda r: (*score_key(r, sum(w > 0 for w in r['weights'])), tuple(r['weights'])))
    return best['weights'], evidence


def truth(rows):
    return np.asarray([LABELS.index(r['LLM_name']) for r in rows])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    if SUITE.exists():
        raise FileExistsError('Comparison suite already exists')
    rows = load_rows()
    confirmation = ROOT / 'data/confirmation_v1/dataset.jsonl'
    if not confirmation.exists():
        raise FileNotFoundError('Prepare confirmation batch before freezing protocol')
    SUITE.mkdir()
    (SUITE / 'source').mkdir()
    names = ['compare_classifiers.py', 'prepare_confirmation.py', 'experiment.py', 'models.py', 'extended_experiments.py']
    for name in names:
        (SUITE / 'source' / name).write_bytes((ROOT / 'src' / name).read_bytes())
    plan = ROOT / 'experiments/CLASSIFIERS_ENSEMBLES_PLAN.md'
    (SUITE / 'PROTOCOL.md').write_bytes(plan.read_bytes())
    groups = validation_groups(rows)
    write_json(SUITE / 'validation_groups.json', groups)
    checkpoints = {f'{arch}_{seed}': f'results/rq34_v1/rq4/seed_{seed}/{arch}/baseline/checkpoint.pt'
                   for arch in ('cnn', 'lstm') for seed in SEEDS}
    design = dict(created_at_utc=utc(), data_sha256=sha(ROOT / 'data/processed/dataset.jsonl'),
                  confirmation_sha256=sha(confirmation), groups_sha256=sha(SUITE / 'validation_groups.json'),
                  source_hashes={name: sha(ROOT / 'src' / name) for name in names},
                  checkpoint_paths=checkpoints, checkpoint_hashes={k:sha(ROOT / v) for k,v in checkpoints.items()},
                  C_grid=list(CS), representations=REPS, estimators=ESTIMATORS,
                  selection='Macro F1, accuracy; linear C tie smaller C; ensemble tie log loss then active components.',
                  validation_group_counts={k:len(v) for k,v in groups.items()},
                  limitations='Existing neural checkpoints used all original validation groups for epoch selection. Original test is exploratory; confirmation is same-source convenience sample.')
    write_json(SUITE / 'design.json', design)
    write_json(SUITE / 'registry.json', {})
    print('Protocol frozen; calibration/selection group counts:', design['validation_group_counts'], flush=True)


def verify():
    design = read_json(SUITE / 'design.json')
    assert sha(ROOT / 'data/processed/dataset.jsonl') == design['data_sha256']
    assert sha(ROOT / 'data/confirmation_v1/dataset.jsonl') == design['confirmation_sha256']
    assert sha(SUITE / 'validation_groups.json') == design['groups_sha256']
    for name, digest in design['source_hashes'].items():
        if sha(ROOT / 'src' / name) != digest:
            raise ValueError(f'Frozen execution source changed: {name}')
    for name, digest in design['checkpoint_hashes'].items():
        assert sha(ROOT / design['checkpoint_paths'][name]) == digest
    return design


def partitions():
    rows = load_rows()
    groups = read_json(SUITE / 'validation_groups.json')
    return {'train': [r for r in rows if r['split'] == 'train'],
            **{split:[r for r in rows if r['prompt_group'] in set(ids)] for split,ids in groups.items()}}


def save_scores(name, split, scores, rows):
    path = SUITE / 'scores' / name
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f'{split}.npz', scores=scores, sample_ids=np.asarray([r['sample_id'] for r in rows]))


def read_scores(name, split, rows):
    with np.load(SUITE / 'scores' / name / f'{split}.npz') as z:
        assert list(z['sample_ids']) == [r['sample_id'] for r in rows]
        return z['scores'].copy()


def selection_probabilities(registry, parts):
    cache = {}
    for name, spec in registry.items():
        if spec['kind'] == 'base':
            cache[name] = probabilities(read_scores(name, 'selection', parts['selection']), spec['temperature'])
        else:
            members = [cache[k] for k in spec['members']]
            cache[name] = hard_vote(members) if spec['kind'] == 'hard' else weighted_average(members, spec['weights'])
    return cache


def linear():
    verify()
    if (SUITE / 'linear_complete.json').exists():
        raise FileExistsError('Linear stage complete')
    parts = partitions()
    texts = {s:[prefix_text(r) for r in rows] for s, rows in parts.items()}
    ys = {s:truth(rows) for s,rows in parts.items()}
    registry = read_json(SUITE / 'registry.json')
    trials, calibrations = [], []
    for rep in REPS:
        t0 = time.perf_counter()
        features = vectorizer(rep)
        xtrain = features.fit_transform(texts['train'])
        matrices = {s:features.transform(texts[s]) for s in ('calibration', 'selection')}
        feature_seconds = time.perf_counter() - t0
        for estimator in ESTIMATORS:
            best = None
            for c in CS:
                model = (LogisticRegression(C=c, max_iter=1500, random_state=448) if estimator == 'logistic'
                         else LinearSVC(C=c, max_iter=15000, random_state=448))
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model.fit(xtrain, ys['train'])
                seconds = time.perf_counter() - t0
                assert list(model.classes_) == [0, 1, 2]
                scores = model.decision_function(matrices['selection'])
                m = metrics(ys['selection'], probabilities(scores))
                record = dict(representation=rep, estimator=estimator, C=c, features=xtrain.shape[1],
                              feature_seconds=feature_seconds, fit_seconds=seconds, **m)
                trials.append(record)
                print(f'Linear {rep}/{estimator} C={c}: selection F1={m["macro_f1"]:.4f}, fit={seconds:.2f}s', flush=True)
                key = (-m['macro_f1'], -m['accuracy'], c)
                if best is None or key < best[0]:
                    best = (key, model, record)
            _, model, record = best
            name = f'linear_{rep}_{estimator}'
            directory = SUITE / 'linear' / name
            directory.mkdir(parents=True, exist_ok=False)
            pipeline = Pipeline([('features', features), ('classifier', model)])
            joblib.dump(pipeline, directory / 'pipeline.joblib')
            restored = joblib.load(directory / 'pipeline.joblib')
            assert np.allclose(restored.decision_function(texts['selection']), model.decision_function(matrices['selection']))
            scores = {s:model.decision_function(matrices[s]) for s in ('calibration', 'selection')}
            temperature = fit_temperature(scores['calibration'], ys['calibration'])
            for split in scores:
                save_scores(name, split, scores[split], parts[split])
            raw = metrics(ys['selection'], probabilities(scores['selection']))
            calibrated = metrics(ys['selection'], probabilities(scores['selection'], temperature))
            calibration = dict(name=name, temperature=temperature, raw_selection_log_loss=raw['log_loss'],
                               scaled_selection_log_loss=calibrated['log_loss'], raw_selection_brier=raw['brier'],
                               scaled_selection_brier=calibrated['brier'])
            calibrations.append(calibration)
            registry[name] = dict(kind='base', category='linear', representation=rep, estimator=estimator,
                                  pipeline=str((directory / 'pipeline.joblib').relative_to(ROOT)), temperature=temperature,
                                  chosen_C=record['C'], **calibrated)
            write_json(directory / 'config.json', {**record, **calibration, 'labels':LABELS})
            write_json(SUITE / 'registry.json', registry)
            pd.DataFrame(trials).to_csv(SUITE / 'linear_trials.csv', index=False)
    pd.DataFrame(calibrations).to_csv(SUITE / 'linear_calibration.csv', index=False)
    write_json(SUITE / 'linear_complete.json', dict(completed_at_utc=utc(), fits=len(trials), selected_pipelines=6,
                                                  fit_seconds=sum(t['fit_seconds'] for t in trials)))


def neural_scores(rows, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cls = TextCNN if checkpoint['config']['architecture'] == 'cnn' else TextLSTM
    model = cls(**checkpoint['config']['model_kwargs'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    chunks = []
    with torch.no_grad():
        for x, lengths, _ in loader(rows, checkpoint['vocab']):
            chunks.append(model(x, lengths).numpy().astype(np.float64))
    return np.concatenate(chunks)


def add_blend(registry, cache, name, members, weights, y, category, kind='soft'):
    arrays = [cache[m] for m in members]
    p = hard_vote(arrays) if kind == 'hard' else weighted_average(arrays, weights)
    cache[name] = p
    registry[name] = dict(kind=kind, category=category, members=members, weights=weights, **metrics(y, p))


def neural():
    design = verify()
    assert (SUITE / 'linear_complete.json').exists()
    if (SUITE / 'neural_complete.json').exists():
        raise FileExistsError('Neural stage complete')
    registry, parts = read_json(SUITE / 'registry.json'), partitions()
    ys = {s:truth(rows) for s,rows in parts.items()}
    calibrations, replays = [], []
    for name, relative in design['checkpoint_paths'].items():
        scores = {s:neural_scores(parts[s], ROOT / relative) for s in ('calibration', 'selection')}
        temperature = fit_temperature(scores['calibration'], ys['calibration'])
        for split in scores:
            save_scores(name, split, scores[split], parts[split])
        # Check the full original validation metrics against the chosen epoch.
        joined_y = np.concatenate([ys[s] for s in ('calibration', 'selection')])
        joined_p = probabilities(np.concatenate([scores[s] for s in ('calibration', 'selection')]))
        full = metrics(joined_y, joined_p)
        directory = (ROOT / relative).parent
        epoch = read_json(directory / 'training.json')['best_epoch']
        history = next(h for h in read_json(directory / 'history.json') if h['epoch'] == epoch)
        for key in ('accuracy', 'macro_f1'):
            assert np.isclose(full[key], history['validation_' + key]), (name, key)
        assert np.isclose(full['log_loss'], history['validation_loss'], atol=1e-6)
        replays.append(dict(name=name, best_epoch=epoch, full_validation_replay=True))
        raw = metrics(ys['selection'], probabilities(scores['selection']))
        scaled = metrics(ys['selection'], probabilities(scores['selection'], temperature))
        calibrations.append(dict(name=name, temperature=temperature, raw_selection_log_loss=raw['log_loss'],
                                 scaled_selection_log_loss=scaled['log_loss'], raw_selection_brier=raw['brier'],
                                 scaled_selection_brier=scaled['brier']))
        registry[name] = dict(kind='base', category='neural_single', checkpoint=relative, temperature=temperature, **scaled)
    cache = selection_probabilities(registry, parts)
    y = ys['selection']
    searches = {}
    for arch in ('cnn', 'lstm'):
        members = [f'{arch}_{seed}' for seed in SEEDS]
        add_blend(registry, cache, f'{arch}_seed_mean', members, [1/3]*3, y, 'neural_combination')
    for suffix, members in [('single', ['cnn_448', 'lstm_448']), ('seed_means', ['cnn_seed_mean', 'lstm_seed_mean'])]:
        add_blend(registry, cache, f'neural_equal_{suffix}', members, [.5, .5], y, 'neural_combination')
        weights, trials = tune_weights([cache[m] for m in members], y)
        searches[f'neural_tuned_{suffix}'] = trials
        add_blend(registry, cache, f'neural_tuned_{suffix}', members, weights, y, 'neural_combination')
    write_json(SUITE / 'neural_weight_search.json', searches)
    pd.DataFrame(calibrations).to_csv(SUITE / 'neural_calibration.csv', index=False)
    write_json(SUITE / 'registry.json', registry)
    write_json(SUITE / 'neural_complete.json', dict(completed_at_utc=utc(), checkpoint_replays=replays, new_training_runs=0))
    print(pd.DataFrame([dict(name=k, **v) for k,v in registry.items() if v['category'] != 'linear'])[['name','accuracy','macro_f1']].to_string(index=False), flush=True)


def overlap_table(cache, y):
    records = []
    for a,b in itertools.combinations(cache, 2):
        pa,pb = cache[a].argmax(axis=1), cache[b].argmax(axis=1)
        ca,cb = pa == y, pb == y
        records.append(dict(a=a,b=b,both_correct=int((ca&cb).sum()),a_only_correct=int((ca&~cb).sum()),
                            b_only_correct=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),disagreements=int((pa!=pb).sum())))
    return records


def ensembles():
    verify()
    assert (SUITE / 'neural_complete.json').exists()
    if (SUITE / 'selection_frozen.json').exists():
        raise FileExistsError('Selection already frozen')
    registry, parts = read_json(SUITE / 'registry.json'), partitions()
    cache, y = selection_probabilities(registry, parts), truth(parts['selection'])
    linear_names = [k for k,v in registry.items() if v['category'] == 'linear']
    best = min(linear_names, key=lambda k:(*score_key(registry[k]),k))
    best_reps = [min([k for k in linear_names if registry[k]['representation'] == rep],
                     key=lambda k:(*score_key(registry[k]),k)) for rep in REPS]
    pd.DataFrame(overlap_table({k:cache[k] for k in ['cnn_seed_mean','lstm_seed_mean',*linear_names]}, y)).to_csv(SUITE/'selection_error_overlap.csv',index=False)
    searches = {}
    for label,members in [('cnn_linear',['cnn_seed_mean',best]), ('trio',['cnn_seed_mean','lstm_seed_mean',best])]:
        add_blend(registry,cache,f'mixed_equal_{label}',members,[1/len(members)]*len(members),y,'mixed_ensemble')
        weights,trials = tune_weights([cache[m] for m in members],y)
        searches[f'mixed_tuned_{label}'] = trials
        add_blend(registry,cache,f'mixed_tuned_{label}',members,weights,y,'mixed_ensemble')
    add_blend(registry,cache,'mixed_hard_trio',['cnn_seed_mean','lstm_seed_mean',best],None,y,'mixed_ensemble',kind='hard')
    add_blend(registry,cache,'linear_representation_mean',best_reps,[1/3]*3,y,'linear_ensemble')
    def count_leaves(name):
        spec=registry[name]
        if spec['kind']=='base':
            return {name}
        weights = spec['weights'] if spec['kind']=='soft' else [1]*len(spec['members'])
        return set().union(*(count_leaves(m) for m,w in zip(spec['members'],weights) if w>0))
    selected = min(registry, key=lambda k:(*score_key(registry[k],len(count_leaves(k))),k))
    write_json(SUITE/'ensemble_weight_search.json',searches)
    write_json(SUITE/'registry.json',registry)
    pd.DataFrame([dict(name=k,category=v['category'],**{m:v[m] for m in ('accuracy','macro_f1','log_loss','brier')},
                       members=json.dumps(v.get('members',[])),weights=json.dumps(v.get('weights')),active_models=len(count_leaves(k)))
                  for k,v in registry.items()]).to_csv(SUITE/'selection_comparison.csv',index=False)
    artifact_hashes = {str(p.relative_to(ROOT)):sha(p) for p in (SUITE/'linear').rglob('pipeline.joblib')}
    frozen = dict(frozen_at_utc=utc(),selected_candidate=selected,best_linear=best,best_linear_per_representation=best_reps,
                  registry_sha256=sha(SUITE/'registry.json'),design_sha256=sha(SUITE/'design.json'),pipeline_hashes=artifact_hashes,
                  candidate_count=len(registry),selection_rows=len(y),selection_groups=len(y)//3,
                  test_results_consulted_for_selection=False,
                  rule='Prespecified selection macro F1, accuracy, log loss, active model count and name. No changes after tests.')
    write_json(SUITE/'selection_frozen.json',frozen)
    print(json.dumps(frozen,indent=2),flush=True)


def evaluate():
    design = verify()
    frozen = read_json(SUITE/'selection_frozen.json')
    assert sha(SUITE/'registry.json')==frozen['registry_sha256']
    for path,digest in frozen['pipeline_hashes'].items():
        assert sha(ROOT/path)==digest
    if (SUITE/'evaluation_complete.json').exists() or (SUITE/'evaluation_started.json').exists():
        raise FileExistsError('Test evaluation already started; inspect artifacts instead of rerunning selection')
    write_json(SUITE/'evaluation_started.json',dict(started_at_utc=utc(),selection_frozen_sha256=sha(SUITE/'selection_frozen.json')))
    registry=read_json(SUITE/'registry.json')
    old=load_rows()
    confirmation=[json.loads(line) for line in (ROOT/'data/confirmation_v1/dataset.jsonl').read_text(encoding='utf-8').splitlines()]
    sets={'original_test':[r for r in old if r['split']=='test'],'confirmation':confirmation}
    records=[]
    for split,rows in sets.items():
        y,cache=truth(rows),{}
        texts=[prefix_text(r) for r in rows]
        for name,spec in registry.items():
            start=time.perf_counter()
            if spec['kind']=='base':
                scores = (joblib.load(ROOT/spec['pipeline']).decision_function(texts) if 'pipeline' in spec
                          else neural_scores(rows,ROOT/spec['checkpoint']))
                p=probabilities(scores,spec['temperature'])
                save_scores(name,split,scores,rows)
                if split=='original_test' and 'checkpoint' in spec:
                    reference=pd.read_csv((ROOT/spec['checkpoint']).parent/'test/predictions.csv')
                    assert list(reference.sample_id)==[r['sample_id'] for r in rows]
                    assert list(reference.predicted)==[LABELS[i] for i in p.argmax(axis=1)]
            else:
                arrays=[cache[m] for m in spec['members']]
                p=hard_vote(arrays) if spec['kind']=='hard' else weighted_average(arrays,spec['weights'])
            cache[name]=p
            seconds=time.perf_counter()-start
            directory=SUITE/'evaluations'/split/name
            directory.mkdir(parents=True,exist_ok=False)
            pred=p.argmax(axis=1)
            frame=pd.DataFrame(dict(sample_id=[r['sample_id'] for r in rows],prompt_group=[r['prompt_group'] for r in rows],
                                    actual=[LABELS[i] for i in y],predicted=[LABELS[i] for i in pred],
                                    **{f'p_{label}':p[:,i] for i,label in enumerate(LABELS)}))
            frame.to_csv(directory/'predictions.csv',index=False)
            m=metrics(y,p)
            matrix=confusion_matrix(y,pred,labels=range(3))
            pd.DataFrame(matrix,index=LABELS,columns=LABELS).to_csv(directory/'confusion_matrix.csv',index_label='actual/predicted')
            write_json(directory/'metrics.json',dict(**m,rows=len(rows),groups=len(rows)//3,
                       per_family_recall={label:float(matrix[i,i]/matrix[i].sum()) for i,label in enumerate(LABELS)}))
            records.append(dict(split=split,name=name,category=spec['category'],selected=name==frozen['selected_candidate'],
                                rows=len(rows),**m,scoring_seconds=seconds))
        print(f'Evaluated {len(cache)} frozen candidates on {split}: {len(rows)} responses',flush=True)
    pd.DataFrame(records).to_csv(SUITE/'test_comparison.csv',index=False)
    write_json(SUITE/'evaluation_complete.json',dict(completed_at_utc=utc(),evaluations=len(records),
                                                  frozen_selection_sha256=sha(SUITE/'selection_frozen.json'),
                                                  original_neural_test_replays=6))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','linear','neural','ensembles','evaluate'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        globals()[args.stage]()


if __name__=='__main__':
    main()
