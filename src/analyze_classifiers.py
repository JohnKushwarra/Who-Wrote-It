"""Audit saved comparison predictions and build tables/plots without fitting models."""
from datetime import datetime, timezone
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from experiment import ROOT, LABELS, write_json
from extended_experiments import load_rows, sha
from compare_classifiers import SUITE, verify
from prepare_confirmation import prefix_key
from prepare_data import normalize_prompt

DISPLAY = {
    'cnn_448':'Original CNN (seed 448)', 'cnn_449':'CNN seed 449', 'cnn_450':'CNN seed 450',
    'lstm_448':'Original LSTM (seed 448)', 'lstm_449':'LSTM seed 449', 'lstm_450':'LSTM seed 450',
    'cnn_seed_mean':'CNN: average of 3 seeds', 'lstm_seed_mean':'LSTM: average of 3 seeds',
    'linear_word_logistic':'Word TF-IDF + logistic', 'linear_word_svm':'Word TF-IDF + SVM',
    'linear_char_logistic':'Character TF-IDF + logistic', 'linear_char_svm':'Character TF-IDF + SVM',
    'linear_word_char_logistic':'Word + character TF-IDF + logistic', 'linear_word_char_svm':'Word + character TF-IDF + SVM',
    'neural_equal_single':'Original CNN + LSTM: equal vote', 'neural_tuned_single':'Original CNN + LSTM: tuned (CNN only)',
    'neural_equal_seed_means':'CNN/LSTM seed averages: equal vote',
    'neural_tuned_seed_means':'CNN/LSTM seed averages: 75/25 vote',
    'mixed_equal_cnn_linear':'CNN average + SVM: equal vote',
    'mixed_tuned_cnn_linear':'CNN average + SVM: tuned (50/50)',
    'mixed_equal_trio':'CNN/LSTM averages + SVM: equal vote',
    'mixed_tuned_trio':'CNN/LSTM averages + SVM: 25/25/50 [selected]',
    'mixed_hard_trio':'CNN/LSTM averages + SVM: hard vote',
    'linear_representation_mean':'Average of 3 linear representations',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def markdown_table(frame):
    cols=list(frame.columns)
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def grouped_interval(a,b,seed=448):
    assert list(a.sample_id)==list(b.sample_id)
    delta=((a.actual==a.predicted).astype(float)-(b.actual==b.predicted).astype(float))
    group_delta=delta.groupby(a.prompt_group).mean().to_numpy()
    rng=np.random.default_rng(seed)
    means=group_delta[rng.integers(0,len(group_delta),size=(10000,len(group_delta)))].mean(axis=1)
    low,high=np.quantile(means,[.025,.975])
    return dict(difference=float(group_delta.mean()),low=float(low),high=float(high),groups=len(group_delta))


def audit():
    verify()
    frozen=read(SUITE/'selection_frozen.json')
    assert sha(SUITE/'registry.json')==frozen['registry_sha256']
    assert sha(SUITE/'design.json')==frozen['design_sha256']
    for relative,digest in frozen['pipeline_hashes'].items():
        assert sha(ROOT/relative)==digest
    assert read(SUITE/'evaluation_started.json')['started_at_utc'] > frozen['frozen_at_utc']
    original=load_rows()
    new=[json.loads(x) for x in (ROOT/'data/confirmation_v1/dataset.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(new)==177 and len({r['prompt_group'] for r in new})==59
    for field,fn in [('LLM_Input',normalize_prompt),('LLM_output',normalize_prompt),('LLM_output',prefix_key)]:
        assert not ({fn(r[field]) for r in original} & {fn(r[field]) for r in new}),field
    assert not ({r['source_id'] for r in original} & {r['source_id'] for r in new})
    new_frame=pd.DataFrame(new)
    assert new_frame.groupby('prompt_group').LLM_name.apply(lambda x:sorted(x)==LABELS).all()
    assert new_frame.groupby('prompt_group').LLM_Input.nunique().eq(1).all()
    assert new_frame.groupby('prompt_group').source_id.nunique().eq(1).all()
    assert new_frame.LLM_name.value_counts().to_dict()=={label:59 for label in LABELS}
    assert len({normalize_prompt(r['LLM_Input']) for r in new})==59
    assert len({normalize_prompt(r['LLM_output']) for r in new})==177
    assert len({prefix_key(r['LLM_output']) for r in new})==177
    registry=read(SUITE/'registry.json')
    summary=pd.read_csv(SUITE/'test_comparison.csv')
    frames,recall,intervals={},[],[]
    for split,rows in [('original_test',[r for r in original if r['split']=='test']),('confirmation',new)]:
        y=np.asarray([LABELS.index(r['LLM_name']) for r in rows])
        cache={}
        for name,spec in registry.items():
            directory=SUITE/'evaluations'/split/name
            f=pd.read_csv(directory/'predictions.csv')
            assert list(f.sample_id)==[r['sample_id'] for r in rows]
            assert list(f.prompt_group)==[r['prompt_group'] for r in rows]
            assert list(f.actual)==[r['LLM_name'] for r in rows]
            p=f[[f'p_{label}' for label in LABELS]].to_numpy()
            assert np.allclose(p.sum(axis=1),1) and np.isfinite(p).all()
            if spec['kind']=='base':
                with np.load(SUITE/'scores'/name/f'{split}.npz') as z:
                    assert list(z['sample_ids'])==list(f.sample_id)
                    replay=softmax(z['scores']/spec['temperature'],axis=1)
            else:
                members=[cache[m] for m in spec['members']]
                if spec['kind']=='soft':
                    replay=np.tensordot(np.asarray(spec['weights']),np.stack(members),axes=(0,0))
                else:
                    votes=np.stack([m.argmax(axis=1) for m in members])
                    counts=np.stack([(votes==k).sum(axis=0) for k in range(3)],axis=1).astype(float)
                    tied=np.ptp(counts,axis=1)==0
                    counts[np.flatnonzero(tied),votes[0,tied]] += .001
                    replay=counts/counts.sum(axis=1,keepdims=True)
            np.testing.assert_allclose(p,replay,atol=1e-12)
            cache[name]=p
            pred=p.argmax(axis=1)
            assert list(f.predicted)==[LABELS[i] for i in pred]
            measured=dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=range(3),average='macro',zero_division=0),
                          log_loss=log_loss(y,p,labels=range(3)),brier=np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))
            saved=read(directory/'metrics.json')
            row=summary.loc[(summary.split==split)&(summary.name==name)].iloc[0]
            for k,v in measured.items():
                assert np.isclose(v,saved[k],atol=1e-12) and np.isclose(v,row[k],atol=1e-12)
            matrix=confusion_matrix(y,pred,labels=range(3))
            np.testing.assert_array_equal(matrix,pd.read_csv(directory/'confusion_matrix.csv',index_col=0).to_numpy())
            for i,label in enumerate(LABELS):
                recall.append(dict(split=split,name=name,family=label,recall=matrix[i,i]/matrix[i].sum(),support=int(matrix[i].sum())))
            frames[(split,name)]=f
        for reference in ['cnn_448','cnn_seed_mean',frozen['best_linear']]:
            intervals.append(dict(split=split,selected=frozen['selected_candidate'],reference=reference,
                                  **grouped_interval(frames[(split,frozen['selected_candidate'])],frames[(split,reference)])))
    pd.DataFrame(recall).to_csv(SUITE/'per_family_recall.csv',index=False)
    pd.DataFrame(intervals).to_csv(SUITE/'paired_accuracy_intervals.csv',index=False)
    verification=dict(audited_prediction_sets=len(frames),fixed_candidates=len(registry),confirmation_groups=59,confirmation_rows=177,
                      no_original_exact_prompt_response_prefix_or_source_id_overlap=True,confirmation_internal_duplicates=False,
                      probabilities_and_ensemble_formulas_replayed=True,metrics_and_confusions_recomputed=True,
                      selection_frozen_before_test=True,source_and_pipeline_hashes_verified=True,
                      original_neural_test_replays=read(SUITE/'evaluation_complete.json')['original_neural_test_replays'])
    write_json(SUITE/'verification.json',verification)
    return summary,registry,frozen,frames,pd.DataFrame(intervals)


def comparison_plot(summary,names,title,filename):
    fig,ax=plt.subplots(figsize=(12,6.4))
    y=np.arange(len(names))
    for offset,split,color,label in [(-.17,'original_test','#336b9a','Original exploratory test (375 responses)'),
                                      (.17,'confirmation','#d28c36','New same-source confirmation (177 responses)')]:
        data=summary.loc[summary.split==split].set_index('name')
        values=[100*data.loc[n,'accuracy'] for n in names]
        ax.barh(y+offset,values,height=.31,color=color,label=label)
        for yy,value in zip(y+offset,values):
            ax.text(value+.6,yy,f'{value:.1f}%',va='center',fontsize=9)
    ax.set_yticks(y,[DISPLAY[n] for n in names]);ax.invert_yaxis()
    ax.set_xlim(0,105);ax.set_xlabel('Accuracy (%)');ax.set_title(title,pad=16)
    ax.axvline(100/3,color='#888888',linestyle=':',linewidth=1)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower left',bbox_to_anchor=(0,-.22),frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(SUITE/'figures'/filename,dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    summary,registry,frozen,frames,intervals=audit()
    (SUITE/'figures').mkdir(exist_ok=True)
    comparison_plot(summary,['cnn_448','lstm_448',*[k for k,v in registry.items() if v['category']=='linear']],
                    'Linear classifiers under the same 128-token budget','linear_comparison.png')
    comparison_plot(summary,['cnn_448','lstm_448',*[k for k,v in registry.items() if v['category']=='neural_combination']],
                    'Combinations of existing neural checkpoints','neural_combinations.png')
    comparison_plot(summary,['cnn_seed_mean',frozen['best_linear'],*[k for k,v in registry.items() if v['category'] in ('mixed_ensemble','linear_ensemble')]],
                    'Predefined ensembles: confirmation of the frozen choices','ensemble_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,name in zip(axes,['cnn_448',frozen['selected_candidate']]):
        f=frames[('confirmation',name)]
        matrix=confusion_matrix(f.actual,f.predicted,labels=LABELS)
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=59)
        for (i,j),value in np.ndenumerate(matrix):
            ax.text(j,i,str(value),ha='center',va='center',color='white' if value>30 else 'black')
        ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS)
        ax.set_xlabel('Predicted family');ax.set_ylabel('Actual family')
        ax.set_title('Original CNN' if name=='cnn_448' else 'Validation-selected 25/25/50 ensemble')
    fig.suptitle('New confirmation batch: 59 responses per family');fig.tight_layout()
    fig.savefig(SUITE/'figures/confirmation_confusions.png',dpi=160,bbox_inches='tight');plt.close(fig)
    features=[]
    for name,spec in registry.items():
        if spec['category']!='linear': continue
        pipeline=joblib.load(ROOT/spec['pipeline'])
        names=pipeline.named_steps['features'].get_feature_names_out()
        clf=pipeline.named_steps['classifier']
        for cls,weights in zip(clf.classes_,clf.coef_):
            for idx in np.argsort(weights)[-10:][::-1]:
                features.append(dict(model=name,family=LABELS[cls],feature=names[idx],coefficient=weights[idx]))
    pd.DataFrame(features).to_csv(SUITE/'linear_top_coefficients.csv',index=False)
    table=summary.pivot(index='name',columns='split',values=['accuracy','macro_f1']).reindex(list(registry))
    display=pd.DataFrame({'Candidate':[DISPLAY[n] for n in table.index],
                          'Original accuracy':[f'{100*x:.1f}%' for x in table['accuracy']['original_test']],
                          'Confirmation accuracy':[f'{100*x:.1f}%' for x in table['accuracy']['confirmation']],
                          'Confirmation macro F1':[f'{x:.3f}' for x in table['macro_f1']['confirmation']]})
    lines=['# Linear classifiers and ensemble results','',
           'Results recomputed from the saved predictions for the original and confirmation tests.',
           '', '## Design and selection', '',
           '18 inexpensive fits: word, character and combined TF-IDF, each with logistic regression and LinearSVC at C = 0.1, 1, 10. Six fitted pipelines retained. The six existing CNN/LSTM checkpoints were reused; no new neural training occurred.', '',
           'All classifiers use the same first 128 name-masked response tokens. TF-IDF receives single-space-joined tokens rather than original whitespace. Training vocabulary/IDF and coefficients use 578 training prompt groups. The original validation groups are split into 62 calibration and 62 selection groups; existing neural epoch selection had previously used all 124 groups.', '',
           'The best linear model by validation is the word-plus-character SVM. It ties the word SVM on accuracy/F1 and wins the log-loss tie-break. Positive temperature scaling preserves each individual model\'s predicted class; it changes confidence and therefore can change ensemble predictions. SVM margin scaling is empirical calibration, not a guarantee of true posterior probabilities.', '',
           '**Frozen overall choice:** 25% CNN seed average + 25% LSTM seed average + 50% combined-feature SVM. It scored 172/186 correct on selection (92.5%), only one more correct than the selected linear model. The choice and artifacts were hashed before either new-suite test evaluation.', '',
           '## Complete fixed comparison', '',markdown_table(display),'',
           'The new batch contains 59 matched prompt groups / 177 balanced responses. Acquisition rows 1500–2499 contained only 500 examples/configuration from the intended source subset; filters and overlap checks reduced these to 59 groups. The minimum sample amendment occurred before fitting and changed no content or overlap filter. This is a small same-source convenience sample, not broad external validation.', '',
           '![Linear comparisons](figures/linear_comparison.png)','',
           '![Neural combinations](figures/neural_combinations.png)','',
           '![Ensemble comparisons](figures/ensemble_comparison.png)','',
           '## Interpretation', '',
           '- Linear models improve on the original CNN in these observed test comparisons at low fitting cost. Word/character representations capture useful signals without neural training. This does not identify a universal best classifier.',
           '- Neural averaging is mixed: the LSTM seed average improves substantially over the original LSTM, but CNN seed averaging does not beat the original CNN here. The validation-tuned original pair gives the LSTM zero weight.',
           '- The selected trio reaches 93.9% on the original test and 89.8% on confirmation. The selected single SVM reaches 93.9% and 91.5%. Thus the tiny validation gain from the more complex trio does not reproduce on confirmation.',
           '- The three-linear-model mean has the highest observed confirmation accuracy among the predefined candidates (92.7%, 164/177), while the equal CNN-average/SVM vote has the highest original-test accuracy (94.4%). Neither replaces the previously frozen overall choice. Promoting a test winner would require another independent assessment.',
           '- The confirmation batch is smaller and more strictly deduplicated than the original test; absolute scores need not match. Exact overlap checks do not establish absence of semantic near-duplicates.', '',
           '## Paired accuracy uncertainty', '',
           'Differences below are selected trio minus reference, resampling whole prompt groups 10,000 times. Intervals condition on fitted models and this source sample; they do not account for the broader model search or population shift.', '']
    interval_table=pd.DataFrame({'Test':intervals.split,'Reference':intervals.reference.map(DISPLAY),
                                 'Difference (pp)':intervals.difference.map(lambda x:f'{100*x:+.1f}'),
                                 '95% interval (pp)':[f'[{100*a:+.1f}, {100*b:+.1f}]' for a,b in zip(intervals.low,intervals.high)]})
    lines += [markdown_table(interval_table),'','![Confirmation confusion matrices](figures/confirmation_confusions.png)','',
              '## Calibration, coefficients and verification','',
              '`linear_calibration.csv` and `neural_calibration.csv` compare selection log loss/Brier before and after temperature scaling. Lower is better; calibration is assessed away from its fitting groups. Hard-vote fractions are vote scores, not calibrated probabilities; its log loss is not a probability-quality claim.', '',
              '`linear_top_coefficients.csv` exposes signed coefficients for the six linear pipelines. These are inspectable model parameters under correlated TF-IDF features, not causal word attributions or universal family fingerprints.', '',
              'All 48 prediction sets were independently audited for exact sample/label/group alignment, probability formulas, metrics and confusion counts. The six neural original-test predictions exactly match their saved prior predictions. No original data or model source was modified.', '',
              '']
    (SUITE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (SUITE/'source/analyze_classifiers.py').write_bytes((ROOT/'src/analyze_classifiers.py').read_bytes())
    write_json(SUITE/'analysis_provenance.json',dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                                                  source_sha256=sha(ROOT/'src/analyze_classifiers.py')))
    print(interval_table.to_string(index=False))
    print(json.dumps(read(SUITE/'verification.json'),indent=2))


if __name__=='__main__':
    main()
