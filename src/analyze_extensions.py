"""Audit extended predictions and produce RQ3/RQ4 tables, figures and interpretations."""
from collections import Counter
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch

from experiment import ROOT, LABELS, evaluate, tokens, write_json
from extended_experiments import SUITE, SEEDS, TASKS, CONDITIONS, WORD, load_rows, loader, verify_design
from models import TextCNN, TextLSTM


def audit_predictions(record, rows):
    path=SUITE/record["prediction_path"]
    pred=pd.read_csv(path)
    expected={r["sample_id"]:r for r in rows}
    if len(pred)!=len(rows) or pred.sample_id.duplicated().any() or set(pred.sample_id)!=set(expected):
        raise AssertionError(f"Wrong evaluation sample IDs: {path}")
    for r in pred.itertuples():
        source=expected[r.sample_id]
        if r.actual!=source["LLM_name"] or r.prompt_group!=source["prompt_group"] or r.predicted not in LABELS:
            raise AssertionError(f"Prediction labels/groups invalid: {path}")
    accuracy=accuracy_score(pred.actual,pred.predicted)
    f1=f1_score(pred.actual,pred.predicted,labels=LABELS,average="macro",zero_division=0)
    if not np.allclose([accuracy,f1],[record["accuracy"],record["macro_f1"]],atol=1e-12):
        raise AssertionError(f"Metric mismatch: {path}")
    matrix=confusion_matrix(pred.actual,pred.predicted,labels=LABELS)
    saved=pd.read_csv(path.parent/"confusion_matrix.csv",index_col=0)
    if list(saved.columns)!=LABELS or list(saved.index)!=LABELS or not np.array_equal(matrix,saved.to_numpy()):
        raise AssertionError(f"Confusion mismatch: {path}")
    pred["correct"]=(pred.actual==pred.predicted).astype(float)
    return pred


def summarize(frame,keys):
    return frame.groupby(keys,sort=False).agg(accuracy_mean=("accuracy","mean"),accuracy_sd=("accuracy","std"),
        macro_f1_mean=("macro_f1","mean"),macro_f1_sd=("macro_f1","std"),seeds=("seed","nunique")).reset_index()


def text_features(row):
    raw=row["LLM_output"]
    full=tokens(raw)
    prefix=full[:128]
    words=[t for t in prefix if WORD.fullmatch(t)]
    alpha=[c for t in prefix for c in t if c.isalpha()]
    all_words=[t.casefold() for t in full if WORD.fullmatch(t)]
    return {"sample_id":row["sample_id"],"prompt_group":row["prompt_group"],"family":row["LLM_name"],
        "full_characters":len(raw),"full_tokens":len(full),"prefix_token_count":len(prefix),
        "prefix_punctuation_fraction":sum(not bool(WORD.fullmatch(t)) for t in prefix)/max(len(prefix),1),
        "prefix_uppercase_fraction":sum(c.isupper() for c in alpha)/max(len(alpha),1),
        "prefix_mean_word_length":np.mean([len(w) for w in words]) if words else float("nan"),
        "ttr_first_50_words":len(set(all_words[:50]))/50 if len(all_words)>=50 else float("nan"),
        "full_markdown_heading":int(bool(re.search(r"(?m)^#{1,6}\s",raw))),
        "prefix_markdown_hash_count":prefix.count("#")}


def lexical_markers(training):
    family_counts={label:Counter() for label in LABELS}
    n={label:0 for label in LABELS}
    for row in training:
        family=row["LLM_name"]
        words={t.casefold() for t in tokens(row["LLM_output"])[:128] if t.isalpha() and len(t)>=3}
        family_counts[family].update(words-ENGLISH_STOP_WORDS)
        n[family]+=1
    results=[]
    for family in LABELS:
        rest=sum((family_counts[l] for l in LABELS if l!=family),Counter())
        other_n=sum(n[l] for l in LABELS if l!=family)
        values=[]
        for term,count in family_counts[family].items():
            if count<25:
                continue
            p=(count+1)/(n[family]+2)
            q=(rest[term]+1)/(other_n+2)
            values.append({"family":family,"term":term,"documents":count,"family_document_fraction":count/n[family],
                           "other_document_fraction":rest[term]/other_n,"smoothed_log2_ratio":np.log2(p/q)})
        results.extend(sorted(values,key=lambda r:(-r["smoothed_log2_ratio"],r["term"]))[:10])
    return pd.DataFrame(results)


def task_gap_intervals(frame,predictions):
    rng=np.random.default_rng(448)
    records=[]
    for architecture in ("cnn","lstm"):
        for target in TASKS:
            source=next(t for t in TASKS if t!=target)
            seed_deltas=[]
            group_deltas=[]
            for seed in SEEDS:
                def match(task):
                    return frame[(frame.architecture==architecture)&(frame.seed==seed)&(frame.train_task==task)&(frame.test_task==target)].iloc[0]
                inside,outside=match(target),match(source)
                seed_deltas.append(float(outside.accuracy-inside.accuracy))
                a=predictions[inside.prediction_path]
                b=predictions[outside.prediction_path]
                pairs=a.merge(b,on=["sample_id","prompt_group","actual"],suffixes=("_in","_out"),validate="one_to_one")
                groups=(pairs.assign(delta=pairs.correct_out-pairs.correct_in).groupby("prompt_group").delta.mean())
                group_deltas.append(groups)
            # Same target prompts across seeds; average seed-specific differences, then resample groups.
            means=pd.concat(group_deltas,axis=1).mean(axis=1).to_numpy()
            draws=rng.choice(means,size=(2000,len(means)),replace=True).mean(axis=1)
            records.append({"architecture":architecture,"target_task":target,"transfer_source":source,
                "transfer_minus_within_accuracy":float(np.mean(seed_deltas)),"difference_sd_across_seeds":float(np.std(seed_deltas,ddof=1)),
                "prompt_bootstrap_low":float(np.quantile(draws,0.025)),"prompt_bootstrap_high":float(np.quantile(draws,0.975)),
                "test_prompt_groups":len(means),"interval_scope":"Conditional on these six trained models; not a population or seed-generalization guarantee."})
    return pd.DataFrame(records)


def validation_overlap(training_rows):
    predictions={}
    for name,cls in (("cnn",TextCNN),("lstm",TextLSTM)):
        saved=torch.load(SUITE/f"rq4/seed_448/{name}/baseline/checkpoint.pt",weights_only=True,map_location="cpu")
        model=cls(**saved["config"]["model_kwargs"])
        model.load_state_dict(saved["state_dict"])
        metrics,true,pred=evaluate(model,loader(training_rows,saved["vocab"]))
        predictions[name]=np.array(pred)==np.array(true)
    a,b=predictions["cnn"],predictions["lstm"]
    return {"split":"original validation only","seed":448,"responses":len(a),"prompt_groups":len(a)//3,
            "both_correct":int((a&b).sum()),"cnn_only_correct":int((a&~b).sum()),"lstm_only_correct":int((~a&b).sum()),
            "both_wrong":int((~a&~b).sum()),"purpose":"Error-complementarity evidence for discussion; no ensemble trained, fitted or selected."}


def main():
    torch.set_num_threads(2)
    verify_design()
    rows=load_rows()
    partitions=json.loads((SUITE/"rq3_partitions.json").read_text(encoding="utf-8"))
    rq3=pd.read_json(SUITE/"rq3_summary.json")
    rq4=pd.read_json(SUITE/"rq4_summary.json")
    frozen=pd.read_json(SUITE/"rq4_frozen_summary.json")
    assert set(zip(rq3.architecture,rq3.seed,rq3.train_task,rq3.test_task))=={(a,s,x,y) for a in ("cnn","lstm") for s in SEEDS for x in TASKS for y in TASKS} and len(rq3)==24
    assert set(zip(rq4.architecture,rq4.seed,rq4.condition))=={(a,s,c) for a in ("cnn","lstm") for s in SEEDS for c in CONDITIONS} and len(rq4)==18
    assert set(zip(frozen.architecture,frozen.seed,frozen.condition))=={(a,s,c) for a in ("cnn","lstm") for s in SEEDS for c in (*CONDITIONS[1:],"order_shuffled")} and len(frozen)==18
    predictions={}
    class_scores=[]
    for kind,frame in (("rq3",rq3),("rq4",rq4),("frozen",frozen)):
        for record in frame.to_dict("records"):
            expected=partitions[record["test_task"]]["test"] if kind=="rq3" else [r for r in rows if r["split"]=="test"]
            pred=audit_predictions(record,expected)
            predictions[record["prediction_path"]]=pred
            for label in LABELS:
                selected=pred[pred.actual==label]
                class_scores.append({"investigation":kind,**{k:record[k] for k in ("architecture","seed","train_task","test_task","condition") if k in record},
                    "family":label,"recall":float(selected.correct.mean()),"support":len(selected)})
    summaries={"rq3":summarize(rq3,["architecture","train_task","test_task"]),
               "rq4":summarize(rq4,["architecture","condition"]),
               "frozen":summarize(frozen,["architecture","condition"])}
    for name,frame in summaries.items(): frame.to_csv(SUITE/f"{name}_aggregate.csv",index=False)
    for name,frame in (("rq3",rq3),("rq4",rq4),("rq4_frozen",frozen)): frame.to_csv(SUITE/f"{name}_runs.csv",index=False)
    pd.DataFrame(class_scores).to_csv(SUITE/"per_family_recall.csv",index=False)
    # Diagnostic subgroup, not model selection: eliminate visible sequence-length variation.
    test_groups={}
    for row in rows:
        if row["split"]=="test": test_groups.setdefault(row["prompt_group"],[]).append(row)
    fixed_groups={g for g,group in test_groups.items() if all(len(tokens(r["LLM_output"]))>=128 for r in group)}
    length_control=[]
    for record in rq4.to_dict("records"):
        pred=predictions[record["prediction_path"]]
        subset=pred[pred.prompt_group.isin(fixed_groups)]
        length_control.append({"architecture":record["architecture"],"seed":record["seed"],"condition":record["condition"],
            "accuracy":accuracy_score(subset.actual,subset.predicted),
            "macro_f1":f1_score(subset.actual,subset.predicted,labels=LABELS,average="macro",zero_division=0),
            "test_groups":len(fixed_groups),"test_rows":len(subset)})
    length_frame=pd.DataFrame(length_control)
    length_frame.to_csv(SUITE/"rq4_fixed_length_runs.csv",index=False)
    summarize(length_frame,["architecture","condition"]).to_csv(SUITE/"rq4_fixed_length_aggregate.csv",index=False)
    gaps=task_gap_intervals(rq3,predictions)
    gaps.to_csv(SUITE/"rq3_transfer_gaps.csv",index=False)
    training=[r for r in rows if r["split"]=="train"]
    features=pd.DataFrame([text_features(r) for r in training])
    features.to_csv(SUITE/"training_text_features.csv",index=False)
    feature_summary=features.groupby("family").agg(examples=("sample_id","size"),median_full_characters=("full_characters","median"),
        median_full_tokens=("full_tokens","median"),mean_prefix_punctuation=("prefix_punctuation_fraction","mean"),
        mean_prefix_uppercase=("prefix_uppercase_fraction","mean"),mean_ttr_50=("ttr_first_50_words","mean"),
        ttr_eligible_responses=("ttr_first_50_words","count"),full_heading_fraction=("full_markdown_heading","mean"))
    feature_summary.to_csv(SUITE/"text_feature_summary.csv")
    lexical_markers(training).to_csv(SUITE/"distinctive_words.csv",index=False)
    overlap=validation_overlap([r for r in rows if r["split"]=="validation"])
    write_json(SUITE/"validation_error_overlap.json",overlap)

    figures=SUITE/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
    names={"question_answering":"Questions","drafting":"Drafting"}
    fig,axes=plt.subplots(1,2,figsize=(9,4.6),layout="constrained")
    for ax,architecture in zip(axes,("cnn","lstm")):
        subset=summaries["rq3"][summaries["rq3"].architecture==architecture]
        means=subset.pivot(index="train_task",columns="test_task",values="accuracy_mean").reindex(index=TASKS,columns=TASKS)
        sd=subset.pivot(index="train_task",columns="test_task",values="accuracy_sd").reindex(index=TASKS,columns=TASKS)
        ax.imshow(means,cmap="Blues",vmin=1/3,vmax=1)
        for i in range(2):
            for j in range(2): ax.text(j,i,f"{means.iloc[i,j]*100:.1f}%\nSD {sd.iloc[i,j]*100:.1f} pp",ha="center",va="center",color="white" if means.iloc[i,j]>.73 else "#202020")
        ax.set_xticks(range(2),[names[t] for t in TASKS]); ax.set_yticks(range(2),[names[t] for t in TASKS])
        ax.set_xlabel("Test task"); ax.set_ylabel("Training task"); ax.set_title(architecture.upper())
    fig.suptitle("RQ3: does attribution transfer between task forms?")
    fig.supxlabel("Mean accuracy across 3 seeds. Equal training: 110 prompts/task. Tests: 25 question / 41 drafting prompts.",fontsize=9)
    fig.savefig(figures/"rq3_transfer.png",dpi=180); plt.close(fig)

    cond_names={"baseline":"Original prefix","punctuation_masked":"Punctuation\nidentities masked","words_masked":"Word identities\nmasked","order_shuffled":"Token order\nshuffled"}
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,kind,title in ((axes[0],"rq4","Retrained under each condition"),(axes[1],"frozen","Baseline weights kept fixed")):
        table=summaries[kind]
        if kind=="frozen": table=pd.concat([summaries["rq4"][summaries["rq4"].condition=="baseline"],table],ignore_index=True)
        conditions=list(CONDITIONS) if kind=="rq4" else [*CONDITIONS,"order_shuffled"]
        for i,(architecture,color) in enumerate((("cnn","#286096"),("lstm","#d38233"))):
            part=table[table.architecture==architecture].set_index("condition").loc[conditions]
            bars=ax.bar(np.arange(len(conditions))+(i-.5)*.35,part.accuracy_mean,width=.35,yerr=part.accuracy_sd,capsize=3,label=architecture.upper(),color=color)
            ax.bar_label(bars,labels=[f"{x:.2f}" for x in part.accuracy_mean],padding=9,fontsize=8)
        ax.set_xticks(range(len(conditions)),[cond_names[c] for c in conditions],rotation=15)
        ax.axhline(1/3,linestyle="--",color="#666666",linewidth=1)
        ax.set_ylim(0,1.08); ax.set_ylabel("Test accuracy"); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("RQ4: what remains learnable, and what disrupts a trained model?")
    fig.supxlabel("375 test responses / 125 prompt groups. Bars: 3-seed mean; error bars: seed SD. Masks preserve token positions and length.",fontsize=9)
    fig.savefig(figures/"rq4_ablations.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(11,4.7),layout="constrained")
    specifications=[("full_characters","Full response length","Characters (descriptive only)"),
                    ("prefix_punctuation_fraction","Punctuation in model-visible prefix","Fraction of first 128 tokens"),
                    ("ttr_first_50_words","Lexical diversity at a fixed word budget","Unique types / first 50 words")]
    for ax,(column,title,ylabel) in zip(axes,specifications):
        ax.boxplot([features.loc[features.family==l,column].dropna() for l in LABELS],tick_labels=LABELS,showfliers=False)
        ax.set_title(title,fontsize=10); ax.set_ylabel(ylabel)
    fig.suptitle("RQ4: descriptive differences in training responses")
    eligible=[int(feature_summary.loc[label,"ttr_eligible_responses"]) for label in LABELS]
    fig.supxlabel(f"578 training responses/family; lexical-diversity counts: {eligible[0]} / {eligible[1]} / {eligible[2]} (Gemma / Mistral / Qwen).\nBoxes: quartiles, median and 1.5-IQR whiskers. Outliers omitted from plot only; responses under 50 words lack TTR.",fontsize=9)
    fig.savefig(figures/"rq4_text_features.png",dpi=180); plt.close(fig)

    lines=["# RQ3 and RQ4 measured results","","Task-transfer and text-pattern experiments. All values below come from saved predictions.","",
           "## RQ3: task-form transfer","","| Model | Training task | Test task | Accuracy mean +/- seed SD | Macro F1 mean |","|---|---|---|---:|---:|"]
    for r in summaries["rq3"].itertuples(): lines.append(f"| {r.architecture.upper()} | {names[r.train_task]} | {names[r.test_task]} | {100*r.accuracy_mean:.1f}% +/- {100*r.accuracy_sd:.1f} pp | {r.macro_f1_mean:.3f} |")
    lines += ["","110 training and 29 validation prompt groups per task; test sets: 25 question groups (75 responses), 41 drafting groups (123 responses). Three balanced source families. All source-only models are evaluated on both tests.","",
              "Within-task and transfer scores must be compared on the same test column. The following gaps use transfer minus within-task accuracy:",""]
    for r in gaps.itertuples(): lines.append(f"- {r.architecture.upper()}, target {names[r.target_task]}: {100*r.transfer_minus_within_accuracy:+.1f} pp (seed SD {100*r.difference_sd_across_seeds:.1f}); conditional prompt-bootstrap interval [{100*r.prompt_bootstrap_low:+.1f}, {100*r.prompt_bootstrap_high:+.1f}] pp.")
    lines += ["","The task labels describe request forms with overlapping subject matter. Small task tests, three seeds and one source restrict generalization. Transfer need not decrease in every direction; any positive gap is descriptive, not proof that out-of-task training is better.","","![Task transfer](figures/rq3_transfer.png)","",
              "## RQ4: controlled retraining","","| Model | Condition | Accuracy mean +/- seed SD | Macro F1 mean |","|---|---|---:|---:|"]
    for r in summaries["rq4"].itertuples(): lines.append(f"| {r.architecture.upper()} | {r.condition} | {100*r.accuracy_mean:.1f}% +/- {100*r.accuracy_sd:.1f} pp | {r.macro_f1_mean:.3f} |")
    lines += ["","Masks replace token identities with ID 1 while retaining positions and sequence lengths. The same original 128-token prefix, vocabulary size, parameter count, split and optimizer settings are used. Words masked retains punctuation and length; punctuation masked retains punctuation positions and word identities. Neither removes every stylistic cue.","",
              "## RQ4: frozen-model stress tests","","| Model | Test perturbation | Accuracy mean +/- seed SD |","|---|---|---:|"]
    for r in summaries["frozen"].itertuples(): lines.append(f"| {r.architecture.upper()} | {r.condition} | {100*r.accuracy_mean:.1f}% +/- {100*r.accuracy_sd:.1f} pp |")
    lines += ["","Frozen-model drops include distribution shift. Retraining shows what a new model can learn from retained signals; it does not assign unique causal importance to an individual feature. Shuffling preserves the token multiset and length but disrupts ordering.","","![Signal ablations](figures/rq4_ablations.png)","",
              "## Descriptive signals and scope","","Training-only summaries and fixed-budget lexical diversity are in `text_feature_summary.csv`; distinctive word document frequencies are in `distinctive_words.csv`. Full-length and line-break statistics describe data but are not direct numerical features of these prefix classifiers. Word markers are associations, not proven causal attributions.","","![Text features](figures/rq4_text_features.png)","",
              f"A post-hoc diagnostic retains {len(fixed_groups)} test prompt groups ({3*len(fixed_groups)} responses) where all three visible sequences are exactly 128 tokens. `rq4_fixed_length_aggregate.csv` reports the same saved models on this subgroup; no retraining or selection occurs. Words-masked performance here cannot be explained by differing visible sequence lengths alone.","",
              "The original test outcomes were already known before this follow-up. These are exploratory experiments, not a pristine final benchmark.","",
              "## Verification","","All 60 prediction sets were checked against their expected sample IDs, labels and groups; accuracy, macro F1 and confusion counts were recomputed. All 30 training checkpoints replayed validation predictions. Seed-448 RQ4 baselines reproduced both original response-only prediction files exactly.",""]
    (SUITE/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    write_json(SUITE/"verification.json",{"prediction_sets_verified":60,"training_runs":30,"rq3_evaluations":24,"rq4_retrained_evaluations":18,
        "rq4_frozen_evaluations":18,"seeds":SEEDS,"original_seed448_reference_predictions":"exact matches for CNN and LSTM",
        "figures":["rq3_transfer.png","rq4_ablations.png","rq4_text_features.png"]})
    for name,frame in summaries.items(): print(name,frame.to_string(index=False),sep="\n")
    print("Validation complementarity:",overlap)


if __name__=="__main__":
    main()
