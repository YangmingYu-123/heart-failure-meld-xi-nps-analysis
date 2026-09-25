"""Generate publication figures from completed analysis outputs without refitting.

Reads private score rows and held-out predictions for histograms and ROC curves,
and aggregate counts, calibration and DCA files. Input results remain private;
only the aggregate PNG/PDF figures and verification JSON may be shared after
author review. No model coefficients or statistical calculations are changed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


COLORS = {"M0": "#666666", "M1": "#0072B2", "M2": "#E69F00", "M3": "#8B5AA5"}
LABELS = {"M0": "Clinical model", "M1": "Clinical + MELD-XI", "M2": "Clinical + NPS", "M3": "Clinical + MELD-XI + NPS"}
STYLES = {"M0": "-", "M1": "-", "M2": "--", "M3": ":"}


def read(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregate input: {path}")
    return pd.read_csv(path, keep_default_na=False)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_roc(root: Path, out: Path):
    p = read(root, "analysis_cc/bootstrap/CV10x10_average_OOF_predictions.csv")
    y = p.COMP6M.to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.6, 5.3), layout="constrained")
    for m in COLORS:
        fpr, tpr, _ = roc_curve(y, p[m].to_numpy(dtype=float))
        auc = roc_auc_score(y, p[m].to_numpy(dtype=float))
        ax.plot(fpr, tpr, color=COLORS[m], ls=STYLES[m], lw=1.5, label=f"{LABELS[m]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#B5B5B5", ls="--", lw=.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - Specificity", ylabel="Sensitivity", title="Discrimination in repeated cross-validation")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    save(fig, out, "Figure2_ROC_complete_case_OOF")


def figure_calibration(root: Path, out: Path):
    cal = read(root, "analysis_cc/bootstrap/calibration_deciles.csv")
    cal = cal[cal.analysis == "CV10x10_average_OOF"]
    fig, axs = plt.subplots(2, 2, figsize=(6.9, 6.3), sharex=True, sharey=True, layout="constrained")
    for ax, m in zip(axs.flat, COLORS):
        z = cal[cal.model == m]
        ax.plot([0, 1], [0, 1], color="#B5B5B5", ls="--", lw=.8)
        ax.errorbar(z.mean_predicted, z.observed_rate,
                    yerr=[z.observed_rate-z.Wilson_CI_lower, z.Wilson_CI_upper-z.observed_rate],
                    fmt="o-", color=COLORS[m], markersize=3.3, lw=1, elinewidth=.65, capsize=2)
        ax.set(xlim=(.05, .82), ylim=(.05, .82), title=LABELS[m])
        ax.set_aspect("equal")
    for ax in axs[-1]: ax.set_xlabel("Mean predicted probability")
    for ax in axs[:, 0]: ax.set_ylabel("Observed event proportion")
    save(fig, out, "Figure3_calibration_complete_case_OOF")


def figure_dca(root: Path, out: Path):
    dca = read(root, "analysis_cc/bootstrap/CV10x10_DCA.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.7), layout="constrained")
    for m in COLORS:
        z = dca[dca.model == m]
        axs[0].plot(z.threshold, z.net_benefit, color=COLORS[m], ls=STYLES[m], lw=1.5, label=LABELS[m])
    z = dca[dca.model == "M0"]
    axs[0].plot(z.threshold, z.treat_all_net_benefit, color="#AAAAAA", ls="--", lw=.9, label="Treat all")
    axs[0].axhline(0, color="#222222", ls="-.", lw=.8, label="Treat none")
    axs[0].set(xlim=(.1, .7), ylim=(-.05, .35), xlabel="Risk threshold probability", ylabel="Net benefit", title="Decision curves")
    axs[0].legend(loc="upper right", fontsize=7)
    for m in ["M1", "M2", "M3"]:
        a = dca[dca.model == m]
        base = dca[dca.model == "M0"]
        axs[1].plot(a.threshold, a.net_benefit.to_numpy()-base.net_benefit.to_numpy(), color=COLORS[m], ls=STYLES[m], lw=1.3, label=LABELS[m])
    axs[1].axhline(0, color=COLORS["M0"], lw=.8)
    axs[1].set(xlim=(.1, .7), xlabel="Risk threshold probability", ylabel="Net benefit difference", title="Difference from clinical model")
    axs[1].legend(loc="lower left", fontsize=7)
    save(fig, out, "Figure4_DCA_complete_case_OOF")


def figure_flow(root: Path, out: Path):
    counts = read(root, "analysis_cc/cohort_flow_counts.csv")
    c=counts[counts.cohort=='all_records'].set_index('panel')
    total=int(c.loc['whole_cohort','n']);events=int(c.loc['whole_cohort','COMP6M'])
    both=int(c.loc['both_scores_observed','n']);cc=int(c.loc['M0_M3_complete_case','n'])
    cc_events=int(c.loc['M0_M3_complete_case','COMP6M'])
    deaths=int(c.loc['whole_cohort','DEATH6M']);readmits=int(c.loc['whole_cohort','READM6M'])
    audit=json.loads((root/'analysis_cc/data_audit.json').read_text(encoding='utf-8'))
    unique=total-int(audit['missing_ID'])-int(audit['duplicated_ID'])
    overlap=int(audit['death_readmission_overlap_n'])
    assert events==deaths+readmits-overlap
    settings=json.loads((root/'analysis_mi/MI_run_settings.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(7.3,7.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,fill='#F3F6F8'):
        p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.012,rounding_size=0.007',fc=fill,ec='#495866',lw=1)
        ax.add_patch(p);ax.text(x,y,text,ha='center',va='center',fontsize=10.5,linespacing=1.35)
    def arr(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.15,color='#34434E'))
    box(.5,.91,.91,.12,f'Zigong hospitalized heart failure source cohort\n{total:,} records; {unique:,} unique inpatient identifiers\nDecember 2016 to June 2019')
    arr((.5,.84),(.5,.77))
    overlap_text='No overlap between recorded component flags' if overlap==0 else f'{overlap} records have both component flags'
    box(.5,.70,.91,.13,f'Six-month outcome flags available for all {total:,} records\n{events} composite events: {deaths} deaths + {readmits} readmissions\n{overlap_text}')
    arr((.35,.62),(.23,.52));arr((.65,.62),(.75,.52))
    box(.235,.45,.40,.12,f'Association analyses\nAll {total:,} records\n{events} composite events','#E7F1F6')
    box(.75,.45,.40,.12,f'Both scores observed\nn = {both:,}\n{total-both} with incomplete scores')
    arr((.235,.375),(.235,.23));arr((.75,.375),(.75,.23))
    ax.text(.75,.305,f'Additional missing clinical\npredictors: {both-cc} records',ha='center',va='center',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',pad=3))
    box(.235,.155,.40,.13,f'{settings["datasets"]} imputed datasets\nPassive score recalculation\nPooled association estimates','#E7F1F6')
    box(.75,.155,.40,.13,f'Common complete cases\nn = {cc:,}; {cc_events} events\nInternal prediction validation')
    save(fig, out, "Figure1_cohort_flow_EN")


def figure_scores(root: Path, out: Path):
    d=pd.read_csv(root/'analysis_cc/analysis_ingredients_local.csv')
    m=d.MELD_XI_recalc.dropna();n=d.NPS_recalc.dropna()
    counts=n.value_counts().reindex(range(5),fill_value=0).sort_index()
    ns=read(root,'analysis_cc/NPS_distribution_and_events.csv')
    assert np.array_equal(counts.to_numpy(),ns.n.to_numpy())
    edges=np.arange(9.44,55.45,2)
    if m.max()>edges[-1]:edges=np.append(edges,np.ceil(m.max())+1)
    fig,(aa,bb)=plt.subplots(1,2,figsize=(7.2047244,3.3858268));fig.subplots_adjust(.075,.19,.98,.81,.31)
    aa.hist(m,bins=edges,color='#356B8C',edgecolor='white',linewidth=.4)
    aa.set(xlabel='MELD-XI score',ylabel='Patients',xlim=(8.4,float(edges[-1])))
    aa.set_xticks([10,20,30,40,50]);aa.set_title(f'MELD-XI (n = {len(m):,})',loc='left',pad=24)
    aa.text(-.14,1.19,'a',transform=aa.transAxes,fontweight='bold',fontsize=10)
    floor=int(m.eq(9.44).sum())
    aa.text(.98,.95,f'Minimum score (9.44)\n{floor:,} patients ({100*floor/len(m):.1f}%)',transform=aa.transAxes,ha='right',va='top',fontsize=7.5)
    bb.bar(counts.index,counts.values,color='#4C827D',width=.65)
    for x,y in counts.items():bb.text(x,y+max(counts.values)*.025,f'{y:d}\n({100*y/counts.sum():.1f}%)',ha='center',va='bottom',fontsize=7)
    bb.set(xlabel='Naples Prognostic Score',ylabel='Patients',xticks=range(5),ylim=(0,max(counts.values)*1.22))
    bb.set_title(f'NPS (n = {len(n):,})',loc='left',pad=24);bb.text(-.14,1.19,'b',transform=bb.transAxes,fontweight='bold',fontsize=10)
    save(fig, out, "FigureS1_score_distributions")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True, help="Private results root produced by the analysis pipeline")
    p.add_argument("--out", type=Path, required=True, help="Figure output directory")
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,
                         'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.6,'xtick.direction':'out','ytick.direction':'out',
                         'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42})
    for f in [figure_flow, figure_scores, figure_roc, figure_calibration, figure_dca]: f(args.results, args.out)
    for source, stem in [('MELD_XI_RCS','FigureS2_MELD_XI_RCS'),
                         ('joint_groups_exploratory','FigureS3_joint_groups')]:
        for extension in ['png','pdf']:
            original=args.results/'analysis_cc'/'exploratory'/f'{source}.{extension}'
            if not original.is_file():raise FileNotFoundError(f'Run analysis_exploratory.py first: {original}')
            target=args.out/f'{stem}.{extension}'
            if original.resolve()!=target.resolve():shutil.copyfile(original,target)
    preds=pd.read_csv(args.results/'analysis_cc/bootstrap/CV10x10_average_OOF_predictions.csv')
    aucs={m:float(roc_auc_score(preds.COMP6M,preds[m])) for m in COLORS}
    perf=read(args.results,'analysis_cc/bootstrap/CV10x10_average_OOF_performance.csv').set_index('model')
    assert all(abs(aucs[m]-float(perf.loc[m,'AUC']))<1e-12 for m in COLORS)
    dist=read(args.results,'analysis_cc/NPS_distribution_and_events.csv')
    (args.out / "figure_generation_metadata.json").write_text(json.dumps({
        "figures": ["Figure1_cohort_flow_EN", "Figure2_ROC_complete_case_OOF", "Figure3_calibration_complete_case_OOF", "Figure4_DCA_complete_case_OOF", "FigureS1_score_distributions", "FigureS2_MELD_XI_RCS", "FigureS3_joint_groups"],
        "scope": "Plotting of completed outputs only; no model fitting. Score rows and held-out predictions are read privately and not exported.",
        "validation": "Plotted ROC AUCs match saved OOF performance within 1e-12; NPS plotted counts equal saved score distribution.",
        "OOF_AUCs":aucs,"OOF_n":len(preds),"OOF_events":int(preds.COMP6M.sum()),
        "NPS_counts":{str(int(r.NPS)):int(r.n) for _,r in dist.iterrows()},
        "visual_scope":"Figure data, groups, metrics and labels correspond to manuscript figures. Matplotlib/font differences can change raster appearance. This script does not reproduce DOCX pagination or journal typesetting."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
