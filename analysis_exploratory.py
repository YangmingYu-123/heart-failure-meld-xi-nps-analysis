"""Prespecified exploratory analyses on the common complete-case HFYC sample.
No source data are modified. All reported tests are exploratory and unadjusted
for multiplicity. Uses the shared cleaning and model-design implementation.
"""
from pathlib import Path
import os, json
BASE = Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(BASE))
from paths import output_dir, output_root
os.environ.setdefault('MPLCONFIGDIR', str(output_root()/'analysis_cc'/'exploratory'/'mpl_cache'))
from analysis_cc import load_clean, model_design, csv, jsonify
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

OUT=output_dir('analysis_cc')/'exploratory'
OUT.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
raw,d,cleanlog=load_clean()
Xall=model_design(d)
mask=Xall.notna().all(axis=1)&d.COMP6M.notna()
a=d.loc[mask].copy()
X=Xall.loc[mask].copy()
y=a.COMP6M.astype(int)
basecols=[c for c in X.columns if c not in ['MELD_XI_per5','NPS_per1']]
assert len(basecols)==11
N=len(a);E=int(y.sum())
rows=[]

def fit(x, yy=y):
    assert x.index.equals(yy.index)
    assert np.isfinite(x.to_numpy()).all()
    rank=np.linalg.matrix_rank(x.to_numpy())
    if rank!=x.shape[1]: raise ValueError(f'Design rank {rank} != {x.shape[1]}: {list(x.columns)}')
    f=sm.GLM(yy,x,family=sm.families.Binomial()).fit(maxiter=200)
    assert f.converged
    return f

def test(full,reduced,name,kind,n=N,events=E):
    df=int(round(full.df_model-reduced.df_model))
    lr=max(0.,2*(full.llf-reduced.llf))
    row={'analysis':kind,'comparison':name,'n':n,'events':events,'test':'likelihood ratio','LR_chi2':lr,'df':df,'p':float(stats.chi2.sf(lr,df)),'multiplicity_adjusted':False}
    rows.append(row)
    return row

def coef(f, term, analysis, comparison='', n=N, events=E, extra=None):
    beta=float(f.params[term]);se=float(f.bse[term])
    row={'analysis':analysis,'comparison':comparison,'term':term,'n':n,'events':events,'beta':beta,'SE':se,'OR':np.exp(beta),'CI_lower':np.exp(beta-1.95996398454*se),'CI_upper':np.exp(beta+1.95996398454*se),'p':float(f.pvalues[term]),'test':'Wald','multiplicity_adjusted':False}
    if extra:row.update(extra)
    rows.append(row)
    return row

m0=fit(X[basecols]);m3=fit(X)
# Confirm the imported pipeline reproduces the completed common-case analysis.
previous=pd.read_csv(output_root()/'analysis_cc'/'CC_model_performance_apparent.csv')
for name,f in [('M0',m0),('M3',m3)]:
    old=previous.query('cohort == "all_records" and model == @name').iloc[0]
    assert int(old.n)==N and int(old.events)==E
    assert abs(old.loglik-f.llf)<1e-7

knots=np.quantile(a.MELD_XI_recalc,[.05,.35,.65,.95])
if len(np.unique(knots))!=4: raise ValueError('Repeated RCS knots require a documented prespecified adjustment')
ref=float(a.MELD_XI_recalc.median())

def rcs_basis(x):
    x=np.asarray(x,dtype=float)
    k=knots;den=(k[-1]-k[0])**2
    columns=[x/5]
    for tj in k[:-2]:
        z=np.maximum(x-tj,0)**3
        z-=np.maximum(x-k[-2],0)**3*(k[-1]-tj)/(k[-1]-k[-2])
        z+=np.maximum(x-k[-1],0)**3*(k[-2]-tj)/(k[-1]-k[-2])
        columns.append(z/den)
    return np.column_stack(columns)

rs=X[basecols+['NPS_per1']].copy()
br=rcs_basis(a.MELD_XI_recalc)
rterms=['MELD_linear_per5','MELD_rcs1','MELD_rcs2']
for j,t in enumerate(rterms):rs[t]=br[:,j]
frcs=fit(rs)
frcs_reduced=fit(X[basecols+['NPS_per1']])
total=test(frcs,frcs_reduced,'MELD-XI RCS overall, adjusted for M0 and NPS','MELD_RCS')
nonlin=test(frcs,m3,'MELD-XI RCS vs linear MELD-XI, adjusted for M0 and NPS','MELD_RCS')
# Show the full observed range; rug makes sparse tails visible.
xgrid=np.unique(np.r_[np.linspace(a.MELD_XI_recalc.min(),a.MELD_XI_recalc.max(),501),ref,knots])
diff=rcs_basis(xgrid)-rcs_basis([ref])
params=frcs.params.loc[rterms].to_numpy()
cov=frcs.cov_params().loc[rterms,rterms].to_numpy()
logor=diff@params
se=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',diff,cov,diff)))
rcsdata=pd.DataFrame({'MELD_XI':xgrid,'reference':ref,'OR':np.exp(logor),'CI_lower':np.exp(logor-1.95996398454*se),'CI_upper':np.exp(logor+1.95996398454*se),'logOR':logor,'SE_logOR':se})
csv(rcsdata,OUT/'MELD_XI_RCS_curve_data.csv')
# Check reference contrast and continuity/linearity of the tails numerically.
assert np.max(np.abs(rcs_basis([ref])-rcs_basis([ref])))==0
assert np.allclose(np.diff(rcs_basis(np.array([knots[-1]+1,knots[-1]+2,knots[-1]+3])),n=2,axis=0),0,atol=1e-9)

npsgroup=a.NPS_recalc.clip(lower=1).astype(int)
xn=X[basecols+['MELD_XI_per5']].copy()
for k in [2,3,4]:xn[f'NPS_{k}']=npsgroup.eq(k).astype(float)
fn=fit(xn);fn_reduced=fit(X[basecols+['MELD_XI_per5']])
npstest=test(fn,fn_reduced,'NPS categories 0-1, 2, 3, 4 adjusted for M0 and linear MELD-XI','NPS_categorical')
npsrows=[]
for k in [1,2,3,4]:
    sel=npsgroup.eq(k);n=int(sel.sum());e=int(y[sel].sum());lo,hi=sm.stats.proportion_confint(e,n,method='wilson')
    extra={'group':'0-1' if k==1 else str(k),'group_n':n,'group_events':e,'event_rate':e/n,'event_CI_lower':lo,'event_CI_upper':hi}
    if k==1:
        r={'analysis':'NPS_categorical','comparison':'reference 0-1','term':'reference','n':N,'events':E,'OR':1.,'CI_lower':1.,'CI_upper':1.,'p':np.nan,'test':'reference',**extra};rows.append(r)
    else:r=coef(fn,f'NPS_{k}','NPS_categorical',f'{k} vs 0-1',extra=extra)
    npsrows.append(r)
csv(pd.DataFrame(npsrows),OUT/'NPS_categorical_OR_events.csv')

# Median-by-prespecified NPS >=3 groups; no data-dependent search for a cutoff.
hm=a.MELD_XI_recalc.ge(ref);hn=a.NPS_recalc.ge(3)
qcode=hm.astype(int)*2+hn.astype(int)
qnames={0:'Lower MELD-XI / NPS 0-2',1:'Lower MELD-XI / NPS 3-4',2:'Higher MELD-XI / NPS 0-2',3:'Higher MELD-XI / NPS 3-4'}
xq=X[basecols].copy()
for k in [1,2,3]:xq[f'joint_group_{k}']=qcode.eq(k).astype(float)
fq=fit(xq);qtest=test(fq,m0,'Joint four-category grouping added to M0','joint_groups')
qrows=[]
for k in range(4):
    sel=qcode.eq(k);n=int(sel.sum());e=int(y[sel].sum());lo,hi=sm.stats.proportion_confint(e,n,method='wilson')
    extra={'group':qnames[k],'group_code':k,'group_n':n,'group_events':e,'event_rate':e/n,'event_CI_lower':lo,'event_CI_upper':hi,'MELD_median_threshold':ref,'MELD_high_definition':'>= median','NPS_high_definition':'>= 3'}
    if k==0:
        r={'analysis':'joint_groups','comparison':'reference group','term':'reference','n':N,'events':E,'OR':1.,'CI_lower':1.,'CI_upper':1.,'p':np.nan,'test':'reference',**extra};rows.append(r)
    else:r=coef(fq,f'joint_group_{k}','joint_groups',f'{qnames[k]} vs {qnames[0]}',extra=extra)
    qrows.append(r)
csv(pd.DataFrame(qrows),OUT/'joint_groups_OR_Wilson_events.csv')

# One score-by-subgroup interaction at a time, controlling for the other score.
# Derived subgroup main effects are added only if not already present in M0.
subgroups=[('sex',a.GENDER.eq(1).astype(float).where(a.GENDER.notna()),['Female','Male'],'male'),
('age_category',a.AGECAT.gt(6).astype(float).where(a.AGECAT.notna()),['AGECAT 1-6','AGECAT 7-8'],None),
('NYHA',a.NYHA.eq(4).astype(float).where(a.NYHA.notna()),['NYHA II-III','NYHA IV'],'NYHA_IV'),
('CKD',a.CKD,['No CKD','CKD'],None)]
subrows=[]
for sg,g,labels,existing in subgroups:
    valid=g.notna();g=g.loc[valid];xx=X.loc[valid].copy();yy=y.loc[valid];sn=len(yy);ev=int(yy.sum())
    if existing is None:xx['subgroup_main']=g
    fbase=fit(xx,yy)
    for exposure,unit in [('MELD_XI_per5','per 5 points'),('NPS_per1','per 1 point')]:
        xi=xx.copy();term=exposure+'_x_subgroup';xi[term]=xi[exposure]*g
        f=fit(xi,yy)
        tr=test(f,fbase,f'{exposure} by {sg}','subgroup_interaction',sn,ev)
        tr.update({'subgroup':sg,'exposure':exposure,'exposure_unit':unit})
        for level in [0,1]:
            contrast=pd.Series(0.,index=xi.columns);contrast[exposure]=1.;contrast[term]=level
            beta=float(contrast@f.params);variance=float(contrast@f.cov_params()@contrast);se_=np.sqrt(max(0.,variance))
            ng=int(g.eq(level).sum());eg=int(yy[g.eq(level)].sum())
            r={'analysis':'subgroup_contrast','subgroup':sg,'group':labels[level],'exposure':exposure,'exposure_unit':unit,'n':sn,'events':ev,'group_n':ng,'group_events':eg,'beta':beta,'SE':se_,'OR':np.exp(beta),'CI_lower':np.exp(beta-1.95996398454*se_),'CI_upper':np.exp(beta+1.95996398454*se_),'p':2*stats.norm.sf(abs(beta/se_)),'interaction_p':tr['p'],'test':'within-level Wald contrast from interaction model','multiplicity_adjusted':False}
            rows.append(r);subrows.append(r)
csv(pd.DataFrame(subrows),OUT/'subgroup_OR_interactions.csv')

# Equal-weight exploratory sum, scaled using this sample only; no new clinical score.
meld_mean=float(a.MELD_XI_recalc.mean());meld_sd=float(a.MELD_XI_recalc.std(ddof=1))
nps_mean=float(a.NPS_recalc.mean());nps_sd=float(a.NPS_recalc.std(ddof=1))
zsum=(a.MELD_XI_recalc-meld_mean)/meld_sd+(a.NPS_recalc-nps_mean)/nps_sd
xz=X[basecols].copy();xz['equal_weight_zsum']=zsum
fz=fit(xz);zt=test(fz,m0,'M0 + equal-weight standardized sum vs M0','equal_weight_sum')
zr=coef(fz,'equal_weight_zsum','equal_weight_sum','per 1 unit in z(MELD-XI) + z(NPS)',extra={'MELD_mean':meld_mean,'MELD_SD':meld_sd,'NPS_mean':nps_mean,'NPS_SD':nps_sd,'validation':'apparent only, exploratory; not a clinical score'})
for name,f,xx in [('M0',m0,X[basecols]),('M0_zsum',fz,xz),('M3',m3,X)]:
    p=f.predict(xx);rows.append({'analysis':'equal_weight_apparent_performance','comparison':name,'n':N,'events':E,'AUC_apparent':roc_auc_score(y,p),'Brier_apparent':brier_score_loss(y,p),'AIC':f.aic,'parameters_including_intercept':xx.shape[1],'validation':'apparent only, not internally validated'})

# A prespecified linear continuous-score interaction, separate from subgroup tests.
xprod=X.copy();xprod['MELD_x_NPS']=xprod.MELD_XI_per5*xprod.NPS_per1
fprod=fit(xprod)
prodtest=test(fprod,m3,'MELD-XI by NPS product interaction','score_interaction')
coef(fprod,'MELD_x_NPS','score_interaction','product of MELD-XI per5 and NPS per1')

csv(pd.DataFrame(rows),OUT/'sensitivity_exploratory.csv')
metadata={'n_complete_case':N,'events':E,'primary_outcome':'recorded COMP6M','knots_percentiles':[5,35,65,95],'MELD_RCS_knots':knots.tolist(),'MELD_floor_n':int(a.MELD_XI_recalc.eq(9.44).sum()),'MELD_median_reference':ref,'NPS_groups':['0-1','2','3','4'],'joint_group_thresholds':{'MELD_high_greater_equal':ref,'NPS_high_greater_equal':3},'M0_terms':basecols,'NPS_category_adjustment':'M0 plus continuous linear MELD-XI','MELD_RCS_adjustment':'M0 plus continuous linear NPS','subgroup_model':'M0 + both scores + subgroup main effect when not already present + one score-by-subgroup product at a time','subgroup_interaction_n':8,'subgroup_CKD_missing_n':int(a.CKD.isna().sum()),'multiplicity_adjustment':'none; every analysis exploratory','equal_weight_sum':'z(MELD-XI)+z(NPS) with sample means and sample SDs; apparent performance only, not clinical score','overall_RCS_test':total,'nonlinear_RCS_test':nonlin,'NPS_category_global_test':npstest,'four_group_global_test':qtest,'score_product_interaction':prodtest,'baseline_pipeline_consistency':'M0 and M3 log-likelihoods match analysis_cc.py to <1e-7','tail_linearity_test':'passed','all_models_converged':True}
(OUT/'exploratory_metadata.json').write_text(json.dumps(jsonify(metadata),indent=2),encoding='utf-8')

# Dose-response curve with observed counts as a separate aligned panel.
fig,(ax,ah)=plt.subplots(2,1,figsize=(6.9,5.9),sharex=True,gridspec_kw={'height_ratios':[4.1,1.0],'hspace':.08})
ax.fill_between(xgrid,rcsdata.CI_lower,rcsdata.CI_upper,color='#0072B2',alpha=.16,lw=0,label='95% confidence interval')
ax.plot(xgrid,rcsdata.OR,color='#0072B2',lw=2,label='Adjusted odds ratio')
ax.axhline(1,color='#777777',ls='--',lw=.8);ax.axvline(ref,color='#888888',ls=':',lw=.8)
ax.set_yscale('log');ax.set_yticks([0.6,1,2,4,6]);ax.set_yticklabels(['0.6','1','2','4','6']);ax.yaxis.set_minor_formatter(NullFormatter());ax.set_ylabel('Adjusted odds ratio (log scale)');ax.set_title('MELD-XI and the six-month composite outcome',loc='left',fontweight='bold',pad=15)
ax.text(.97,.97,f'n = {N:,}; events = {E:,}\nOverall P = {total["p"]:.3f}\nNonlinearity P = {nonlin["p"]:.3f}\nReference = median ({ref:.2f})',transform=ax.transAxes,ha='right',va='top',fontsize=9)
ax.legend(loc='lower left',frameon=False,fontsize=8)
for k in knots:ah.axvline(k,color='#0072B2',ls=':',lw=.8)
ah.hist(a.MELD_XI_recalc,bins=np.linspace(a.MELD_XI_recalc.min(),a.MELD_XI_recalc.max(),30),color='#a7b7c7',edgecolor='white',lw=.5)
ah.set_ylabel('Patients');ah.set_xlabel('MELD-XI score');ah.set_xlim(a.MELD_XI_recalc.min()-.2,a.MELD_XI_recalc.max()+.3)
fig.subplots_adjust(left=.16,right=.98,top=.90,bottom=.24)
fig.text(.16,.035,'Adjusted for the clinical reference model and continuous NPS.\nKnots: 5th, 35th, 65th and 95th percentiles. Exploratory complete-case analysis.\nNo adjustment for multiple testing.',fontsize=8,va='bottom')
for ext in ['png','pdf']:fig.savefig(OUT/f'MELD_XI_RCS.{ext}',dpi=350)
plt.close(fig)

# Four groups show observed event rates with Wilson CI and adjusted odds ratios.
fig,(ar,ao)=plt.subplots(1,2,figsize=(10.5,4.2),gridspec_kw={'width_ratios':[1.3,1]})
qdf=pd.DataFrame(qrows); ypos=np.arange(4)
rates=qdf.event_rate.to_numpy()*100
ar.errorbar(rates,ypos,xerr=np.vstack([rates-qdf.event_CI_lower.to_numpy()*100,qdf.event_CI_upper.to_numpy()*100-rates]),fmt='o',color='#0072B2',capsize=3)
ar.set_yticks(ypos);ar.set_yticklabels([f'{qnames[k]}\n(n = {qrows[k]["group_n"]}, events = {qrows[k]["group_events"]})' for k in range(4)],fontsize=9);ar.invert_yaxis();ar.set_xlim(0,max(65,qdf.event_CI_upper.max()*100+4));ar.set_xlabel('Observed six-month event rate (%)');ar.set_title('A  Observed event rate (95% Wilson CI)',loc='left',fontweight='bold',fontsize=10)
for k,v in enumerate(rates):ar.text(qdf.iloc[k].event_CI_upper*100+1,ypos[k],f'{v:.1f}%',va='center',fontsize=8)
ors=qdf.OR.to_numpy();lo=qdf.CI_lower.to_numpy();hi=qdf.CI_upper.to_numpy()
ao.errorbar(ors,ypos,xerr=np.vstack([ors-lo,hi-ors]),fmt='o',color='#D55E00',capsize=3)
ao.axvline(1,color='#777777',ls='--',lw=.8);ao.set_yticks(ypos);ao.set_yticklabels([]);ao.invert_yaxis();ao.set_xscale('log');ao.set_xlim(min(.4,lo.min()*.8),max(4,hi.max()*1.3));ao.set_xticks([.5,1,2,4]);ao.set_xticklabels(['0.5','1','2','4']);ao.xaxis.set_minor_formatter(NullFormatter());ao.set_xlabel('Adjusted odds ratio (log scale)');ao.set_title('B  Adjusted odds ratio (95% CI)',loc='left',fontweight='bold',fontsize=10)
fig.subplots_adjust(left=.25,right=.97,top=.86,bottom=.28,wspace=.27)
fig.text(.25,.045,f'Lower MELD-XI < {ref:.2f}; higher MELD-XI >= {ref:.2f}. NPS high group: 3-4 points.\nReference: lower MELD-XI / NPS 0-2. Exploratory complete-case analysis; no multiple-testing adjustment.',fontsize=8)
for ext in ['png','pdf']:fig.savefig(OUT/f'joint_groups_exploratory.{ext}',dpi=350)
plt.close(fig)

note=f'''# Exploratory complete-case analyses

All analyses used the common complete-case sample (n = {N}, events = {E}), except CKD subgroup analysis (n = {N-int(a.CKD.isna().sum())}). The eight score-by-subgroup interaction tests were exploratory without multiplicity adjustment. A nominal P value below 0.05 should not establish subgroup heterogeneity.

MELD-XI RCS used four distinct knots at {', '.join(f'{k:.6f}' for k in knots)}. The lower floor of 9.44 included {int(a.MELD_XI_recalc.eq(9.44).sum())} patients. The adjusted overall test was P = {total['p']:.6f}; the two-degree-of-freedom nonlinearity test was P = {nonlin['p']:.6f}. The curve references the complete-case median, {ref:.6f}, and adjusts for M0 plus continuous NPS. The displayed range includes the entire observed range, with an aligned histogram showing sparse tails.

NPS was analysed using categories 0-1, 2, 3 and 4, adjusting for M0 and linear MELD-XI. The global three-degree-of-freedom test was P = {npstest['p']:.6f}. No cubic spline was applied to this five-level ordinal score. Four joint categories used MELD-XI >= {ref:.6f} and NPS >= 3 as high categories. Their global test was P = {qtest['p']:.6f}. Group event rates use Wilson confidence intervals; adjusted odds ratios compare with lower MELD-XI and NPS 0-2.

The equal-weight standardized sum was an exploratory restriction of the relative score contributions. Its apparent performance was not internally validated and cannot be interpreted as a newly validated clinical score. No threshold was optimized against the outcome. Model selection was not based on subgroup P values.

Output files contain all tests and estimates, including nonsignificant results. Baseline M0 and M3 likelihoods match the original shared analysis pipeline. Every model converged; every fitted design matrix had full column rank. The RCS implementation passed an upper-tail linearity check. Odds ratios must not be labelled risk ratios.
'''
(OUT/'exploratory_analysis_notes.md').write_text(note,encoding='utf-8')
print('OUTPUT',OUT)
print(json.dumps(jsonify(metadata),indent=2))
print(pd.DataFrame(rows).query('analysis == "subgroup_interaction"')[['comparison','n','p']].to_string(index=False))
print(pd.DataFrame(npsrows)[['group','group_n','group_events','OR','CI_lower','CI_upper','p']].to_string(index=False))
print(pd.DataFrame(qrows)[['group','group_n','group_events','event_rate','OR','CI_lower','CI_upper','p']].to_string(index=False))
print(pd.DataFrame(rows).query('analysis == "equal_weight_apparent_performance"').to_string(index=False))


