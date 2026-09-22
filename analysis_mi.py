"""Full-cohort association MICE with raw score components, 20 independent chains.
PMM donors on log-transformed skewed components; scores passively recalculated.
Outcome indicators in imputation for association only, never used in CV/validation.
"""
from pathlib import Path
import sys, os, json, warnings, time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python_packages'))
os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.mice import MICEData
from scipy import stats
from threadpoolctl import threadpool_limits
from analysis_cc import load_clean,recalculate,model_design,csv
threadpool_limits(limits=1)
warnings.filterwarnings('ignore',category=FutureWarning)
OUT=ROOT/'analysis_mi';OUT.mkdir(exist_ok=True)
raw,d,log=load_clean()
B=20;BURN=15
rawcols=['TBIL','CREAT','ALB','TC','NEUT_C','LYM','MONO_C','SBP','PULSE','BNP','NA']
logcols=['TBIL','CREAT','TC','NEUT_C','LYM','MONO_C','BNP']
imp=pd.DataFrame(index=d.index)
for col in rawcols:
    imp[col]=np.log(d[col]) if col in logcols else d[col]
# Categorical complete fields explicitly encoded. Omit composite to avoid collinearity.
imp['AGECAT']=d.AGECAT
imp['male']=d.GENDER.eq(1).astype(int)
imp['NYHA_III']=d.NYHA.eq(3).astype(int);imp['NYHA_IV']=d.NYHA.eq(4).astype(int)
for col in ['DIABETES','COPD','DEATH6M','READM6M','VISIT_N']:imp[col]=d[col]
assert imp[['AGECAT','male','NYHA_III','NYHA_IV','DIABETES','COPD','DEATH6M','READM6M','VISIT_N']].notna().all().all()
records=[];trace=[];diagnostics=[];snapshots=[]
pvals=[];mi_preds=[];covs={}
for chain in range(B):
    np.random.seed(20260915+chain)
    mi=MICEData(imp.copy(),perturbation_method='gaussian',k_pmm=5)
    for it in range(BURN):
        mi.update_all()
        for col in rawcols:
            miss=imp[col].isna()
            if miss.any():
                trace.append({'chain':chain+1,'iteration':it+1,'variable':col,'imputed_mean_transformed':mi.data.loc[miss,col].mean(),'imputed_sd_transformed':mi.data.loc[miss,col].std()})
    completed=d.copy()
    for col in rawcols:
        completed[col]=np.exp(mi.data[col]) if col in logcols else mi.data[col]
        # exact preservation of observed inputs
        completed.loc[d[col].notna(),col]=d.loc[d[col].notna(),col]
        miss=d[col].isna()
        if miss.any():diagnostics.append({'chain':chain+1,'variable':col,'n_missing':int(miss.sum()),'observed_mean':d[col].mean(),'imputed_mean':completed.loc[miss,col].mean(),'observed_median':d[col].median(),'imputed_median':completed.loc[miss,col].median(),'observed_min':d[col].min(),'imputed_min':completed.loc[miss,col].min(),'observed_max':d[col].max(),'imputed_max':completed.loc[miss,col].max()})
    completed=recalculate(completed)
    X=model_design(completed); assert X.notna().all().all()
    base=[c for c in X if c not in ['MELD_XI_per5','NPS_per1']]
    terms={'M0':base,'M1':base+['MELD_XI_per5'],'M2':base+['NPS_per1'],'M3':base+['MELD_XI_per5','NPS_per1']}
    y=completed.COMP6M.astype(int)
    for model,cols in terms.items():
        fit=sm.GLM(y,X[cols],family=sm.families.Binomial()).fit(maxiter=100)
        assert fit.converged
        covs.setdefault(model,[]).append(np.asarray(fit.cov_params()))
        for term in cols:
            records.append({'chain':chain+1,'analysis':'primary_logistic','model':model,'term':term,'beta':fit.params[term],'SE':fit.bse[term],'n':len(y),'events':int(y.sum())})
    # Supplementary RR for common endpoint; no OR -> RR mislabelling.
    fit=sm.GLM(y,X[terms['M3']],family=sm.families.Poisson()).fit(cov_type='HC0')
    for term in terms['M3']:
        records.append({'chain':chain+1,'analysis':'modified_Poisson','model':'M3','term':term,'beta':fit.params[term],'SE':fit.bse[term],'n':len(y),'events':int(y.sum())})
    # Product interaction explicitly exploratory
    Xi=X.copy(); Xi['MELD_x_NPS']=Xi.MELD_XI_per5*Xi.NPS_per1
    fit=sm.GLM(y,Xi,family=sm.families.Binomial()).fit()
    term='MELD_x_NPS'
    records.append({'chain':chain+1,'analysis':'interaction','model':'M3_product','term':term,'beta':fit.params[term],'SE':fit.bse[term],'n':len(y),'events':int(y.sum())})
    # Recorded readmission and earlier composite: same reference model, descriptive secondary.
    for outcome in ['READM6M','COMP28','COMP3M']:
        yy=completed[outcome].astype(int)
        fit=sm.GLM(yy,X,family=sm.families.Binomial()).fit()
        for term in ['MELD_XI_per5','NPS_per1']:
            records.append({'chain':chain+1,'analysis':outcome,'model':'M3','term':term,'beta':fit.params[term],'SE':fit.bse[term],'n':len(yy),'events':int(yy.sum())})
    snapshots.append(completed[rawcols+['MELD_XI_recalc','NPS_recalc']].assign(chain=chain+1,row=np.arange(len(completed))))
    print(f'Completed independent imputation chain {chain+1}/{B}',flush=True)
records=pd.DataFrame(records);csv(records,OUT/'MI_coefficients_each.csv')
pd.concat(snapshots,ignore_index=True).to_pickle(OUT/'MI_completed_predictors_local.pkl')
csv(pd.DataFrame(trace),OUT/'MI_trace.csv');csv(pd.DataFrame(diagnostics),OUT/'MI_diagnostics.csv')
pooled=[]
for (analysis,model,term),g in records.groupby(['analysis','model','term'],sort=False):
    q=g.beta.mean();u=(g.SE**2).mean();b=g.beta.var(ddof=1);t=u+(1+1/B)*b;se=np.sqrt(t)
    lam=(1+1/B)*b/t
    old=(B-1)/max(lam**2,1e-16)
    dfcom=2008-13
    obs=(dfcom+1)/(dfcom+3)*dfcom*(1-lam)
    df=1/(1/old+1/obs)
    crit=stats.t.ppf(.975,df)
    pooled.append({'analysis':analysis,'model':model,'term':term,'m':B,'n':int(g.n.iloc[0]),'events':int(g.events.iloc[0]),'beta':q,'SE':se,'effect_ratio':np.exp(q),'CI_lower':np.exp(q-crit*se),'CI_upper':np.exp(q+crit*se),'p':2*stats.t.sf(abs(q/se),df),'df_Barnard_Rubin':df,'fraction_missing_variance':lam,'MCSE_beta':np.sqrt(b/B),'within_variance':u,'between_variance':b,'ratio_type':'RR' if analysis=='modified_Poisson' else 'OR'})
p=pd.DataFrame(pooled);csv(p,OUT/'MI_pooled_associations.csv')
note={'cohort_n':2008,'events':830,'method':'MICEData chained equations with Gaussian coefficient perturbation and predictive mean matching, 5 donors','datasets':B,'independent_seeds':list(range(20260915,20260915+B)),'burnin_sweeps_per_chain':BURN,'variables':list(imp.columns),'log_transformed_before_PMM':logcols,'score_treatment':'passive recalculation from imputed components, observed data held exact','pool':'Rubin with Barnard-Rubin degrees of freedom; normal coefficient MI, no model performance pooling','intended_use':'association only; outcomes included in imputation; must not report predictions from these imputations as external or cross-validated performance','observed_derived_score_consistency':'scores recalc checked in analysis_cc','versions':{'numpy':np.__version__,'pandas':pd.__version__,'statsmodels':sm.__version__}}
(OUT/'MI_run_settings.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
print(p[p.term.isin(['MELD_XI_per5','NPS_per1','MELD_x_NPS'])].to_string(index=False))

