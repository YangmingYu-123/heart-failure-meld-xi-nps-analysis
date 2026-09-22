"""HFYC_NPS: read-only source QC and common complete-case logistic models.
Run with bundled Python. Statistical dependencies in sibling python_packages.
AGECAT is ordinal, not exact age; all available patients retained as main cohort.
"""
from pathlib import Path
import sys, argparse, json, hashlib
pkg=Path(__file__).resolve().parent/'python_packages'
if pkg.exists(): sys.path.insert(0,str(pkg))
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score,brier_score_loss
SOURCE=Path(r'D:\EmpowerStats\Analysis\HFYC_NPS\HFYC_NPS.xls')
DICT=Path(r'D:\易侕论文\心衰Codex+OpenClaw+心衰队列的首发攻略\HF数据\变量说明文件.xlsx')
SCORES=['TBIL','CREAT','ALB','TC','NEUT_C','LYM','MONO_C']
M0_RAW=['AGECAT','GENDER','NYHA','SBP','PULSE','BNP','NA','DIABETES','COPD']
M0_DESCRIPTION='AGECAT ordinal + male + NYHA III/IV + SBP per 10 + pulse per 10 + ln BNP + Na per 5 + diabetes + COPD'
def csv(d,p): d.to_csv(p,index=False,encoding='utf-8-sig')
def jsonify(x):
    if isinstance(x,dict): return {str(k):jsonify(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [jsonify(v) for v in x]
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x) if np.isfinite(x) else None
    return x

def recalculate(d):
    d=d.copy()
    d['MELD_XI_recalc']=5.11*np.log((d.TBIL/17.1).clip(lower=1))+11.76*np.log((d.CREAT/88.4).clip(lower=1))+9.44
    d['MELD_XI_cap4']=5.11*np.log((d.TBIL/17.1).clip(lower=1))+11.76*np.log((d.CREAT/88.4).clip(lower=1,upper=4))+9.44
    d['NLR']=d.NEUT_C/d.LYM;d['LMR']=d.LYM/d.MONO_C
    specifications=[('NPS_ALB',d.ALB.lt(40),d.ALB.notna()),('NPS_TC',(d.TC*38.67).le(180),d.TC.notna()),('NPS_NLR',d.NLR.gt(2.96),d.NLR.notna()),('NPS_LMR',d.LMR.le(4.44),d.LMR.notna())]
    for c,condition,valid in specifications: d[c]=condition.astype(float).where(valid)
    d['NPS_recalc']=d[[s[0] for s in specifications]].sum(axis=1,min_count=4)
    return d

def load_clean(source=SOURCE):
    raw=pd.read_csv(source,sep='\t',na_values=['NA','']);d=raw.copy();log=[]
    for c in ['TBIL','CREAT','ALB','TC','LYM','MONO_C','SBP','DBP','BMI','BNP','HB','NA','PULSE']:
        bad=d[c].notna()&d[c].le(0)
        for i in d.index[bad]: log.append({'source_row_including_header':int(i+2),'variable':c,'value':d.at[i,c],'action':'set missing','reason':'nonpositive value'})
        d.loc[bad,c]=np.nan
    for c in ['NEUT_C']:
        bad=d[c].lt(0)
        for i in d.index[bad]: log.append({'source_row_including_header':int(i+2),'variable':c,'value':d.at[i,c],'action':'set missing','reason':'negative cell count'})
        d.loc[bad,c]=np.nan
    return raw,recalculate(d),pd.DataFrame(log)

def model_design(d,meld='MELD_XI_recalc',age_categorical=False):
    x=pd.DataFrame(index=d.index);x['const']=1.
    if age_categorical:
        x['AGECAT_1to4']=d.AGECAT.le(4).astype(float)
        for k in [5,7,8]:x[f'AGECAT_{k}']=d.AGECAT.eq(k).astype(float)
    else:x['AGECAT']=d.AGECAT.astype(float)
    x['male']=d.GENDER.eq(1).astype(float).where(d.GENDER.notna())
    x['NYHA_III']=d.NYHA.eq(3).astype(float).where(d.NYHA.notna())
    x['NYHA_IV']=d.NYHA.eq(4).astype(float).where(d.NYHA.notna())
    x['SBP_per10']=d.SBP/10;x['PULSE_per10']=d.PULSE/10;x['log_BNP']=np.log(d.BNP);x['NA_per5']=d.NA/5
    x['DIABETES']=d.DIABETES.astype(float);x['COPD']=d.COPD.astype(float)
    x['MELD_XI_per5']=d[meld]/5;x['NPS_per1']=d.NPS_recalc
    return x

def summary(s):
    a=s.dropna()
    return {'available':len(a),'missing':int(s.isna().sum()),'missing_pct':s.isna().mean()*100,'min':a.min(),'q1':a.quantile(.25),'median':a.median(),'q3':a.quantile(.75),'max':a.max(),'mean':a.mean(),'sd':a.std()}

def fit_models(d,label,out,agecat=False,meld='MELD_XI_recalc'):
    X=model_design(d,meld,agecat);mask=X.notna().all(axis=1)&d.COMP6M.notna();X=X.loc[mask];a=d.loc[mask];y=a.COMP6M.astype(int)
    base=[c for c in X if c not in ['MELD_XI_per5','NPS_per1']]
    cols={'M0':base,'M1':base+['MELD_XI_per5'],'M2':base+['NPS_per1'],'M3':base+['MELD_XI_per5','NPS_per1']}
    fits={};metrics=[];coefs=[];tests=[];preds=pd.DataFrame({'source_row_including_header':a.index+2,'COMP6M':y.to_numpy()})
    for name,terms in cols.items():
        f=sm.GLM(y,X[terms],family=sm.families.Binomial()).fit(maxiter=200);fits[name]=f;p=f.predict(X[terms]);ci=f.conf_int();preds[name]=p.to_numpy()
        metrics.append({'cohort':label,'model':name,'n':len(y),'events':int(y.sum()),'parameters_including_intercept':len(terms),'AUC_apparent':roc_auc_score(y,p),'Brier_apparent':brier_score_loss(y,p),'AIC':f.aic,'loglik':f.llf,'converged':f.converged})
        for c in terms:coefs.append({'cohort':label,'model':name,'term':c,'beta':f.params[c],'SE':f.bse[c],'OR':np.exp(f.params[c]),'CI_lower':np.exp(ci.loc[c,0]),'CI_upper':np.exp(ci.loc[c,1]),'p':f.pvalues[c]})
    for large,small in [('M1','M0'),('M2','M0'),('M3','M0'),('M3','M1'),('M3','M2')]:
        f,g=fits[large],fits[small];lr=2*(f.llf-g.llf);df=f.df_model-g.df_model
        tests.append({'cohort':label,'comparison':large+' vs '+small,'LR_chi2':lr,'df':df,'p':stats.chi2.sf(lr,df),'AUC_delta_apparent':metrics[int(large[1])]['AUC_apparent']-metrics[int(small[1])]['AUC_apparent'],'Brier_delta_apparent':metrics[int(large[1])]['Brier_apparent']-metrics[int(small[1])]['Brier_apparent']})
    xi=X[cols['M3']].copy();xi['MELD_x_NPS']=xi.MELD_XI_per5*xi.NPS_per1
    f=sm.GLM(y,xi,family=sm.families.Binomial()).fit(maxiter=200);lr=2*(f.llf-fits['M3'].llf)
    tests.append({'cohort':label,'comparison':'M3 + product interaction vs M3','LR_chi2':lr,'df':1,'p':stats.chi2.sf(lr,1),'AUC_delta_apparent':np.nan,'Brier_delta_apparent':np.nan})
    csv(preds,out/f'predictions_{label}.csv')
    return pd.DataFrame(metrics),pd.DataFrame(coefs),pd.DataFrame(tests)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=SOURCE);ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'analysis_cc');args=ap.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True)
    raw,d,log=load_clean(args.data);csv(log,out/'cleaning_log.csv')
    meta=pd.read_excel(DICT,sheet_name='meta');csv(meta[meta.iloc[:,0].isin(SCORES+M0_RAW+['VISIT_N','COMP6M','DEATH6M','READM6M','HOSPOUT','DEATHTM','READMTM'])],out/'variable_dictionary_used.csv')
    consistency=[]
    for original,recalc in [('NPS','NPS_recalc'),('MELD_XI','MELD_XI_recalc')]:
        both=raw[original].notna()&d[recalc].notna();delta=(raw[original]-d[recalc]).abs()
        consistency.append({'score':original,'original_available':int(raw[original].notna().sum()),'recalculated_available':int(d[recalc].notna().sum()),'paired_n':int(both.sum()),'mismatch_n_tolerance_1e-7':int((both&delta.gt(1e-7)).sum()),'missing_pattern_mismatch':int(raw[original].isna().ne(d[recalc].isna()).sum()),'max_absolute_error':delta[both].max()})
    csv(pd.DataFrame(consistency),out/'score_consistency.csv')
    audit={'data_path':str(args.data),'sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'n_rows':len(raw),'n_columns':raw.shape[1],'missing_ID':int(raw.ID.isna().sum()),'duplicated_ID':int(raw.ID.duplicated().sum()),'fully_duplicated_rows':int(raw.duplicated().sum()),'VISIT_N_counts':raw.VISIT_N.value_counts().to_dict(),'COMP6M_OR_mismatch_n':int(raw.COMP6M.ne((raw.DEATH6M.eq(1)|raw.READM6M.eq(1)).astype(int)).sum()),'death_readmission_overlap_n':int((raw.DEATH6M.eq(1)&raw.READM6M.eq(1)).sum()),'endpoint_counts':raw[['COMP6M','DEATH6M','READM6M','COMP28','COMP3M']].sum().to_dict(),'endpoint_missing':raw[['COMP6M','DEATH6M','READM6M']].isna().sum().to_dict(),'in_hospital_death_n':int(raw.HOSPOUT.eq(2).sum()),'in_hospital_deaths_in_DEATH6M':int((raw.HOSPOUT.eq(2)&raw.DEATH6M.eq(1)).sum()),'M0':M0_DESCRIPTION,'cohort_note':'VISIT_N is prior admission count; ID is unique. All 2008 records are main cohort; VISIT_N=1 is sensitivity, not deduplication.','score_consistency':consistency,'versions':{'Python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'statsmodels':sm.__version__}}
    for c in ['DEATHTM','READMTM']:audit[c]={**summary(d[c]),'over180_n':int(d[c].gt(180).sum())}
    audit['BNP_equal5000_n']=int(d.BNP.eq(5000).sum());audit['MELD_floor9_44_n']=int(d.MELD_XI_recalc.eq(9.44).sum());audit['BMI_over70_n']=int(d.BMI.gt(70).sum())
    (out/'data_audit.json').write_text(json.dumps(jsonify(audit),ensure_ascii=False,indent=2),encoding='utf-8')
    variables=list(dict.fromkeys(SCORES+M0_RAW+['MELD_XI_recalc','NPS_recalc','NLR','LMR','HB','EGFR','CKD','LIVER_DZ','LVEF','BMI','VISIT_N']))
    csv(pd.DataFrame([{'variable':c,**summary(d[c])} for c in variables]),out/'missingness_and_distribution.csv')
    counts=[]
    for label,a in [('all_records',d),('VISIT_N_1',d[d.VISIT_N.eq(1)]),('exclude_inhospital_deaths',d[d.HOSPOUT.ne(2)])]:
        panels={'whole_cohort':pd.Series(True,index=a.index),'MELD_observed':a.MELD_XI_recalc.notna(),'NPS_observed':a.NPS_recalc.notna(),'both_scores_observed':a[['MELD_XI_recalc','NPS_recalc']].notna().all(axis=1),'M0_M3_complete_case':model_design(a).notna().all(axis=1)}
        for panel,mask in panels.items():
            b=a.loc[mask];counts.append({'cohort':label,'panel':panel,'n':len(b),'COMP6M':b.COMP6M.sum(),'DEATH6M':b.DEATH6M.sum(),'READM6M':b.READM6M.sum(),'event_pct':b.COMP6M.mean()*100})
    csv(pd.DataFrame(counts),out/'cohort_flow_counts.csv')
    groups=[]
    for k in range(5):
        a=d[d.NPS_recalc.eq(k)];n=len(a);e=int(a.COMP6M.sum());lo,hi=sm.stats.proportion_confint(e,n,method='wilson');groups.append({'NPS':k,'n':n,'pct_of_observed':n/d.NPS_recalc.notna().sum()*100,'events':e,'event_rate':e/n,'CI_lower':lo,'CI_upper':hi})
    csv(pd.DataFrame(groups),out/'NPS_distribution_and_events.csv')
    table=[];categorical=['AGECAT','GENDER','NYHA','DIABETES','COPD','CKD','LIVER_DZ','NPS_recalc','VISIT_N']
    for c in variables:
        levels=sorted(d[c].dropna().unique()) if c in categorical else ['median [Q1, Q3]']
        for k in levels:
            r={'variable':c,'summary':str(k)}
            for label,a in [('Overall',d),('No_event',d[d.COMP6M.eq(0)]),('Event',d[d.COMP6M.eq(1)])]:
                s=a[c];r[label+'_missing_n']=int(s.isna().sum());r[label]=f'{int(s.eq(k).sum())} ({100*s.eq(k).sum()/s.notna().sum():.1f}%)' if c in categorical else f'{s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]'
            table.append(r)
    csv(pd.DataFrame(table),out/'Table1_observed_baseline.csv')
    metrics=[];coefs=[];tests=[]
    scenarios=[('all_records',d,False,'MELD_XI_recalc'),('VISIT_N_1',d[d.VISIT_N.eq(1)],False,'MELD_XI_recalc'),('exclude_inhospital_deaths',d[d.HOSPOUT.ne(2)],False,'MELD_XI_recalc'),('age_categorical_1to4merged',d,True,'MELD_XI_recalc'),('MELD_creatinine_capped4',d,False,'MELD_XI_cap4')]
    for label,a,agecat,meld in scenarios:
        m,c,t=fit_models(a,label,out,agecat,meld);metrics.append(m);coefs.append(c);tests.append(t)
    m=pd.concat(metrics);c=pd.concat(coefs);t=pd.concat(tests);csv(m,out/'CC_model_performance_apparent.csv');csv(c,out/'CC_logistic_coefficients.csv');csv(t,out/'CC_nested_LR_tests.csv')
    ingredients=d[SCORES+M0_RAW+['COMP6M','DEATH6M','READM6M','COMP28','COMP3M','VISIT_N','HOSPOUT','MELD_XI_recalc','NPS_recalc']].copy();ingredients.insert(0,'source_row_including_header',ingredients.index+2);csv(ingredients,out/'analysis_ingredients_local.csv')
    print(m.to_string(index=False));print(c.query("cohort == 'all_records' and term in ['MELD_XI_per5','NPS_per1']").to_string(index=False));print(t.query("cohort == 'all_records'").to_string(index=False))
if __name__=='__main__':main()
