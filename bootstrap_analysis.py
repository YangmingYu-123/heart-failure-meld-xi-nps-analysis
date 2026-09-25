"""Common complete-case bootstrap validation. Seed 20260915, B=1000.
Each bootstrap fit is evaluated in its resample and in the original dataset
(Harrell optimism correction). No variable selection. Apparent bootstrap
percentile intervals are explicitly not validation intervals. Category-free
NRI is exploratory and no clinical decision threshold is inferred from it.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
from paths import output_dir
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.linalg import solve
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.model_selection import StratifiedKFold
from analysis_cc import load_clean,model_design,csv
SEED=20260915
B=1000
OUT=output_dir('analysis_cc')/'bootstrap'

def fit(X,y,maxiter=50):
    beta=np.zeros(X.shape[1]);beta[0]=logit(np.clip(y.mean(),1e-6,1-1e-6))
    for _ in range(maxiter):
        p=expit(X@beta);w=np.maximum(p*(1-p),1e-10)
        h=(X.T*w)@X;g=X.T@(y-p)
        delta=solve(h,g,assume_a='pos')
        beta+=delta
        if np.max(np.abs(delta))<1e-8:return beta
    raise RuntimeError('IRLS did not converge')

def calibration(y,p):
    lp=logit(np.clip(p,1e-10,1-1e-10))
    z=np.column_stack([np.ones(len(y)),lp]);b=fit(z,y)
    # Calibration-in-the-large fixes slope at one.
    a=0.
    for _ in range(40):
        q=expit(a+lp);delta=(y-q).sum()/np.sum(q*(1-q));a+=delta
        if abs(delta)<1e-10:break
    return a,b[1],b[0]

def performance(y,p):
    a,s,j=calibration(y,p)
    return {'AUC':roc_auc_score(y,p),'Brier':brier_score_loss(y,p),'calibration_in_large':a,'calibration_slope':s,'joint_calibration_intercept':j}

def increment(y,new,old):
    e=y==1;n=y==0;diff=new-old;up=diff>1e-12;down=diff< -1e-12
    return {'delta_AUC':roc_auc_score(y,new)-roc_auc_score(y,old),'delta_Brier':brier_score_loss(y,new)-brier_score_loss(y,old),'IDI':diff[e].mean()-diff[n].mean(),'NRI_category_free':up[e].mean()-down[e].mean()+down[n].mean()-up[n].mean()}

def net_benefit(y,p,thresholds):
    pos=p[:,None]>=thresholds
    return np.mean(pos*y[:,None],axis=0)-np.mean(pos*(1-y[:,None]),axis=0)*thresholds/(1-thresholds)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    _,d,_=load_clean();design=model_design(d);ok=design.notna().all(axis=1)&d.COMP6M.notna();d=d.loc[ok];design=design.loc[ok]
    y=d.COMP6M.to_numpy(dtype=float);X=design.to_numpy(dtype=float)
    # Center/scale continuous predictors for numerical conditioning, keeping intercept.
    centers=X[:,1:].mean(axis=0);scales=X[:,1:].std(axis=0);scales[scales==0]=1
    X[:,1:]=(X[:,1:]-centers)/scales
    indices={'M0':list(range(11)),'M1':list(range(12)),'M2':list(range(11))+[12],'M3':list(range(13))}
    xx={m:X[:,ix] for m,ix in indices.items()}
    apppred={m:expit(a@fit(a,y)) for m,a in xx.items()}
    appmetrics={m:performance(y,p) for m,p in apppred.items()}
    comparisons=[('M1','M0'),('M2','M0'),('M3','M0'),('M3','M1'),('M3','M2')]
    appdelta={a+' vs '+b:increment(y,apppred[a],apppred[b]) for a,b in comparisons}
    thresholds=np.arange(10,71)/100
    appnb={m:net_benefit(y,p,thresholds) for m,p in apppred.items()}
    rng=np.random.default_rng(SEED);rows=[];deltas=[];dcarows=[];fail=[]
    for b in range(B):
        ix=rng.integers(0,len(y),len(y));yb=y[ix];pred_boot={};pred_orig={}
        try:
            for m,a in xx.items():
                beta=fit(a[ix],yb);pb=expit(a[ix]@beta);po=expit(a@beta);pred_boot[m]=pb;pred_orig[m]=po
                mb=performance(yb,pb);mo=performance(y,po)
                rows.append({'replicate':b+1,'model':m,**{k+'_bootstrap':v for k,v in mb.items()},**{k+'_original':v for k,v in mo.items()}})
                nbb=net_benefit(yb,pb,thresholds);nbo=net_benefit(y,po,thresholds)
                for t,v,w in zip(thresholds,nbb,nbo):dcarows.append({'replicate':b+1,'model':m,'threshold':t,'NB_bootstrap':v,'NB_original':w})
            for a,c in comparisons:
                mb=increment(yb,pred_boot[a],pred_boot[c]);mo=increment(y,pred_orig[a],pred_orig[c])
                deltas.append({'replicate':b+1,'comparison':a+' vs '+c,**{k+'_bootstrap':v for k,v in mb.items()},**{k+'_original':v for k,v in mo.items()}})
        except Exception as err:
            fail.append({'replicate':b+1,'error':str(err)})
            raise
        if (b+1)%100==0:print('Completed bootstrap',b+1,flush=True)
    r=pd.DataFrame(rows);q=pd.DataFrame(deltas);dc=pd.DataFrame(dcarows)
    csv(r,OUT/'bootstrap_performance.csv');csv(q,OUT/'bootstrap_deltas.csv');csv(dc,OUT/'bootstrap_dca.csv')
    sums=[]
    for m in xx:
        z=r[r.model==m]
        for metric,app in appmetrics[m].items():
            vb=z[metric+'_bootstrap'];vo=z[metric+'_original'];opt=(vb-vo).mean()
            sums.append({'model':m,'metric':metric,'apparent':app,'mean_optimism':opt,'optimism_corrected':app-opt,'apparent_bootstrap_percentile_2_5':vb.quantile(.025),'apparent_bootstrap_percentile_97_5':vb.quantile(.975),'B':len(z)})
    csv(pd.DataFrame(sums),OUT/'bootstrap_performance_summary.csv')
    ds=[]
    for comp in appdelta:
        z=q[q.comparison==comp]
        for metric,app in appdelta[comp].items():
            vb=z[metric+'_bootstrap'];vo=z[metric+'_original'];opt=(vb-vo).mean()
            ds.append({'comparison':comp,'metric':metric,'apparent':app,'mean_optimism':opt,'optimism_corrected':app-opt,'apparent_bootstrap_percentile_2_5':vb.quantile(.025),'apparent_bootstrap_percentile_97_5':vb.quantile(.975),'B':len(z)})
    csv(pd.DataFrame(ds),OUT/'bootstrap_delta_summary.csv')
    ds=[]
    for m in xx:
        for i,t in enumerate(thresholds):
            z=dc[(dc.model==m)&np.isclose(dc.threshold,t)];app=appnb[m][i];opt=(z.NB_bootstrap-z.NB_original).mean()
            ds.append({'model':m,'threshold':t,'apparent_net_benefit':app,'mean_optimism':opt,'optimism_corrected_net_benefit':app-opt,'treat_all_net_benefit':y.mean()-(1-y.mean())*t/(1-t),'treat_none_net_benefit':0})
    csv(pd.DataFrame(ds),OUT/'bootstrap_dca_summary.csv')
    # 10 repeated stratified 10-fold CV. All participants have complete raw inputs;
    # the feature set and score formula are fixed. No imputation or variable selection.
    oof={m:np.zeros((len(y),10)) for m in xx};cvmetrics=[]
    for rep in range(10):
        kf=StratifiedKFold(10,shuffle=True,random_state=SEED+rep)
        for train,test in kf.split(X,y):
            for m,a in xx.items():oof[m][test,rep]=expit(a[test]@fit(a[train],y[train]))
        for m in xx:cvmetrics.append({'repeat':rep+1,'model':m,**performance(y,oof[m][:,rep])})
    csv(pd.DataFrame(cvmetrics),OUT/'CV10x10_performance_by_repeat.csv')
    cvpred={m:p.mean(axis=1) for m,p in oof.items()};predtable=pd.DataFrame({'source_row_including_header':d.index+2,'COMP6M':y,**cvpred});csv(predtable,OUT/'CV10x10_average_OOF_predictions.csv')
    csv(pd.DataFrame([{'model':m,**performance(y,p),'estimand':'metrics of average of 10 OOF predictions per participant'} for m,p in cvpred.items()]),OUT/'CV10x10_average_OOF_performance.csv')
    calrows=[]
    for tag,preds in [('apparent',apppred),('CV10x10_average_OOF',cvpred)]:
        for m,p in preds.items():
            groups=pd.qcut(p,10,labels=False,duplicates='drop')
            for g in sorted(np.unique(groups)):
                use=groups==g;n=int(use.sum());events=int(y[use].sum());lo,hi=sm.stats.proportion_confint(events,n,method='wilson')
                calrows.append({'analysis':tag,'model':m,'decile':int(g)+1,'n':n,'events':events,'mean_predicted':p[use].mean(),'observed_rate':events/n,'Wilson_CI_lower':lo,'Wilson_CI_upper':hi})
    csv(pd.DataFrame(calrows),OUT/'calibration_deciles.csv')
    csv(pd.DataFrame([{'model':m,'threshold':t,'net_benefit':nb,'treat_all_net_benefit':y.mean()-(1-y.mean())*t/(1-t),'treat_none_net_benefit':0} for m,p in cvpred.items() for t,nb in zip(thresholds,net_benefit(y,p,thresholds))]),OUT/'CV10x10_DCA.csv')
    (OUT/'validation_metadata.json').write_text(json.dumps({'seed':SEED,'B':B,'n':len(y),'events':int(y.sum()),'failures':fail,'bootstrap_method':'resample patients; refit each model in bootstrap; evaluate in bootstrap and original cohort; apparent minus average difference','interval_method':'percentiles of apparent bootstrap-resample metric; not CIs for corrected performance','NRI':'category-free; exploratory, no established decision threshold','CV':'10 repeats of stratified 10-fold CV. Participant predictions averaged over repeats; individual-repeat metrics provided separately.'},indent=2),encoding='utf-8')
    print(pd.DataFrame(sums).to_string(index=False));print(pd.DataFrame(ds).head().to_string(index=False))
if __name__=='__main__':main()
