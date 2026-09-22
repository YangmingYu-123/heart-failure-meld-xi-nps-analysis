from pathlib import Path
import sys
sys.path.insert(0,str(Path(r'C:\Users\MSI\.codex\visualizations\2026\09\15\01a0a3e6-ceae-7230-94c6-1c2cebfb90c8')/'python_packages'))
import numpy as np,pandas as pd
from scipy import stats
root=Path(r'C:\Users\MSI\.codex\visualizations\2026\09\15\01a0a3e6-ceae-7230-94c6-1c2cebfb90c8');out=root/'analysis_mi'
r=pd.read_csv(out/'MI_coefficients_each.csv');B=int(r.chain.nunique());rows=[]
for (analysis,model,term),g in r.groupby(['analysis','model','term'],sort=False):
 q=g.beta.mean();u=(g.SE**2).mean();b=g.beta.var(ddof=1);t=u+(1+1/B)*b;se=np.sqrt(t);lam=(1+1/B)*b/t
 df_old=(B-1)/max(lam**2,1e-16);pcols={'M0':11,'M1':12,'M2':12,'M3':13,'M3_product':14};dfcom=2008-pcols.get(model,13);df_obs=(dfcom+1)/(dfcom+3)*dfcom*(1-lam);df=1/(1/df_old+1/df_obs);crit=stats.t.ppf(.975,df)
 rows.append({'analysis':analysis,'model':model,'term':term,'m':B,'n':int(g.n.iloc[0]),'events':int(g.events.iloc[0]),'beta':q,'SE':se,'effect_ratio':np.exp(q),'CI_lower':np.exp(q-crit*se),'CI_upper':np.exp(q+crit*se),'p':2*stats.t.sf(abs(q/se),df),'df_Barnard_Rubin':df,'dfcom_model_specific':dfcom,'fraction_missing_variance':lam,'MCSE_beta':np.sqrt(b/B),'within_variance':u,'between_variance':b,'ratio_type':'RR' if analysis=='modified_Poisson' else 'OR'})
p=pd.DataFrame(rows);p.to_csv(out/'MI_pooled_associations_model_specific_df.csv',index=False,encoding='utf-8-sig')
print(p[(p.analysis=='primary_logistic')&(p.model.isin(['M1','M2','M3']))&(p.term.isin(['MELD_XI_per5','NPS_per1']))].to_string(index=False))
