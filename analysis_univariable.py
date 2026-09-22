
from pathlib import Path
import sys
ROOT=Path(r'C:\Users\MSI\.codex\visualizations\2026\09\15\01a0a3e6-ceae-7230-94c6-1c2cebfb90c8')
sys.path.insert(0,str(ROOT/'python_packages'))
import numpy as np,pandas as pd, statsmodels.api as sm
from analysis_cc import load_clean,recalculate,csv
_,d,_=load_clean()
d['MELD']=d['MELD_XI_recalc']; d['NPS']=d['NPS_recalc']; d['logBNP']=np.log(d.BNP)
# Common complete-case sample for the prespecified clinical + scores analysis
vars_needed=['AGECAT','GENDER','NYHA','SBP','PULSE','logBNP','NA','DIABETES','COPD','MELD','NPS','COMP6M']
a=d.dropna(subset=vars_needed).copy()
y=a.COMP6M.astype(int)
rows=[]
def one(name,s,scale=1,kind='continuous'):
    mask=s.notna(); yy=y[mask]; xx=pd.DataFrame({'const':1,'x':s[mask]/scale})
    fit=sm.GLM(yy,xx,family=sm.families.Binomial()).fit()
    ci=fit.conf_int().loc['x']
    rows.append({'variable':name,'n':int(mask.sum()),'events':int(yy.sum()),'OR_per_unit':float(np.exp(fit.params.x)),'CI_lower':float(np.exp(ci[0])),'CI_upper':float(np.exp(ci[1])),'p':float(fit.pvalues.x),'analysis':'univariable logistic; common complete-case cohort'})
one('Age category (per ordered category)',a.AGECAT)
one('Male sex',a.GENDER.eq(1).astype(float))
one('NYHA III vs II',a.NYHA.eq(3).astype(float))
one('NYHA IV vs II',a.NYHA.eq(4).astype(float))
one('SBP (per 10 mmHg)',a.SBP,10)
one('Pulse (per 10 beats/min)',a.PULSE,10)
one('ln(BNP) (per 1 unit)',a.logBNP)
one('Sodium (per 5 mmol/L)',a.NA,5)
one('Diabetes',a.DIABETES)
one('COPD',a.COPD)
one('MELD-XI (per 5 points)',a.MELD,5)
one('NPS (per 1 point)',a.NPS)
# extra clinical variables, each with variable-specific complete n but full outcome
extra=[('BMI',a.BMI,1),('CCI',a.CCI,1),('CKD',a.CKD,1),('MI',a.MI,1),('CVD',a.CVD,1),('PVD',a.PVD,1),('HB',a.HB,1),('eGFR',a.EGFR,10),('HFTYPE left vs right',a.HFTYPE.eq(1).astype(float),1),('HFTYPE both vs right',a.HFTYPE.eq(3).astype(float),1)]
for name,s,scale in extra:
    # use all rows with variable; preserve endpoint
    mask=s.notna()&d.COMP6M.notna(); yy=d.loc[mask,'COMP6M'].astype(int); xx=pd.DataFrame({'const':1,'x':s[mask]/scale})
    fit=sm.GLM(yy,xx,family=sm.families.Binomial()).fit(); ci=fit.conf_int().loc['x']
    rows.append({'variable':name,'n':int(mask.sum()),'events':int(yy.sum()),'OR_per_unit':float(np.exp(fit.params.x)),'CI_lower':float(np.exp(ci[0])),'CI_upper':float(np.exp(ci[1])),'p':float(fit.pvalues.x),'analysis':'univariable logistic; variable-specific complete outcome'})
out=ROOT/'analysis_univariable';out.mkdir(exist_ok=True)
csv(pd.DataFrame(rows),out/'univariable_logistic.csv')
(out/'univariable_methods.md').write_text('All primary univariable estimates were computed in the common complete-case cohort used for model performance (n=1736, events=709). Additional clinical variables were fitted with variable-specific complete outcome records and are descriptive only. Odds ratios are not risk ratios; no p-value-based variable selection was used for the prespecified multivariable model.',encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))

