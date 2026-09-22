from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python_packages'))
import numpy as np,pandas as pd
from scipy.special import expit
from scipy.optimize import minimize,brentq
from scipy.stats import chi2
from scipy.optimize._numdiff import approx_derivative
from analysis_cc import load_clean,model_design,csv
OUT=ROOT/'analysis_death';OUT.mkdir(exist_ok=True)
_,d,_=load_clean()
d['logBNP']=np.log(d.BNP)
cols=['AGECAT','GENDER','NYHA','logBNP','MELD_XI_recalc','NPS_recalc']
a=d.dropna(subset=cols).copy()
X=np.column_stack([np.ones(len(a)),a.AGECAT,a.GENDER.eq(1).astype(float),a.NYHA,a.logBNP,a.MELD_XI_recalc/5,a.NPS_recalc])
names=['const','AGECAT','male','NYHA_ordinal','logBNP','MELD_XI_per5','NPS_per1']
y=a.DEATH6M.to_numpy(float)
# Center/scale nonintercept terms improves optimization; affine constant in penalty does not affect LR.
center=X[:,1:].mean(axis=0);scale=X[:,1:].std(axis=0)
Z=X.copy();Z[:,1:]=(X[:,1:]-center)/scale
def objective(b):
    eta=Z@b;p=expit(eta);w=p*(1-p)
    info=Z.T@(w[:,None]*Z);sign,ld=np.linalg.slogdet(info)
    if sign<=0:return 1e100
    return -(np.sum(y*eta-np.logaddexp(0,eta))+.5*ld)
def grad(b):
    p=expit(Z@b);w=p*(1-p);info=Z.T@(w[:,None]*Z)
    h=w*np.einsum('ij,jk,ik->i',Z,np.linalg.inv(info),Z)
    return -Z.T@(y-p+h*(.5-p))
start=np.zeros(Z.shape[1]);start[0]=np.log((y.sum()+.5)/(len(y)-y.sum()+.5))
fit=minimize(objective,start,jac=grad,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
b=fit.x
assert np.max(abs(grad(b)))<1e-5
err=np.max(abs(grad(b)-approx_derivative(lambda v: np.array([objective(v)]),b).ravel()))
assert err<1e-4
L=objective(b); threshold=chi2.ppf(.95,1)
results=[]
for j in [5,6]:
    rest=[k for k in range(len(b)) if k!=j]
    def profile(fixed):
        z=b.copy();z[j]=fixed
        def fun(u):z[rest]=u;return objective(z)
        def jac(u):z[rest]=u;return grad(z)[rest]
        r=minimize(fun,b[rest],jac=jac,method='BFGS',options={'gtol':1e-7,'maxiter':500})
        if np.max(abs(jac(r.x)))>1e-4:raise RuntimeError('profile convergence')
        return float(r.fun)
    def rootfn(v):return 2*(profile(v)-L)-threshold
    lower=b[j]-.25;upper=b[j]+.25
    while rootfn(lower)<0:lower=b[j]-2*(b[j]-lower)
    while rootfn(upper)<0:upper=b[j]+2*(upper-b[j])
    lo=brentq(rootfn,lower,b[j],xtol=1e-7)/scale[j-1];hi=brentq(rootfn,b[j],upper,xtol=1e-7)/scale[j-1]
    beta=b[j]/scale[j-1];lr=2*(profile(0)-L)
    results.append({'outcome':'DEATH6M','method':'Firth logistic','term':names[j],'n':len(y),'events':int(y.sum()),'beta':beta,'OR':np.exp(beta),'CI_lower':np.exp(lo),'CI_upper':np.exp(hi),'p_penalized_LR':chi2.sf(lr,1),'CI_method':'95% profile penalized likelihood'})
csv(pd.DataFrame(results),OUT/'death_Firth_profile.csv')
(OUT/'death_Firth_diagnostics.json').write_text(json.dumps({'n':len(y),'events':int(y.sum()),'terms':names,'optimizer_message':str(fit.message),'score_max_abs':float(np.max(abs(grad(b)))),'analytic_vs_numeric_gradient_error':float(err),'intended_interpretation':'complete-case exploratory death association, no validated death prediction model','full_cohort_deaths':57},indent=2),encoding='utf-8')
print(pd.DataFrame(results).to_string(index=False))

