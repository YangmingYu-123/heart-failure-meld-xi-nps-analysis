"""Univariable logistic associations within the common complete-case cohort.

Multilevel categorical predictors use a single treatment-coded model containing
all non-reference indicators. Additional clinical predictors use their available
records within this same common cohort, not the full 2,008-record cohort.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_cc import load_clean, model_design, csv
from paths import output_dir


def main():
    _, d, _ = load_clean()
    complete = model_design(d).notna().all(axis=1) & d.COMP6M.notna()
    a = d.loc[complete].copy()
    a['logBNP'] = np.log(a.BNP)
    rows = []

    def model(series, labels, reference=None, scale=1, additional=False):
        use = series.notna()
        s = series.loc[use]
        y = a.loc[use, 'COMP6M'].astype(int)
        x = pd.DataFrame({'const': 1.0}, index=y.index)
        if reference is None:
            x[labels] = s / scale
        else:
            codes = set(s.unique())
            if not codes.issubset(set(labels)) or reference not in codes:
                raise ValueError(f'Unexpected categorical coding: {codes}; labels: {labels}')
            for code, label in labels.items():
                if code != reference:
                    x[label] = s.eq(code).astype(float)
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        if not fit.converged:
            raise RuntimeError('Univariable model failed to converge')
        for term in x.columns.drop('const'):
            ci = fit.conf_int().loc[term]
            rows.append({
                'variable': term, 'n': len(y), 'events': int(y.sum()),
                'OR_per_unit': float(np.exp(fit.params[term])),
                'CI_lower': float(np.exp(ci.iloc[0])),
                'CI_upper': float(np.exp(ci.iloc[1])),
                'p': float(fit.pvalues[term]),
                'analysis': ('univariable logistic; variable-specific complete records within common complete-case cohort'
                             if additional else 'univariable logistic; common complete-case cohort'),
                'categorical_reference_code': reference,
                'coding': 'joint treatment-coded categorical indicators' if reference is not None else 'single predictor',
            })

    model(a.AGECAT, 'Age category (per ordered category)')
    model(a.GENDER.eq(1).astype(float).where(a.GENDER.notna()), 'Male sex')
    model(a.NYHA, {2: 'NYHA II (reference)', 3: 'NYHA III vs II', 4: 'NYHA IV vs II'}, reference=2)
    for c, label, scale in [('SBP', 'SBP (per 10 mmHg)', 10), ('PULSE', 'Pulse (per 10 beats/min)', 10),
                            ('logBNP', 'ln(BNP) (per 1 unit)', 1), ('NA', 'Sodium (per 5 mmol/L)', 5),
                            ('DIABETES', 'Diabetes', 1), ('COPD', 'COPD', 1),
                            ('MELD_XI_recalc', 'MELD-XI (per 5 points)', 5), ('NPS_recalc', 'NPS (per 1 point)', 1)]:
        model(a[c], label, scale=scale)
    for c, label, scale in [('BMI', 'BMI (per 1 kg/m2)', 1), ('CCI', 'CCI (per 1 point)', 1),
                            ('CKD', 'CKD', 1), ('MI', 'MI', 1), ('CVD', 'CVD', 1), ('PVD', 'PVD', 1),
                            ('HB', 'Haemoglobin (per 1 g/L)', 1), ('EGFR', 'eGFR (per 10 mL/min/1.73 m2)', 10)]:
        model(a[c], label, scale=scale, additional=True)
    model(a.HFTYPE, {1: 'HFTYPE left vs right', 2: 'Right heart failure (reference)',
                      3: 'HFTYPE both vs right'}, reference=2, additional=True)
    out = output_dir('analysis_univariable')
    result = pd.DataFrame(rows)
    csv(result, out/'univariable_logistic.csv')
    (out/'univariable_methods.md').write_text(
        f'Univariable models used the common complete-case cohort (n={len(a)}, events={int(a.COMP6M.sum())}). '
        'Additional clinical predictors used variable-specific complete records within that cohort. '
        'NYHA class was entered as two indicators in one model (II reference; III and IV contrasts), '
        'and heart-failure type as two indicators in one model (right reference; left and both contrasts). '
        'eGFR effects are per 10 mL/min/1.73 m2. Odds ratios are not risk ratios. '
        'No univariable P-value selection was used to choose multivariable predictors.', encoding='utf-8')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
