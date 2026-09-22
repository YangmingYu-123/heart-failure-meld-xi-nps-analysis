# Supplementary Code 1

Custom statistical analysis source accompanying the MELD-XI and Naples Prognostic Score heart failure manuscript.

## Contents

- analysis_cc.py: input checks, score recalculation, baseline summaries and complete-case models.
- analysis_mi.py: multiple imputation and pooled association analyses.
- pool_mi_model_df.py: pooled estimates using model-specific complete-data degrees of freedom.
- analysis_univariable.py: univariable logistic regressions.
- analysis_exploratory.py: subgroup, interaction, spline and sensitivity analyses.
- analysis_death.py: Firth logistic sensitivity analysis for death.
- bootstrap_analysis.py: bootstrap optimism correction and repeated cross-validation.
- metadata/: preserved multiple-imputation and validation settings.

## Source data and access

The original data descriptor cites https://doi.org/10.13026/8a9e-w734, which resolves to the PhysioNet version 1.2 record. At verification, that record requires registration and a signed data use agreement. Follow current repository access and license requirements. Do not infer that all later versions have identical contents.

The supplied analytical input was HFYC_NPS.xls: a tab-delimited text extract with 2,008 records and 226 fields. Its recorded SHA-256 is b2572d59d7d6d5a19c307c0c33477e0456c5675f972f2dad627d874947d80f1e. The source descriptor reports 166 attributes. This archive does not establish which repository version was used to create that local extract, nor does it contain the original download-to-coded-extract transformation. The authors should retain the variable mapping and document that provenance before claiming full end-to-end reproducibility. No participant-level dataset, imputed dataset or individual prediction file is included.

## Environment and execution

The scripts use Python, numpy, pandas, scipy, statsmodels, scikit-learn, patsy, matplotlib and threadpoolctl. Preserved run metadata contain recorded versions for some dependencies; a complete original environment lock is not available. Do not treat the version strings as a tested installation recipe for another environment.

These are archived analysis scripts, not a portable installer. Before execution, update SOURCE (and DICT, if used) in analysis_cc.py and replace absolute project paths in analysis_univariable.py and pool_mi_model_df.py. Keep all scripts together. The optional python_packages directory is a local dependency override and is not distributed. Install dependencies in your own environment instead.

Use the coded input expected by analysis_cc.py, with the column names and units described in the manuscript. Passing --data to analysis_cc.py alone does not change the default input used by the other scripts; their imported load_clean function uses SOURCE.

Suggested sequence after configuring paths:

1. python analysis_cc.py
2. python analysis_mi.py
3. python pool_mi_model_df.py
4. python analysis_univariable.py
5. python analysis_exploratory.py
6. python analysis_death.py
7. python bootstrap_analysis.py

Scripts write analysis output directories beside the source files. Run in a fresh working copy to preserve previous outputs. Multiple imputation and validation can take substantial time.

## Verification scope

The packaged Python source was checked for syntax and matched byte-for-byte to the existing project files. The analyses were not rerun as part of preparing this declarations revision. No claim of independent reproduction in a clean environment is made. SHA256SUMS.json records archive-member checksums.
