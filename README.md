# MELD-XI and NPS heart-failure analysis

Python source accompanying the study of MELD-XI and Naples Prognostic Score
(NPS) in hospitalized patients with heart failure. The package contains code
and aggregate metadata only. It does not distribute participant-level data.

## Obtain authorized data

Register on PhysioNet, meet the current project-access requirements, and sign
the dataset's data-use agreement before downloading version 1.2:
https://physionet.org/content/heart-failure-zigong/1.2/
Dataset DOI: https://doi.org/10.13026/8a9e-w734

Keep the downloaded project ZIP or its `dat.csv` in a private location.
Neither the source data nor derived patient-level outputs may be uploaded to
this public repository. The code does not authenticate or download data.

## Install dependencies

Use Python 3.12 and install `requirements.txt` in your research environment.
Version pins record the actual local statistical environment; they are not a
claim that another operating system or freshly installed environment has been
independently validated. `openpyxl` is optional and is needed only if a private
Excel variable dictionary is supplied instead of the bundled public field map.

## Run from the native PhysioNet release

From this directory:

```text
python run_all.py --physionet /private/path/project-1.2.zip --out /private/path/results
```

`--physionet` also accepts the extracted native `dat.csv`. The runner invokes
`prepare_physionet.py`, which builds a 44-field minimal coded input under
`results/private_input/` using explicit source-field and category mappings.
The three composite outcomes are logical ORs of the recorded death/readmission
flags. They are not reconstructed from the potentially inconsistent time
fields. Shared cleaning and score calculations are then used by every model.

The converter refuses to overwrite an existing coded input. Use a new results
directory for a new native-data run. To rerun an existing converted input:

```text
python run_all.py --data /private/path/results/private_input/physionet_coded.tsv --float-precision round_trip --out /private/path/rerun_results
```

The source `dat.csv` used for local verification has SHA-256
`f2be23eeaf2e6f457512ad7653f6ce98a87ac23376596eef6d3bf46e4f311ae8`.
All 44 required fields were compared with the authors' earlier 2,008-record,
226-field coded extract: missing-value masks were identical and all values
agreed within an absolute tolerance of 1e-7. The code reconstructs the minimal
analysis input; it does not recreate the full historical 226-column export or
prove which release originally generated that export.

Prepared TSVs use 17 significant digits and are read with pandas' `round_trip`
float parser, preserving the native values as first parsed from `dat.csv`.
This matters for predictive mean matching: differences of one floating-point
bit can change donor ordering and hence the particular imputation chain.
Native mode configures this automatically; a manually rerun prepared TSV
requires the `--float-precision round_trip` option shown above. The original
historical coded extract uses the default parser when no option is supplied.

## Pipeline and outputs

`run_all.py` runs these scripts in order:

1. `analysis_cc.py`: source QC, cleaning log, score consistency, baseline
   summaries, common complete-case models and sensitivity models.
2. `analysis_mi.py`: 20 independently seeded chained-equation imputation runs
   with predictive mean matching; association analyses only.
3. `pool_mi_model_df.py`: Rubin/Barnard–Rubin pooling using model-specific
   complete-data degrees of freedom.
4. `analysis_univariable.py`: univariable logistic models. NYHA II and right
   heart failure are categorical reference groups; non-reference indicators
   enter the same model. Additional predictors use available records within
   the common complete-case cohort, with denominators reported per model.
5. `analysis_exploratory.py`: subgroup interactions, restricted cubic spline,
   categorical NPS, joint-group and other exploratory analyses.
6. `analysis_death.py`: exploratory Firth logistic association for recorded
   six-month death.
7. `bootstrap_analysis.py`: 1,000 bootstrap resamples and 10 repeats of
   stratified 10-fold cross-validation, with calibration and decision curves.

Multiple imputation and validation can take several minutes or longer. The
runner stops if any stage fails. Generated input, cleaning logs, imputed
predictor pickle files, and individual prediction tables are private research
data. Keep the entire results directory outside the public repository.

For individual scripts, set `HFYC_DATA` to the coded input, `HFYC_OUT` to
the results root, and `HFYC_FLOAT_PRECISION=round_trip` for a prepared TSV.
`analysis_cc.py` additionally accepts `--data`, `--out`, and
optional `--dictionary`. Without a private dictionary, it uses the bundled
`metadata/physionet_variable_mapping.csv`; the statistical pipeline therefore
does not depend on a separately shared workbook.

## Verification and interpretation

`metadata/provenance.json`, `metadata/native_mapping_verification.json`, and
`metadata/rerun_verification.json` describe the exact scope of completed verification.
The full pipeline was rerun locally from the authorized native ZIP, including
20 imputation chains and 1,000 bootstrap resamples. The historical imputation
script used `numpy.random.seed`, which does not seed `MICEData` in statsmodels
0.15 when `rng` is omitted. This revision passes an explicitly seeded NumPy
Generator to each chain. The canonical imputation coefficients, pooled results
and traces were regenerated in a second process and were byte-identical;
`metadata/MI_determinism_verification.json` records the checksums. Historical
imputation results are superseded by the corrected deterministic run.
No independent external validation of clinical predictive performance is
claimed. Multiple-imputation outcomes were used to aid association analyses;
performance validation is performed separately in the common complete-case
sample. Exploratory analyses are not adjusted for multiple testing. Odds
ratios must not be interpreted as risk ratios.

`SHA256SUMS.json` covers every public package member except itself. Verify its
contents before publishing the package. Do not upload the original or derived
patient-level inputs, IDs, imputed rows, or individual predictions.

## Rebuild the manuscript figures

After `run_all.py` completes, run:

```text
python plot_public_figures.py --results /private/path/results --out /private/path/figures
```

This creates PNG and PDF files named Figures 1–4 and Supplementary Figures
S1–S3, matching the manuscript's figure identities. It reads the complete
private results directory: patient-level score rows are required for the
histogram, and held-out predictions are required for ROC curves. Those inputs
remain private and are not copied to the figure directory. Flow counts,
calibration and decision curves are taken from already-computed outputs; no
model is refitted by the plotting command. S2 and S3 are copied from the plots
generated by `analysis_exploratory.py`.

The figures reproduce the plotted data, grouping, labels and numerical
quantities. Pixel identity with an earlier image is not guaranteed across
fonts, Matplotlib versions or output devices. DOCX layout and journal
typesetting are outside the statistical code package's reproducibility scope.
`metadata/figure_generation_verification.json` records the AUC/count consistency checks.

## Code licence and source-data terms

No code reuse licence is included in this supplied package. Public visibility
alone does not grant an open-source licence; the authors should confirm the
repository licence and add an appropriate code licence if absent before the
final software release. See
`LICENSE_STATUS.md`. PhysioNet's Restricted Health Data License governs the
source data separately and is not a licence for this code. No source data are
included in this package, and a code licence cannot authorize redistributing
the restricted dataset or derived participant-level files.
