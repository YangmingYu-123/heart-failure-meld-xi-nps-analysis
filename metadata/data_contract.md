# Source-field mapping and private inputs

`prepare_physionet.py` maps the native PhysioNet version 1.2 `dat.csv` to the
44 coded fields required by this analysis. The exact names, transformations,
and units/category codes are listed in `physionet_variable_mapping.csv`.

Numeric source fields are copied without rounding. Explicit category mappings
are applied to age groups, sex, NYHA, heart-failure type and in-hospital
outcome. Unknown categories or nonbinary endpoint flags fail with an error.
Identifiers must be present and unique. Recorded endpoint flags are used to
construct `COMP28`, `COMP3M` and `COMP6M`; the time fields are retained for QC
and are not used to redefine those outcomes.

The age groups are the repository's interval labels: `(21,29]`, `(29,39]`,
`(39,49]`, `(49,59]`, `(59,69]`, `(69,79]`, `(79,89]`, `(89,110]`, mapped to
ordered categories 1–8. They are not exact ages. Sex is male=1/female=2;
NYHA retains II=2/III=3/IV=4; heart-failure type is left=1/right=2/both=3.

The downstream loader logs and sets non-positive values of selected lab,
vital-sign and BMI fields to missing and sets negative neutrophil counts to
missing. MELD-XI and NPS are recalculated from the component variables by the
same shared implementation. The historical score columns are not required
to derive exposures from native data.

The generated TSV, cleaning logs, imputed datasets and predictions are
participant-level data. Keep them private under the signed data-use agreement.
No patient examples or rows are included in the public code package.
