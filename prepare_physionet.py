"""Build the minimum coded input directly from the authorized PhysioNet v1.2 data.

Accepts the downloaded project ZIP or its native dat.csv. No download or network
request is performed. The derived output is participant-level controlled data;
keep it private. Mapping was checked against the authors' prior coded extract.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd

NUMERIC = {
    'ID':'inpatient.number', 'PULSE':'pulse', 'SBP':'systolic.blood.pressure',
    'DBP':'diastolic.blood.pressure', 'BMI':'BMI', 'LVEF':'LVEF',
    'TBIL':'total.bilirubin', 'CREAT':'creatinine.enzymatic.method',
    'ALB':'albumin', 'TC':'cholesterol', 'NEUT_C':'neutrophil.count',
    'LYM':'lymphocyte.count', 'MONO_C':'monocyte.count', 'BNP':'brain.natriuretic.peptide',
    'NA':'sodium', 'HB':'hemoglobin', 'EGFR':'glomerular.filtration.rate',
    'DIABETES':'diabetes', 'COPD':'Chronic.obstructive.pulmonary.disease',
    'CKD':'moderate.to.severe.chronic.kidney.disease', 'LIVER_DZ':'liver.disease',
    'CCI':'CCI.score', 'MI':'myocardial.infarction', 'CVD':'cerebrovascular.disease',
    'PVD':'peripheral.vascular.disease', 'VISIT_N':'visit.times',
    'DEATH28':'death.within.28.days', 'READM28':'re.admission.within.28.days',
    'DEATH3M':'death.within.3.months', 'READM3M':'re.admission.within.3.months',
    'DEATH6M':'death.within.6.months', 'READM6M':'re.admission.within.6.months',
    'DEATHTM':'time.of.death..days.from.admission.',
    'READMTM':'re.admission.time..days.from.admission.',
}
CATEGORICAL = {
    'AGECAT': ('ageCat', {'(21,29]':1, '(29,39]':2, '(39,49]':3, '(49,59]':4,
                         '(59,69]':5, '(69,79]':6, '(79,89]':7, '(89,110]':8}),
    'GENDER': ('gender', {'Male':1, 'Female':2}),
    'NYHA': ('NYHA.cardiac.function.classification', {'II':2, 'III':3, 'IV':4}),
    'HFTYPE': ('type.of.heart.failure', {'Left':1, 'Right':2, 'Both':3}),
    'HOSPOUT': ('outcome.during.hospitalization', {'Alive':1, 'Dead':2, 'DischargeAgainstOrder':3}),
}
UNITS = {'PULSE':'beats/min', 'SBP':'mmHg', 'DBP':'mmHg', 'BMI':'kg/m2', 'LVEF':'%',
         'TBIL':'micromol/L', 'CREAT':'micromol/L', 'ALB':'g/L', 'TC':'mmol/L',
         'NEUT_C':'10^9/L', 'LYM':'10^9/L', 'MONO_C':'10^9/L', 'BNP':'pg/mL',
         'NA':'mmol/L', 'HB':'g/L', 'EGFR':'mL/min/1.73 m2', 'DEATHTM':'days from admission',
         'READMTM':'days from admission'}


def read_native(path):
    path=Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.rsplit('/',1)[-1]=='dat.csv']
            if len(names)!=1:
                raise ValueError('Expected exactly one native dat.csv within the project archive')
            payload=z.read(names[0])
    else:
        payload=path.read_bytes()
    return pd.read_csv(io.BytesIO(payload)), hashlib.sha256(payload).hexdigest()


def mapping_rows():
    rows=[{'coded_field':k,'native_field':v,'transformation':'numeric copy; missing values preserved',
           'units_or_coding':UNITS.get(k,'see native data dictionary')} for k,v in NUMERIC.items()]
    rows.extend({'coded_field':k,'native_field':v,'transformation':'explicit category map',
                 'units_or_coding':json.dumps(codes,ensure_ascii=False)} for k,(v,codes) in CATEGORICAL.items())
    for period in ['28','3M','6M']:
        rows.append({'coded_field':'COMP'+period,'native_field':f'DEATH{period} and READM{period}',
                     'transformation':'logical OR of recorded endpoint flags; not recomputed from time fields',
                     'units_or_coding':'0=no recorded component; 1=at least one component'})
    for score in ['MELD_XI','NPS']:
        rows.append({'coded_field':score,'native_field':'score components listed above',
                     'transformation':'recalculated using the shared analysis_cc.clean_frame implementation',
                     'units_or_coding':'points; score formula documented in manuscript'})
    return pd.DataFrame(rows)


def prepare(native):
    required=set(NUMERIC.values()) | {v for v,_ in CATEGORICAL.values()}
    missing=required-set(native.columns)
    if missing:
        raise ValueError(f'Native source is missing required fields: {sorted(missing)}')
    out=pd.DataFrame(index=native.index)
    for target,source in NUMERIC.items():
        out[target]=pd.to_numeric(native[source],errors='raise')
    for target,(source,codes) in CATEGORICAL.items():
        unknown=set(native[source].dropna().unique())-set(codes)
        if unknown:
            raise ValueError(f'Unrecognized categories in {source}: {sorted(unknown)}')
        out[target]=native[source].map(codes)
    for period in ['28','3M','6M']:
        pair=out[['DEATH'+period,'READM'+period]]
        if not pair.isin([0,1]).all().all():
            raise ValueError(f'Endpoint flags must be complete binary values for period {period}')
        out['COMP'+period]=pair.eq(1).any(axis=1).astype(int)
    if out.ID.isna().any() or out.ID.duplicated().any():
        raise ValueError('Missing or duplicated patient identifiers in native input')
    # Shared cleaning makes every score definition identical to downstream analyses.
    from analysis_cc import clean_frame
    _,clean,_=clean_frame(out)
    out['MELD_XI']=clean.MELD_XI_recalc
    out['NPS']=clean.NPS_recalc
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Authorized native dat.csv or project ZIP')
    p.add_argument('--out',type=Path,required=True,help='Private output TSV; never upload this file')
    p.add_argument('--report',type=Path,help='Optional aggregate provenance JSON')
    args=p.parse_args()
    native,digest=read_native(args.source)
    coded=prepare(native)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        raise FileExistsError('Refusing to replace an existing private input; choose a new --out path')
    coded.to_csv(args.out,sep='\t',index=False,na_rep='NA',float_format='%.17g')
    report={'source_basename':args.source.name,'native_dat_csv_sha256':digest,
            'native_rows':len(native),'native_columns':len(native.columns),
            'coded_rows':len(coded),'coded_columns':len(coded.columns),
            'missing_ID':int(coded.ID.isna().sum()),'duplicate_ID':int(coded.ID.duplicated().sum()),
            'recorded_endpoint_counts':coded[['COMP28','COMP3M','COMP6M','DEATH6M','READM6M']].sum().to_dict(),
            'coded_tsv_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
            'private_output':True,'native_record_version':'1.2; verify against access/download record'}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
