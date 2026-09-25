"""Convenience runner for the complete analysis sequence.

This runner only configures paths and launches the archived scripts. It does
not download PhysioNet data, alter the source file, or publish any output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HFYC_NPS analyses in manuscript order.")
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", help="Authorized coded tab-delimited input")
    inputs.add_argument("--physionet", help="Authorized native dat.csv or project ZIP; prepares a private coded input")
    parser.add_argument("--dictionary", help="Optional private variable-dictionary .xlsx workbook")
    parser.add_argument("--float-precision",choices=['high','round_trip'],help="Use round_trip when rerunning a prepared 17-digit TSV; native mode sets it automatically")
    parser.add_argument("--out", required=True, help="Writable results root")
    args = parser.parse_args()

    env = os.environ.copy()
    result_root=Path(args.out).expanduser().resolve()
    if args.physionet:
        coded=result_root/'private_input'/'physionet_coded.tsv'
        subprocess.run([sys.executable,str(ROOT/'prepare_physionet.py'),
                        '--source',str(Path(args.physionet).expanduser().resolve()),
                        '--out',str(coded),'--report',str(result_root/'preparation_metadata.json')],check=True,env=env)
        env['HFYC_FLOAT_PRECISION']='round_trip'
    else:
        coded=Path(args.data).expanduser().resolve()
        if args.float_precision:
            env['HFYC_FLOAT_PRECISION']=args.float_precision
        else:
            env.pop('HFYC_FLOAT_PRECISION',None)
    env.update({"HFYC_DATA": str(coded), "HFYC_OUT": str(result_root)})
    if args.dictionary:
        env['HFYC_DICT']=str(Path(args.dictionary).expanduser().resolve())
    else:
        env.pop('HFYC_DICT',None)
    scripts = [
        "analysis_cc.py",
        "analysis_mi.py",
        "pool_mi_model_df.py",
        "analysis_univariable.py",
        "analysis_exploratory.py",
        "analysis_death.py",
        "bootstrap_analysis.py",
    ]
    for script in scripts:
        command = [sys.executable, str(ROOT / script)]
        if script == "analysis_cc.py":
            command.extend(["--data", env["HFYC_DATA"], "--out", env["HFYC_OUT"]])
            if args.dictionary:
                command.extend(["--dictionary", env["HFYC_DICT"]])
        print(f"Running {script} ...", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
