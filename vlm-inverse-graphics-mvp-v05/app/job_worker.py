"""Run one complete VLM-to-SVG job in an isolated process.

Main server never imports expensive image-processing libraries on request threads.
This makes repeated uploads robust against native library state between jobs.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from dotenv import load_dotenv


def _write_atomic(path: Path, data: dict) -> None:
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,path)


def main(args: list[str]) -> int:
    if len(args)!=6:
        raise SystemExit('Usage: python -m app.job_worker job_dir iterations demo geometry color detail')
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    from .pipeline import run_pipeline
    job_dir=Path(args[0]).resolve()
    source=job_dir/'original.jpg'
    if not source.is_file():
        raise FileNotFoundError('Input image missing')
    def progress(stage,percentage,message):
        _write_atomic(job_dir/'progress.json',{'stage':stage,'progress':percentage,'message':message})
    try:
        run_pipeline(job_dir,source.read_bytes(),int(args[1]),args[2]=='true',progress,
                     geometry_enabled=args[3]=='true',color_enabled=args[4]=='true',
                     detail_enabled=args[5]=='true')
        return 0
    except Exception as exc:
        _write_atomic(job_dir/'error.json',{'error':f'{type(exc).__name__}: {str(exc)[:450]}'})
        return 1


if __name__=='__main__':
    sys.exit(main(sys.argv[1:]))
