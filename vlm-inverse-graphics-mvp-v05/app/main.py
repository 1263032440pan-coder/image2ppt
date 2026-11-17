"""Run: uvicorn app.main:app --reload"""
from __future__ import annotations
import asyncio
import json
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from .upload import to_jpeg_bytes
ROOT=Path(__file__).resolve().parents[1]

load_dotenv(ROOT/'.env')
app=FastAPI(title='Inverse Graphics MVP',version='0.4.0')
RUNTIME=ROOT/'.runtime'
RUNTIME.mkdir(exist_ok=True)
JOBS: dict[str,dict]={}
ALLOWED_FILES={'original.jpg','observation.json','scene.json','geometry.json','color.json','detail.json','region_metrics.json','color_heatmap.png','preview.svg','render.png','report.json'}

@app.get('/')
def index():
    return FileResponse(ROOT/'web/index.html')

@app.get('/api/health')
def health():
    return {'status':'ok','model':os.getenv('MOMA_MODEL','deepseek-v4.1-flash-1017'),
            'api_configured':bool(os.getenv('MOMA_API_KEY')),
            'max_iterations':int(os.getenv('MAX_ITERATIONS','3')),
            'demo_mode_available':True}


def progress_callback(job_id: str):
    def update(stage: str,pct: int,message: str):
        JOBS[job_id].update({'status':('finalizing' if stage=='completed' else stage),'progress':pct,'message':message})
    return update


def process(job_id: str,image: bytes,iterations: int,demo: bool,geometry_enabled: bool,
            color_enabled: bool,detail_enabled: bool):
    """One process per job: safer than reusing native OpenCV/SLIC/Cairo states."""
    import subprocess
    import sys
    import time
    directory=RUNTIME/job_id
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'original.jpg').write_bytes(image)
    command=[sys.executable,'-m','app.job_worker',str(directory),str(iterations),
             str(demo).lower(),str(geometry_enabled).lower(),
             str(color_enabled).lower(),str(detail_enabled).lower()]
    proc=None
    try:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        deadline=time.monotonic()+900
        while proc.poll() is None:
            progress_file=directory/'progress.json'
            if progress_file.is_file():
                try:
                    data=json.loads(progress_file.read_text(encoding='utf-8'))
                    JOBS[job_id].update(status=data['stage'],progress=data['progress'],message=data['message'])
                except (OSError,ValueError,KeyError):
                    pass
            if time.monotonic()>deadline:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill()
                raise TimeoutError('Job reached the 15-minute safety limit')
            time.sleep(.45)
        if proc.returncode != 0:
            error_file=directory/'error.json'
            message='Worker process failed'
            if error_file.is_file():
                try: message=json.loads(error_file.read_text(encoding='utf-8')).get('error',message)
                except (ValueError,OSError): pass
            raise RuntimeError(message)
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        JOBS[job_id].update(report=report,status='completed',progress=100,
                            message='SVG、PNG、Color IR、Detail IR 与评分报告已生成')
    except Exception as exc:
        JOBS[job_id].update(status='failed',message=f'{type(exc).__name__}: {str(exc)[:450]}',progress=100)

@app.post('/api/jobs',status_code=202)
async def submit(image: UploadFile=File(...),iterations: int=Form(2),demo: bool=Form(False),
                 geometry_enabled: bool=Form(True),color_enabled: bool=Form(True),
                 detail_enabled: bool=Form(True)):
    if not 1<=iterations<=3:
        raise HTTPException(422,'Iterations must be 1-3.')
    if not demo and not os.getenv('MOMA_API_KEY'):
        raise HTTPException(400,'MOMA_API_KEY is not configured. Add it to .env or select demo mode.')
    if any(j.get('status') not in ('completed','failed') for j in JOBS.values()):
        raise HTTPException(429,'Local MVP supports one active job at a time.')
    raw=await image.read(10*1024*1024+1)
    try:
        converted=to_jpeg_bytes(raw)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    job_id=uuid.uuid4().hex[:16]
    JOBS[job_id]={'id':job_id,'status':'queued','progress':0,'message':'准备处理图像', 'demo':demo}
    # Local MVP: in-memory task state. External worker/queue needed for production.
    asyncio.create_task(asyncio.to_thread(process,job_id,converted,iterations,demo,geometry_enabled,color_enabled,detail_enabled))
    return JOBS[job_id]

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404,'Job not found (tasks are in memory; restart clears state).')
    return JOBS[job_id]

@app.get('/api/jobs/{job_id}/files/{filename}')
def get_file(job_id: str,filename: str):
    if job_id not in JOBS or filename not in ALLOWED_FILES:
        raise HTTPException(404,'File not found')
    target=RUNTIME/job_id/filename
    if not target.is_file():
        raise HTTPException(404,'File not ready')
    if filename.endswith('.svg'):
        return FileResponse(target,media_type='image/svg+xml',headers={'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'; sandbox"})
    return FileResponse(target)

@app.get('/api/sample')
def sample():
    return FileResponse(ROOT/'examples/koi_ukiyoe.jpg',media_type='image/jpeg')
