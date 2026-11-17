from pathlib import Path
from PIL import Image
from io import BytesIO
from app.metrics import compare_images
from app.pipeline import to_jpeg_bytes,run_pipeline,ROOT

def test_identity_metrics():
    image=Image.new('RGB',(100,100),(20,50,140));out=BytesIO();image.save(out,format='PNG')
    scores=compare_images(out.getvalue(),out.getvalue())
    assert scores['score']>.99

def test_demo_end_to_end(tmp_path):
    sample=(ROOT/'examples/koi_ukiyoe.jpg').read_bytes()
    jpg=to_jpeg_bytes(sample)
    stages=[]
    report=run_pipeline(tmp_path,jpg,2,True,lambda a,b,c:stages.append(a))
    assert report['mode']=='demo'
    assert report['iterations_completed']==1
    assert 0<=report['best_score']<=1
    for file in ('original.jpg','observation.json','scene.json','preview.svg','render.png','report.json'):
        assert (tmp_path/file).exists()
    assert stages[-1]=='completed'
