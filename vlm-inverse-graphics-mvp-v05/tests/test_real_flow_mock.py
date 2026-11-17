"""Exercise the real VLM orchestration with deterministic mocked network responses."""
from app.pipeline import run_pipeline,to_jpeg_bytes,ROOT
import app.pipeline as pipeline

class FakeClient:
    def json(self,prompt,images=None,max_tokens=4096):
        if prompt.startswith('你是二维逆向图形学'):
            assert images and len(images)==1
            return {"summary":"3 koi and a bridge","style":{},"objects":[
                {"id":"sun","type":"sun","bbox":[700,100,150,150]},
                {"id":"bridge","type":"bridge","bbox":[550,290,350,140]}]}
        if prompt.startswith('你是 Visual Graphics Design Agent'):
            assert not images
            return {"background":"#EAE0C3","objects":[
                {"id":"sun","type":"sun","bbox":[700,100,150,150],"layer":1,"fill":"#DA6048"},
                {"id":"bridge","type":"bridge","bbox":[550,290,350,140],"layer":2,"fill":"#CA5344"}]}
        if prompt.startswith('你是负责几何校对'):
            assert images and len(images)==2
            return {"summary":"Move sun right","patches":[{"id":"sun","bbox":[710,100,145,145]}]}
        raise AssertionError('unexpected agent call')

def test_agent_observe_design_render_review(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,'MessagesClient',FakeClient)
    # Force a review iteration so the mock verifies the full feedback path;
    # the improved continuous colour field can otherwise pass the stop gate.
    original_regions=pipeline.region_comparison
    def force_review(*args, **kwargs):
        result=original_regions(*args, **kwargs)
        result['global']['mean_delta_e00']=10.0
        return result
    monkeypatch.setattr(pipeline,'region_comparison',force_review)
    img=to_jpeg_bytes((ROOT/'examples/koi_ukiyoe.jpg').read_bytes())
    stages=[]
    report=run_pipeline(tmp_path,img,2,False,lambda s,p,m:stages.append(s))
    assert report['mode']=='vlm'
    assert report['iterations_completed']==2
    assert 'observing' in stages and 'designing' in stages and 'reviewing' in stages
    assert (tmp_path/'review_1.json').exists()
    assert (tmp_path/'iter_2.svg').exists()
