from app.schema import normalize_scene, apply_patches

def test_untrusted_attributes_cannot_become_svg():
    scene=normalize_scene({"background":"url(https://bad.invalid)","objects":[
        {"id":"<img src=x onerror=alert(1)>","type":"sun", "bbox":[1,2,-20,10],
         "fill":"javascript:alert(1)","params":{"onload":"evil","rotation":999}}]})
    obj=scene['objects'][0]
    assert '<' not in obj['id']
    assert obj['fill'].startswith('#')
    assert obj['params']['rotation']==180
    assert obj['bbox'][2]==5
    assert 'onload' not in obj['params']


def test_patch_only_targets_known_objects():
    scene=normalize_scene({"objects":[{"id":"a","type":"koi","bbox":[0,0,200,100]}]})
    updated,changed=apply_patches(scene,[{"id":"missing","fill":"#000000"},
                                          {"id":"a","bbox":[5,10,220,110],"params":{"rotation":-15}}])
    assert changed==['a']
    assert updated['objects'][0]['bbox']==[5,10,220,110]
    assert scene['objects'][0]['bbox']==[0,0,200,100]
