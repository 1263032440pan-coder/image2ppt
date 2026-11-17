from app.model import image_part,parse_json,extract_text

def test_parser_mixed_text():
    assert parse_json('```json\n{"ok":true}\n```')['ok'] is True
    assert parse_json('<think>draft</think> {"objects":[]}')['objects']==[]
    assert extract_text({'content':[{'type':'text','text':'hello'}]})=='hello'
    assert extract_text({'choices':[{'message':{'content':'world'}}]})=='world'

def test_multimodal_content_encodings():
    a=image_part(b'hello','anthropic')
    b=image_part(b'hello','openai')
    assert a['source']['media_type']=='image/jpeg'
    assert b['image_url']['url'].startswith('data:image/jpeg;base64,')
