"""Whole-plant behavior stays separate from the protected single-leaf pipeline."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import config
import plant_features as pf
from model_contract import validate_bundle, library_versions, explain_logistic
from train_wholeplant import inventory, train

ROOT = Path(__file__).resolve().parents[1]


def scene(tmp_path, kind='plant'):
    image = np.full((180,240,3),220,np.uint8)
    if kind=='plant':
        for center in [(70,40),(155,45),(95,95),(160,135)]:
            cv2.ellipse(image,center,(20,11),15,0,360,(35,155,55),-1)
    elif kind=='yellow': cv2.circle(image,(100,85),40,(20,180,180),-1)
    elif kind=='clipped': cv2.ellipse(image,(0,90),(35,65),0,0,360,(30,160,50),-1)
    path=tmp_path/f'{kind}.png'; cv2.imwrite(str(path),image)
    return path


def test_multiple_foliage_regions_are_retained(tmp_path):
    image,mask,_,quality=pf.prepare_plant(scene(tmp_path))
    assert quality['passed'] and quality['components']==4
    assert mask[40,70] and mask[135,160] and not mask[0,0]
    features=pf.extract_features(image,mask)
    assert list(features)==pf.FEATURE_NAMES and len(features)==24
    assert np.isfinite(list(features.values())).all()
    assert features['g_mean'] > features['r_mean']
    assert 0 <= features['chrom_g_mean'] <= 1
    assert -1 <= features['gli_mean'] <= 1


@pytest.mark.parametrize('kind',['blank','yellow','clipped'])
def test_reject_blank_accept_yellow_flag_clipped(tmp_path,kind):
    features,visuals,_,quality=pf.analyze_plant_full(scene(tmp_path,kind))
    assert quality['passed'] == (kind!='blank')
    assert bool(features) == quality['passed']
    assert visuals['mask'].startswith('data:image/jpeg;base64,')
    if kind=='clipped': assert quality['border_contact'] and quality['warnings']


def test_empty_mask_and_decode_errors(tmp_path):
    with pytest.raises(ValueError,match='Empty'): pf.extract_features(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8))
    bad=tmp_path/'bad.jpg';bad.write_bytes(b'not an image')
    with pytest.raises(ValueError,match='decoded'): pf.prepare_plant(bad)


def test_tiny_uniform_green_rejected(tmp_path):
    path=tmp_path/'tiny.png'; cv2.imwrite(str(path),np.full((8,8,3),(0,200,0),np.uint8))
    _,_,_,q=pf.prepare_plant(path)
    assert not q['passed'] and 'image_too_small' in q['reason_codes'] and 'uniform_image' in q['reason_codes']


def test_identity_and_warm_background_limit(tmp_path):
    image=np.full((100,100,3),(65,110,180),np.uint8)
    cv2.rectangle(image,(30,30),(70,70),(0,180,0),-1)
    path=tmp_path/'object.png';cv2.imwrite(str(path),image)
    _,mask,_,q=pf.prepare_plant(path)
    assert not mask[0,0] and mask[50,50]
    # Known limitation: a green non-plant object can pass. Do not pretend semantic validation.
    assert q['passed']


def test_extractor_reproducibility_and_empty_bands(tmp_path):
    path=scene(tmp_path)
    first=pf.analyze_plant_full(path); second=pf.analyze_plant_full(path)
    assert first==second
    mask=np.zeros((100,100),np.uint8);mask[30,30:40]=255
    features=pf.extract_features(np.full((100,100,3),(10,150,40),np.uint8),mask)
    assert np.isfinite(list(features.values())).all()


def test_manifest_no_invented_identity_and_conflicts(tmp_path):
    root=tmp_path/'data'; (root/'Healthy').mkdir(parents=True)
    source=scene(tmp_path); (root/'Healthy'/'a.png').write_bytes(source.read_bytes())
    rows,frame,_=inventory(root)
    assert rows[0]['plant_id']==rows[0]['session_id']==rows[0]['label_method']==''
    (root/'Healthy'/'copy.png').write_bytes(source.read_bytes())
    rows,frame,_=inventory(root);assert len(frame)==1 and rows[1]['duplicate_of']
    (root/'Disease_Confounded').mkdir(); (root/'Disease_Confounded'/'bad.png').write_bytes(source.read_bytes())
    with pytest.raises(ValueError,match='Cross-label duplicate'): inventory(root)


def test_training_requires_explicit_development_mode_and_samples(tmp_path):
    with pytest.raises(ValueError,match='development-only'): train(tmp_path,tmp_path/'out')
    with pytest.raises(ValueError,match='insufficient classes/groups'): train(tmp_path,tmp_path/'out',True)
    assert not (tmp_path/'out').exists()


def deployment_bundle():
    deployment=json.loads((ROOT/'wholeplant_deployment_v1.json').read_text())
    bundle=joblib.load(ROOT/deployment['model'])
    if bundle['library_versions'] != library_versions():
        pytest.skip('Saved deployment requires its exact Python 3.13 training environment; mismatch refusal is intentional')
    return deployment,bundle


def test_saved_model_contract_and_coefficients():
    deployment,bundle=deployment_bundle();validate_bundle(bundle)
    frame=pd.read_csv((ROOT/deployment['model']).parent/'features.csv')
    X=frame[pf.FEATURE_NAMES].iloc[:1]; model=bundle['all_models']['Logistic Regression']
    explanation=explain_logistic(model,X,model.predict(X)[0])
    assert explanation['available']
    assert sum(r['contribution'] for r in explanation['contributions'])+explanation['intercept']==pytest.approx(explanation['decision_score'])
    bad=copy.copy(bundle);bad['preprocessing_version']='wrong'
    with pytest.raises(ValueError,match='preprocessing_version'):validate_bundle(bad)
    bad=copy.copy(bundle);bad['feature_schema']=list(reversed(pf.FEATURE_NAMES))
    with pytest.raises(ValueError,match='feature_schema'):validate_bundle(bad)


def test_wholeplant_flask_all_images_and_preview(tmp_path):
    deployment,_=deployment_bundle()
    env={**os.environ,'LECOR_ANALYSIS_MODE':'wholeplant','LECOR_PREVIEW_ONLY':'false',
         'LECOR_MODEL_PATH':deployment['model'],'LECOR_BENCHMARK_PATH':deployment['report'], 'PYTHONDONTWRITEBYTECODE':'1'}
    script='''
import io,json
from collections import Counter
from pathlib import Path
import app
assert app.primary_model is not None,app.model_load_error
client=app.app.test_client()
assert b'Experimental whole-plant' in client.get('/').data
assert b'whole-plant photo' in client.get('/').data
outcomes=Counter(); first=None
for path in sorted(Path('dataset_wholeplant_v1_20261013').rglob('*.*')):
 response=client.post('/predict',data={'image':(io.BytesIO(path.read_bytes()),'../../'+path.name)})
 if path.suffix.lower() not in app.ALLOWED_EXTENSIONS:
  assert response.status_code==415 and response.json['code']=='unsupported_extension'
  continue
 assert response.status_code==200,(path,response.json)
 data=response.json;outcomes[data['outcome']]+=1
 assert data['quality']['passed'] and data['visualizations']['mask']
 if data['outcome']=='classified':
  assert data['model_explanation']['available'] == (data['best_model_name']=='Logistic Regression')
  assert sum(data['probabilities'].values())>0.99
  if data['prediction']=='Disease_Confounded':assert data['visual_stress_indicator'] is None
 else: assert data['outcome']=='uncertain' and data['prediction'] is None
 if first is None:first=(path,data)
path,data=first
again=client.post('/predict',data={'image':(io.BytesIO(path.read_bytes()),path.name)}).json
assert again==data
app.primary_model=None
preview=client.post('/predict',data={'image':(io.BytesIO(path.read_bytes()),path.name),'mask_preview':'true'})
assert preview.status_code==200 and preview.json['code']=='mask_review'
assert preview.json['prediction'] is None and preview.json['visualizations']['mask']
assert app.load_model_bundle()
app.app.config['PREVIEW_ONLY']=True
preview=client.post('/predict',data={'image':(io.BytesIO(path.read_bytes()),path.name)}).json
assert preview['outcome']=='preview_only' and preview['prediction'] is None
app.app.config['PREVIEW_ONLY']=False
assert client.post('/predict').status_code==400
assert client.post('/predict',data={'image':(io.BytesIO(b'bad'),'x.jpg')}).status_code==400
assert not app.load_model_bundle('leaf_stress_model.pkl')
print(json.dumps(dict(outcomes)))
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
