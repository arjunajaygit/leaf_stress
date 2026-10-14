import os
import subprocess
import sys

import pytest
from result_presentation import describe_result


@pytest.mark.parametrize('mode,expected',[('wholeplant',.5),('leaf',.6)])
def test_mode_default_threshold(mode,expected):
    env={**os.environ,'LECOR_ANALYSIS_MODE':mode}
    env.pop('LECOR_MIN_CLASS_PROBABILITY',None)
    result=subprocess.run([sys.executable,'-c','import config; print(config.MIN_CLASS_PROBABILITY)'],env=env,capture_output=True,text=True,check=True)
    assert float(result.stdout)==expected


def test_confounded_result_never_assigns_water_stress_level():
    result=describe_result('Disease_Confounded',.503)
    assert result['tentative'] and 'unclear' in result['title']
    assert 'cannot be assigned' in result['explanation']


@pytest.mark.parametrize('label',['Healthy','Moderate_Water_Stress','Severe_Water_Stress','Disease_Confounded'])
def test_every_category_has_explanation_and_action(label):
    result=describe_result(label,.8)
    assert all(result[k] for k in ('title','explanation','next_check','note'))
    assert result['reliability']=='Experimental visual estimate'
