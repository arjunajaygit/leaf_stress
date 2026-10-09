"""Legacy fixture is test-only; production refuses this artifact's missing metadata."""
import joblib
import pytest
import config

@pytest.fixture(autouse=True)
def legacy_api_fixture(request, monkeypatch):
    if request.module.__name__.split('.')[-1] not in {'test_stage1', 'test_stage2', 'test_stage3'}:
        return
    import app as web
    bundle = joblib.load(config.BASE_DIR / 'leaf_stress_model.pkl')
    for name, value in {'bundle': bundle, 'primary_model': bundle['primary_model'],
                        'all_models': bundle['all_models'], 'feature_names': bundle['feature_names_in_'],
                        'classes': bundle['classes_'], 'feature_stats': bundle['feature_stats'],
                        'model_load_error': None}.items():
        monkeypatch.setattr(web, name, value)
