"""Versioned artifact contract; trusted pickle deserialization is still required."""
from datetime import datetime, timezone
from importlib.metadata import version
import platform
import re

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import config
from dataset_manifest import FEATURE_NAMES, CLASSES

SCHEMA_VERSION = 'lecor-features-v1'
EXTRACTOR_VERSION = 'lecor-15-features-v1'
PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'opencv-python', 'pandas', 'Pillow')


def library_versions():
    return {'python': platform.python_version(), **{p: version(p) for p in PACKAGES}}


def training_metadata(dataset_hash, cv_protocol):
    return dict(schema_version=SCHEMA_VERSION, extractor_version=EXTRACTOR_VERSION,
                preprocessing_version=config.preprocessing_signature()['version'],
                feature_schema=list(FEATURE_NAMES), dataset_hash=dataset_hash,
                dataset_hash_definition='SHA-256 of the exact training feature CSV bytes',
                library_versions=library_versions(), training_date=datetime.now(timezone.utc).isoformat(),
                cv_protocol=cv_protocol)


def validate_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError('Versioned model bundle required')
    schema, extractor, names = SCHEMA_VERSION, EXTRACTOR_VERSION, list(FEATURE_NAMES)
    preprocessing, mask_quality = config.preprocessing_signature(), config.mask_quality_settings()
    if bundle.get('analysis_mode') == 'wholeplant':
        import plant_features as plant
        schema, extractor, names = plant.SCHEMA_VERSION, plant.EXTRACTOR_VERSION, plant.FEATURE_NAMES
        preprocessing = plant.preprocessing_signature()
        mask_quality = preprocessing
    expected = {'schema_version': schema, 'extractor_version': extractor,
                'preprocessing_version': preprocessing['version'],
                'feature_schema': names, 'feature_names_in_': names,
                'preprocessing': preprocessing, 'mask_quality': mask_quality}
    for key, value in expected.items():
        if bundle.get(key) != value:
            raise ValueError(f'Model {key} missing or mismatched; compatible retraining required')
    if not re.fullmatch(r'[0-9a-f]{64}', bundle.get('dataset_hash', '')):
        raise ValueError('Model dataset hash missing or invalid')
    if not bundle.get('dataset_hash_definition') or not isinstance(bundle.get('cv_protocol'), dict) or not bundle['cv_protocol']:
        raise ValueError('Model training provenance missing')
    if datetime.fromisoformat(bundle.get('training_date', '')).tzinfo is None:
        raise ValueError('Training date must include timezone')
    if bundle.get('library_versions') != library_versions():
        raise ValueError('Model library versions differ from runtime; use the recorded training environment')
    labels = bundle.get('classes_', [])
    if len(labels) != len(CLASSES) or set(labels) != set(CLASSES):
        raise ValueError('Model class schema mismatch')
    models = bundle.get('all_models')
    if not isinstance(models, dict) or not models:
        raise ValueError('Model comparison collection missing')
    for model in [bundle.get('primary_model'), *models.values()]:
        if (list(getattr(model, 'feature_names_in_', [])) != names
                or getattr(model, 'n_features_in_', None) != len(names)
                or list(getattr(model, 'classes_', [])) != labels
                or not callable(getattr(model, 'predict', None))):
            raise ValueError('Estimator feature/class schema mismatch')
    if not callable(getattr(bundle['primary_model'], 'predict_proba', None)):
        raise ValueError('Primary estimator requires class probabilities')


def explain_logistic(model, X, predicted_class):
    """Exact additive multinomial decision score, not a probability or cause."""
    unavailable = {'available': False, 'message': 'Coefficient explanation is available only for a StandardScaler + multiclass Logistic Regression pipeline.'}
    if not isinstance(model, Pipeline) or len(model.steps) != 2:
        return unavailable
    scaler, classifier = [step for _, step in model.steps]
    if not isinstance(scaler, StandardScaler) or not isinstance(classifier, LogisticRegression):
        return unavailable
    if not scaler.with_mean or not scaler.with_std or classifier.coef_.shape[0] != len(classifier.classes_):
        return unavailable
    if list(X.columns) != list(model.feature_names_in_):
        raise ValueError('Explanation feature order mismatch')
    index = list(classifier.classes_).index(predicted_class)
    z = scaler.transform(X)[0]
    coefficients = classifier.coef_[index]
    contributions = coefficients * z
    intercept = float(classifier.intercept_[index])
    score = float(model.decision_function(X)[0, index])
    if not np.isfinite(contributions).all() or not np.isclose(contributions.sum() + intercept, score):
        raise ValueError('Explanation does not reconstruct the model decision score')
    return {'available': True, 'predicted_class': predicted_class,
            'message': 'Model explanation, not a biological cause. Each term is coefficient in standardized feature space × standardized value. Terms sum with the intercept to this class decision score; probabilities depend on all class scores. Correlated features can redistribute coefficients.',
            'intercept': intercept, 'decision_score': score,
            'contributions': [{'feature': name, 'standardized_value': float(value), 'coefficient': float(coef), 'contribution': float(term)}
                              for name, value, coef, term in zip(X.columns, z, coefficients, contributions)]}
