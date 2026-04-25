import joblib
import numpy as np
import pandas as pd
import pytest

from app.model_handler import FEATURE_NAMES

SAMPLE_INPUT = {
    'LIMIT_BAL': 20000, 'SEX': 2, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 24,
    'PAY_0': 2, 'PAY_2': 2, 'PAY_3': -1, 'PAY_4': -1, 'PAY_5': -2, 'PAY_6': -2,
    'BILL_AMT1': 3913, 'BILL_AMT2': 3102, 'BILL_AMT3': 689,
    'BILL_AMT4': 0, 'BILL_AMT5': 0, 'BILL_AMT6': 0,
    'PAY_AMT1': 0, 'PAY_AMT2': 689, 'PAY_AMT3': 0,
    'PAY_AMT4': 0, 'PAY_AMT5': 0, 'PAY_AMT6': 0,
}


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_model_loads_and_predicts(version):
    model = joblib.load(f'models/model_{version}.pkl')
    row = pd.DataFrame([[SAMPLE_INPUT[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0, 1]

    assert prediction in (0, 1)
    assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_model_reload_gives_same_prediction(version):
    path = f'models/model_{version}.pkl'
    model_a = joblib.load(path)
    model_b = joblib.load(path)
    row = pd.DataFrame([[SAMPLE_INPUT[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    
    np.testing.assert_array_equal(model_a.predict(row), model_b.predict(row))
    np.testing.assert_array_almost_equal(model_a.predict_proba(row), model_b.predict_proba(row), decimal=10)
