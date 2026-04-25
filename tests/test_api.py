import pytest

from app.api import app

SAMPLE_INPUT = {
    'LIMIT_BAL': 20000, 'SEX': 2, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 24,
    'PAY_0': 2, 'PAY_2': 2, 'PAY_3': -1, 'PAY_4': -1, 'PAY_5': -2, 'PAY_6': -2,
    'BILL_AMT1': 3913, 'BILL_AMT2': 3102, 'BILL_AMT3': 689,
    'BILL_AMT4': 0, 'BILL_AMT5': 0, 'BILL_AMT6': 0,
    'PAY_AMT1': 0, 'PAY_AMT2': 689, 'PAY_AMT3': 0,
    'PAY_AMT4': 0, 'PAY_AMT5': 0, 'PAY_AMT6': 0,
}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json() == {'status': 'healthy'}


def test_predict_valid_v1(client):
    payload = {**SAMPLE_INPUT, 'version': 'v1'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body['prediction'] in (0, 1)
    assert 0.0 <= body['probability'] <= 1.0
    assert body['model_version'] == 'v1'


def test_predict_valid_v2_via_query(client):
    response = client.post('/predict?version=v2', json=SAMPLE_INPUT)
    assert response.status_code == 200
    body = response.get_json()
    assert body['model_version'] == 'v2'


def test_predict_default_version(client):
    response = client.post('/predict', json=SAMPLE_INPUT)
    assert response.status_code == 200
    assert response.get_json()['model_version'] == 'v1'


def test_predict_missing_field(client):
    response = client.post('/predict', json={'LIMIT_BAL': 20000})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_predict_non_numeric_field(client):
    payload = {**SAMPLE_INPUT, 'AGE': 'twenty-four'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_predict_unknown_version(client):
    response = client.post('/predict?version=v99', json=SAMPLE_INPUT)
    assert response.status_code == 400
    assert 'Unknown model version' in response.get_json()['error']


def test_predict_non_json_body(client):
    response = client.post('/predict', data='not-json', content_type='text/plain')
    assert response.status_code == 400
