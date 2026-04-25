import random
import joblib
import pandas as pd

# Порядок колонок должен совпадать с тем, который использовался при обучении
FEATURE_NAMES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
]

MODELS = {
    'v1': joblib.load('models/model_v1.pkl'),
    'v2': joblib.load('models/model_v2.pkl'),
}

DEFAULT_VERSION = 'v1'

# Переключатель A/B-теста: при AB_ENABLED=True запросы без явной version делятся
# случайно 50/50 между v1 и v2. При AB_ENABLED=False все такие запросы идут на
# DEFAULT_VERSION (контрольное поведение, по умолчанию для штатной работы сервиса).
# Для запуска A/B-теста значение переключается на True. В production-среде управление
# выносится в переменную окружения (например, AB_ENABLED=true в docker-compose или
# в конфиге оркестратора).
AB_ENABLED = False


def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError('Input must be a JSON object')
    
    missing = [name for name in FEATURE_NAMES if name not in data]

    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    
    for name in FEATURE_NAMES:
        value = data[name]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'Field {name} must be numeric, got {type(value).__name__}')


def choose_version(explicit_version):
    if explicit_version:

        if explicit_version not in MODELS:
            raise ValueError(f'Unknown model version: {explicit_version}. Available: {list(MODELS.keys())}')

        return explicit_version

    if AB_ENABLED:
        return random.choice(['v1', 'v2'])

    return DEFAULT_VERSION


def predict(data, version=None):
    used_version = choose_version(version)
    model = MODELS[used_version]
    row = pd.DataFrame([[data[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    return prediction, probability, used_version