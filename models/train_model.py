import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = 'data/UCI_Credit_Card.csv'
TARGET = 'default.payment.next.month'


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['ID', TARGET])
    y = df[TARGET]

    return X, y


def build_pipeline(n_estimators, max_depth, learning_rate):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=RANDOM_STATE,
        )),
    ])


def report(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f'\n=== {name} ===')
    print(classification_report(y_test, y_pred, target_names=['no_default', 'default']))
    print('ROC AUC:', round(roc_auc_score(y_test, y_proba), 4))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))


def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print(f'Train: {len(X_train)} rows, Test: {len(X_test)} rows')
    print(f'Default rate in test: {y_test.mean():.3f}')

    # v1: неглубокое дерево
    pipe_v1 = build_pipeline(n_estimators=100, max_depth=3, learning_rate=0.1)
    pipe_v1.fit(X_train, y_train)
    report('v1', pipe_v1, X_test, y_test)
    joblib.dump(pipe_v1, 'models/model_v1.pkl')

    # v2: более глубокое дерево
    pipe_v2 = build_pipeline(n_estimators=200, max_depth=5, learning_rate=0.05)
    pipe_v2.fit(X_train, y_train)
    report('v2', pipe_v2, X_test, y_test)
    joblib.dump(pipe_v2, 'models/model_v2.pkl')

    # проверка воспроизводимости
    reloaded_v1 = joblib.load('models/model_v1.pkl')
    reloaded_v2 = joblib.load('models/model_v2.pkl')
    assert (reloaded_v1.predict(X_test) == pipe_v1.predict(X_test)).all(), 'v1 reload mismatch'
    assert (reloaded_v2.predict(X_test) == pipe_v2.predict(X_test)).all(), 'v2 reload mismatch'
    

    print('\n models/model_v1.pkl и models/model_v2.pkl сохранены.')


if __name__ == '__main__':
    main()