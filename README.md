# Сервис прогнозирования дефолта по кредитным картам

Учебный проект по дисциплине "Внедрение моделей машинного обучения".

## Описание

Сервис принимает признаки клиента и возвращает вероятность дефолта в следующем расчётном месяце. Бинарный классификатор обучен на UCI Default of Credit Card Clients (30000 наблюдений, 23 признака, целевая переменная `default.payment.next.month`). Модель обёрнута в Flask-сервис с двумя эндпоинтами (`/health`, `/predict`) и упакована в Docker-образ. Поддерживаются две версии модели (v1 и v2) с возможностью A/B-теста через query-параметр или флаг внутри сервиса.

Цели проекта:
- Воспроизводимый pipeline обучения и сериализации модели
- Production-like Flask API с валидацией, JSON-логированием и тестами
- Контейнеризация и развёртывание через docker-compose с nginx-фронтом
- Документированный план A/B-теста v1 и v2

## Структура репозитория

```
ml_models_implementation_draft/
├── app/                      Flask-сервис
│   ├── __init__.py
│   ├── api.py                эндпоинты /health, /predict, JSON-логирование
│   └── model_handler.py      загрузка моделей, валидация, A/B-маршрутизация
├── models/                   обучение и артефакты модели
│   ├── train_model.py        обучение v1 и v2, сохранение в .pkl
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── tests/                    pytest-тесты модели и API
│   ├── test_model.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile            python:3.11-slim, EXPOSE 5000, healthcheck
│   └── nginx.conf            reverse proxy на ml-service:5000
├── docs/
│   ├── ARCHITECTURE.md       обоснование архитектуры, MLOps-концепты, бизнес-метрики
│   └── ab_test_plan.md       план A/B-теста v1 vs v2
├── data/
│   └── UCI_Credit_Card.csv   датасет (30000 строк)
├── docker-compose.yml        ml-service + nginx
├── requirements.txt
├── pyproject.toml            настройка pytest pythonpath
└── README.md
```

## Установка и локальный запуск

Требования: Python 3.11.

```bash
git clone https://github.com/a-kapset/ml_models_implementation_sp.git
cd ml_models_implementation_sp

# Виртуальное окружение
python -m venv .venv
source .venv/Scripts/activate    # Windows bash

# Зависимости
pip install -r requirements.txt

# Обучение моделей (создаёт models/model_v1.pkl и model_v2.pkl)
python models/train_model.py

# Запуск сервиса
python -m app.api
```

После старта сервис доступен по URL `http://localhost:5000`.

Запуск тестов:

```bash
pytest tests/ -v
```

## Запуск в Docker

Сборка образа из корня проекта:

```bash
docker build -t credit-default-service:v1.1.0 -f docker/Dockerfile .
```

Запуск контейнера:

```bash
docker run -d --name credit-default -p 5000:5000 credit-default-service:v1.1.0
```

Образ собран на `python:3.11-slim`, размер около 691 МБ, содержит код сервиса, обученные модели и зависимости. В Dockerfile настроен `HEALTHCHECK`, который оркестратор использует для маршрутизации трафика только на готовые реплики.

## Запуск через docker-compose (ml-service + nginx)

Docker-compose поднимает Flask-сервис и nginx как reverse proxy перед ним. Порт 5000 ml-service не публикуется на хост, внешний трафик идёт только через nginx на порт 80.

```bash
docker compose up -d --build
```

Проверка через nginx:

```bash
curl http://localhost/health
curl -X POST "http://localhost/predict?version=v1" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'
```

Остановка:

```bash
docker compose down
```

## Готовый образ в Docker Hub

Опубликованный образ доступен в Docker Hub:

- Репозиторий: https://hub.docker.com/r/artsiomk/ml-implementation
- Теги: `v1.1.0`, `latest`

Запуск без сборки:

```bash
docker pull artsiomk/ml-implementation:latest
docker run -d -p 5000:5000 artsiomk/ml-implementation:latest
```

## API

### GET /health

Проверка работоспособности.

```bash
curl http://localhost:5000/health
```

Ответ:

```json
{"status": "healthy"}
```

### POST /predict

Прогноз дефолта. Тело запроса - JSON с 23 числовыми признаками. Опциональное поле `version` (`v1` или `v2`) или query-параметр `?version=` выбирает версию модели; без указания версии запрос идёт на v1 по умолчанию.

Полный список признаков:

| Поле | Описание |
|---|---|
| `LIMIT_BAL` | лимит кредита, тайваньские доллары |
| `SEX` | пол (1 - мужской, 2 - женский) |
| `EDUCATION` | образование (1 - аспирантура, 2 - университет, 3 - школа, 4 - прочее) |
| `MARRIAGE` | семейное положение (1 - в браке, 2 - холост, 3 - прочее) |
| `AGE` | возраст в годах |
| `PAY_0` ... `PAY_6` | статус погашения за сентябрь, август, июль, июнь, май, апрель 2005 (-1 = вовремя, 1 = задержка 1 месяц и далее) |
| `BILL_AMT1` ... `BILL_AMT6` | сумма счёта за тот же период |
| `PAY_AMT1` ... `PAY_AMT6` | сумма платежа за тот же период |

Пример запроса (модель v1):

```bash
curl -X POST "http://localhost:5000/predict?version=v1" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'
```

Ответ:

```json
{"prediction": 1, "probability": 0.7552882972970761, "model_version": "v1"}
```

Тот же запрос на модель v2:

```bash
curl -X POST "http://localhost:5000/predict?version=v2" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'
```

Ответ:

```json
{"prediction": 1, "probability": 0.7561371364855446, "model_version": "v2"}
```

Описание полей ответа:
- `prediction` - 0 (дефолта не будет) или 1 (предсказан дефолт)
- `probability` - вероятность дефолта, число от 0 до 1
- `model_version` - использованная версия модели, `v1` или `v2`

Пример запроса с неполным набором признаков:

```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000}'
```

Ответ (HTTP 400):

```json
{"error": "Missing required fields: ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']"}
```

HTTP-код 400 возвращается на любые невалидные входные данные (не-JSON тело, неполный набор признаков, нечисловое значение, неизвестная версия модели).

## Метрики моделей

Обучение и оценка - на разделении 80/20 со стратификацией по target, `random_state=42`.

| Версия | n_estimators | max_depth | learning_rate | F1 (default) | ROC AUC |
|---|---|---|---|---|---|
| v1 | 100 | 3 | 0.10 | 0.47 | 0.7784 |
| v2 | 200 | 5 | 0.05 | 0.47 | 0.7766 |

Версии близки по качеству на офлайн-выборке. Разница в гиперпараметрах сделана сознательно для иллюстрации сценария A/B-теста, в котором новая версия не обязательно превосходит старую и решение принимается по результатам анализа на трафике.

## Архитектура (краткое описание)

Сервис построен как монолит: одно Flask-приложение в Docker-контейнере, обе версии модели загружаются в память при старте. Маршрутизация v1 и v2 выполняется внутри сервиса. Перед Flask-приложением стоит nginx как reverse proxy (порт 80, порт 5000 ml-service не публикуется), что соответствует production-схеме развёртывания.

Логи пишутся в stdout одной JSON-строкой на запрос с полями `timestamp`, `endpoint`, `method`, `status`, `model_version`, `prediction`, `probability` - формат пригоден для сборщиков Filebeat или Fluentd с дальнейшей отправкой в Elasticsearch и визуализацией в Kibana.

Полное обоснование выбора монолита, концепты RabbitMQ, DVC, MLflow, ONNX и uWSGI, а также формулы бизнес-метрик (Expected Loss, Approval Rate at Fixed Bad Rate) - в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## A/B-тестирование

Сервис поддерживает два режима выбора версии модели:

- **Ручной**: query-параметр `?version=v1` или `?version=v2`, либо поле `version` в теле JSON. Используется для тестирования и для маршрутизации в продуктовой среде, где решение о версии принимает upstream-сервис
- **Случайное 50/50**: включается константой `AB_ENABLED = True` в `app/model_handler.py`. По умолчанию выключено, штатное поведение детерминировано (все запросы идут на v1). В production переключатель выносится в переменную окружения

Полный план теста: цель, размер выборки (~2400 на группу), z-test для Recall, bootstrap-CI для F1, критерий успешности и связь с архитектурой - в [docs/ab_test_plan.md](docs/ab_test_plan.md).

## Пример логов работы сервиса

Сервис пишет в stdout одну JSON-строку на каждый запрос. Поля: `timestamp`, `level`, `endpoint`, `method`, `status`, для `/predict` дополнительно `model_version`, `prediction`, `probability`. Формат пригоден для сборщиков Filebeat или Fluentd с дальнейшей отправкой в Elasticsearch.

Фрагмент `docker compose logs ml-service` после прогона curl-запросов через nginx из разделов выше:

```json
{"timestamp": 1777116682.866883,  "level": "info", "endpoint": "/health",  "method": "GET",  "status": 200}
{"timestamp": 1777116682.9366639, "level": "info", "endpoint": "/predict", "method": "POST", "status": 200, "model_version": "v1", "prediction": 1, "probability": 0.7552882972970761}
{"timestamp": 1777116682.987094,  "level": "info", "endpoint": "/predict", "method": "POST", "status": 200, "model_version": "v2", "prediction": 1, "probability": 0.7561371364855446}
{"timestamp": 1777116683.0309496, "level": "info", "endpoint": "/predict", "method": "POST", "status": 400, "model_version": null, "prediction": null, "probability": null}
```

Первая строка - проверка `/health`, две следующие - `/predict` с версиями v1 и v2 (одинаковый вход, разные модели), последняя - 400 на запрос с неполным набором признаков.
