# Breast Cancer Classification MLOps Project

Проєкт з класифікації раку грудей (WIDC Dataset) з використанням MLOps практик.

## Структура
- `configs/`: Конфігураційні файли (YAML).
- `data/`: Сирі та оброблені дані.
- `src/`: Скрипти для підготовки, навчання та оцінки.
- `models/`: Збережені моделі.
- `reports/`: Звіти, графіки та логи.

## Відтворення (Reproducibility)
1. **Встановлення залежностей:**
pip install -r requirements.txt
2. Підготовка даних:
python src/data_prep.py
3. Запуск Baseline (Logistic Regression):
python src/train.py
python src/evaluate.py
4. Запуск AutoML (H2O):
python src/automl_run.py
5. Перегляд результатів:
mlflow ui