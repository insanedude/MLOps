import pandas as pd
import yaml
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- НАЛАШТУВАННЯ ШЛЯХІВ (як у data_prep.py) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_path = os.path.join(project_root, "configs", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 1. Завантаження даних
train_path = os.path.join(project_root, config['data']['train_path'])
train_df = pd.read_csv(train_path)

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']

# 2. Ініціалізація моделі
print(f"Training model: {config['model']['type']}...")

if config['model']['type'] == 'LogisticRegression':
    model = LogisticRegression(**config['model']['params'])
elif config['model']['type'] == 'RandomForestClassifier':
    model = RandomForestClassifier(**config['model']['params'])
else:
    raise ValueError(f"Unsupported model type: {config['model']['type']}")

# 3. Навчання
model.fit(X_train, y_train)

# 4. Збереження моделі
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "model.joblib")

joblib.dump(model, model_path)
print(f"Model saved to {model_path}")