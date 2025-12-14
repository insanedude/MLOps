import pandas as pd
import yaml
import os
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# --- НАЛАШТУВАННЯ ШЛЯХІВ ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_path = os.path.join(project_root, "configs", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 1. Завантаження даних та моделі
test_path = os.path.join(project_root, config['data']['test_path'])
test_df = pd.read_csv(test_path)
X_test = test_df.drop(columns=['target'])
y_test = test_df['target']

model_path = os.path.join(project_root, "models", "model.joblib")
model = joblib.load(model_path)

# 2. Прогнозування
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Ймовірності для ROC-кривої

# 3. Розрахунок метрик
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_macro": f1_score(y_test, y_pred, average='macro')
}

print("Metrics:", metrics)

# 4. Генерація графіків (Artifacts)
reports_dir = os.path.join(project_root, "reports")
os.makedirs(reports_dir, exist_ok=True)

# Матриця плутанини (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
cm_path = os.path.join(reports_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# ROC-крива
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
roc_path = os.path.join(reports_dir, "roc_curve.png")
plt.savefig(roc_path)
plt.close()

# 5. Логування в MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

with mlflow.start_run():
    # Логуємо параметри (з конфігу)
    mlflow.log_params(config['model']['params'])
    mlflow.log_param("model_type", config['model']['type'])

    # Логуємо метрики
    mlflow.log_metrics(metrics)

    # Логуємо артефакти (графіки та модель)
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(model_path)

    print(f"Experiment logged to MLflow. Run ID: {mlflow.active_run().info.run_id}")