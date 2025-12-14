import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import yaml
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- ШЛЯХИ ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_path = os.path.join(project_root, "configs", "config_automl.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 1. Ініціалізація H2O
print("Initializing H2O Cluster...")
h2o.init()

# 2. Завантаження даних у H2O Frame
train_path = os.path.join(project_root, config['data']['train_path'])
test_path = os.path.join(project_root, config['data']['test_path'])

hf_train = h2o.import_file(train_path)
hf_test = h2o.import_file(test_path)

# ВАЖЛИВО: Для класифікації цільова змінна має бути фактором (categorical)
target = config['data']['target_col']
hf_train[target] = hf_train[target].asfactor()
hf_test[target] = hf_test[target].asfactor()

# 3. Запуск AutoML
print(f"Starting AutoML run for {config['automl']['max_runtime_secs']} seconds...")

aml = H2OAutoML(
    max_models=config['automl']['max_models'],
    max_runtime_secs=config['automl']['max_runtime_secs'],
    seed=config['automl']['seed'],
    sort_metric=config['automl']['sort_metric'],
    balance_classes=True  # Корисно при дисбалансі
)

# Навчаємо (x - всі колонки, крім target)
x_cols = hf_train.columns
x_cols.remove(target)

aml.train(x=x_cols, y=target, training_frame=hf_train)

# 4. Отримання результатів (Leaderboard)
lb = aml.leaderboard
print("AutoML Leaderboard:")
print(lb.head(rows=5))

# Зберігаємо лідерборд як CSV (Artifact)
reports_dir = os.path.join(project_root, "reports")
os.makedirs(reports_dir, exist_ok=True)
lb_path = os.path.join(reports_dir, "automl_leaderboard.csv")
h2o.export_file(lb, path=lb_path, force=True)

# 5. Оцінка найкращої моделі на тестових даних
# Отримуємо "Лідера"
best_model = aml.leader
print(f"Best Model ID: {best_model.model_id}")

# Передбачення
preds = best_model.predict(hf_test)
# Конвертуємо в pandas для використання sklearn метрик (для чесного порівняння)
preds_df = preds.as_data_frame()
y_pred = preds_df['predict'].values  # 0 або 1
y_test_true = hf_test[target].as_data_frame().values.ravel()  # Справжні значення

# 6. Розрахунок метрик (sklearn)
# Зверніть увагу: H2O може повертати '0'/'1' як строчки, треба привести до int
y_pred = y_pred.astype(int)
y_test_true = y_test_true.astype(int)

metrics = {
    "accuracy": accuracy_score(y_test_true, y_pred),
    "precision": precision_score(y_test_true, y_pred, zero_division=0),
    "recall": recall_score(y_test_true, y_pred, zero_division=0),
    "f1_macro": f1_score(y_test_true, y_pred, average='macro')
}
print("AutoML Test Metrics:", metrics)

# 7. Матриця плутанини
cm = confusion_matrix(y_test_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f"AutoML Confusion Matrix ({best_model.algo})")
cm_path = os.path.join(reports_dir, "confusion_matrix_automl.png")
plt.savefig(cm_path)
plt.close()

# 8. Логування в MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

with mlflow.start_run(run_name="AutoML_H2O"):
    # Логуємо параметри
    mlflow.log_params(config['automl'])
    mlflow.log_param("best_model_algo", best_model.algo)
    mlflow.log_param("best_model_id", best_model.model_id)

    # Логуємо метрики
    mlflow.log_metrics(metrics)

    # Логуємо артефакти
    mlflow.log_artifact(lb_path)
    mlflow.log_artifact(cm_path)

    # Можна спробувати зберегти саму модель (H2O binary), але це опціонально
    model_save_path = os.path.join(project_root, "models", "h2o_best_model")
    h2o.save_model(model=best_model, path=model_save_path, force=True)
    mlflow.log_artifact(model_save_path)

    print(f"AutoML results logged. Run ID: {mlflow.active_run().info.run_id}")