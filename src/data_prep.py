import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

# --- ВИПРАВЛЕННЯ ШЛЯХІВ ---
# Отримуємо абсолютний шлях до папки, де лежить цей скрипт (src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Піднімаємось на рівень вище, щоб отримати корінь проєкту (practice8/)
project_root = os.path.dirname(script_dir)
# Формуємо правильний шлях до конфігу
config_path = os.path.join(project_root, "configs", "config.yaml")
# --------------------------

# 1. Завантаження конфігурації
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"❌ Error: Config file not found at {config_path}")
    print("Please ensure you created 'configs/config.yaml' in the project root.")
    exit(1)

# Формуємо шляхи до даних, використовуючи project_root
raw_data_path = os.path.join(project_root, config['data']['raw_path'])
train_path = os.path.join(project_root, config['data']['train_path'])
test_path = os.path.join(project_root, config['data']['test_path'])

# 2. Завантаження "сирих" даних
print(f"Loading data from {raw_data_path}...")

# Формуємо імена колонок
col_names = [config['data']['id_col'], config['data']['target_col']] + [f"feat_{i}" for i in range(1, 31)]

try:
    df = pd.read_csv(raw_data_path, header=None, names=col_names)
except FileNotFoundError:
    print(f"❌ Error: Data file not found at {raw_data_path}")
    print("Please save the dataset content into 'data/raw/wdbc.data'")
    exit(1)

# 3. Попередня обробка (Preprocessing)
# Кодуємо цільову змінну: M (Malignant) -> 1, B (Benign) -> 0
df['target'] = df[config['data']['target_col']].map({'M': 1, 'B': 0})

# Вибираємо ознаки (X) та ціль (y)
X = df.drop(columns=[config['data']['id_col'], config['data']['target_col'], 'target'])
y = df['target']

# 4. Розділення даних
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['data']['split']['test_size'],
    random_state=config['data']['split']['random_state'],
    stratify=y if config['data']['split']['stratify'] else None
)

# 5. Збереження
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Створюємо папку processed, якщо її немає (використовуючи абсолютний шлях)
os.makedirs(os.path.dirname(train_path), exist_ok=True)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Data saved to processed folder")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")