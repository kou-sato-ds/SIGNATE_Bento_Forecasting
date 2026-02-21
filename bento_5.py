# ===================================================
# Strategy B: Feature Engineering + GBDT
# Goal: メニュー内容の数値化と最新アルゴリズムの適用
# ===================================================

# [1] Import
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# [2] Read
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample.csv", header=None)
train = train[pd.to_datetime(train['datetime']) >= '2014-05-01'].reset_index(drop=True)

# [3] Features
def eng_features(df):
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    # 特定のキーワードが含まれるか
    df['curry'] = df['name'].apply(lambda x: 1 if 'カレー' in str(x) else 0)
    df['pork'] = df['name'].apply(lambda x: 1 if '豚' in str(x) else 0)
    df['is_fun'] = df['remarks'].apply(lambda x: 1 if x == 'お楽しみメニュー' else 0)
    # 曜日のダミー変数化
    df = pd.concat([df, pd.get_dummies(df['week'])], axis=1)
    return df

train = eng_features(train)
test = eng_features(test)

# 特徴量の整理
features = ['month', 'kcal', 'temperature', 'curry', 'pork', 'is_fun', '月', '火', '水', '木', '金']
X = train[features].fillna(-1)
y = train['y']
X_test = test[features].fillna(-1)

# [4] Split & [5] Fit
# 勾配ブースティング回帰
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_gb.fit(X, y)

# [6] Predict
final_preds = model_gb.predict(X_test)

# [Submit]
sample[1] = final_preds
sample.to_csv("submission_strategy_B.csv", index=False, header=False)
print("bento_5.py: GBDT process completed!")