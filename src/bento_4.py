# ===================================================
# Strategy A: Ridge Trend + RF Residuals
# Goal: トレンド抽出の安定化とハイブリッド予測
# ===================================================

# [1] Import
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# [2] Read
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample.csv", header=None)

# [3] Features
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# 2014年5月以降に絞り込み
train = train[train['datetime'] >= '2014-05-01'].reset_index(drop=True)

def extract_features(df):
    # トレンド用の経過日数
    df['days'] = (df['datetime'] - pd.to_datetime('2013-11-18')).dt.days
    df['is_fun'] = df['remarks'].apply(lambda x: 1 if x == 'お楽しみメニュー' else 0)
    df['is_payday'] = df['payday'].fillna(0)
    return df

train = extract_features(train)
test = extract_features(test)

cols = ['days', 'kcal', 'temperature', 'is_fun', 'is_payday']
X = train[cols].fillna(-1)
y = train['y']
X_test = test[cols].fillna(-1)

# [4] Split (Hold-out for speed, or K-Fold)
# 今回は実装力を試すため、シンプルなFit & Predictの流れ
# [5] Fit
# トレンド学習
model_trend = Ridge(alpha=1.0)
model_trend.fit(X[['days']], y)
res = y - model_trend.predict(X[['days']])

# 残差学習
model_rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model_rf.fit(X, res)

# [6] Predict
pred_trend = model_trend.predict(X_test[['days']])
pred_res = model_rf.predict(X_test)
final_preds = pred_trend + pred_res

# [Submit]
sample[1] = final_preds
sample.to_csv("submission_strategy_A.csv", index=False, header=False)
print("bento_4.py: Hybrid process completed!")