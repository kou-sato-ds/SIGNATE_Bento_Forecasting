# ===================================================
# Project: Bento Demand Forecasting (Advanced Hybrid Model)
# logic: Linear Trend + Random Forest Residuals
# ===================================================

# [1] Import
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# [2] Read
# dataフォルダの中にあるので、"data/ファイル名" と指定します
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# sample.csvもdataフォルダにある場合はここも修正
sample = pd.read_csv("data/sample.csv", header=None)

# [3] Features (Notebookの知見を反映)
# データの結合（前処理を一括で行うため）
train['t'] = 1
test['t'] = 0
df = pd.concat([train, test], sort=False).reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['datetime'])

# 分析に基づき、2014年5月以降に絞る（学習データのみ）
# ※testデータは消さないように注意
train_filtered = df[(df['t'] == 1) & (df['datetime'] >= '2014-05-01')]
test_data = df[df['t'] == 0]
df = pd.concat([train_filtered, test_data]).reset_index(drop=True)

# 特徴量作成
df['days'] = df.index # トレンド用
df['precipitation'] = df['precipitation'].apply(lambda x: -1 if x == '--' else float(x))
df['fun'] = df['remarks'].apply(lambda x: 1 if x == 'お楽しみメニュー' else 0)
df['curry'] = df['name'].apply(lambda x: 1 if 'カレー' in str(x) else 0)

# ダミー変数化 (weather)
df = pd.get_dummies(df, columns=['weather'])

# 使用する列の選定
cols = [c for c in df.columns if c not in ['datetime', 'y', 't', 'name', 'remarks', 'week', 'event', 'payday', 'soldout']]
X_all = df[cols]
y_all = df['y']

X_train_full = X_all[df['t'] == 1]
y_train_full = y_all[df['t'] == 1]
X_test = X_all[df['t'] == 0]

# [4] Split
kf = KFold(n_splits=5, shuffle=True, random_state=777)

# [5] Fit & [6] Predict (Hybrid Approach)
# トレンドをLinearRegressionで、残差をRandomForestで学習
val_scores = []
final_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train_full):
    X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    # Model 1: Linear Regression (Trend)
    model1 = LinearRegression()
    model1.fit(X_tr[['days']], y_tr)
    pred_tr_linear = model1.predict(X_tr[['days']])
    
    # Model 2: Random Forest (Residuals)
    # yの残差（実際の値 - 線形予測値）を学習
    residuals = y_tr - pred_tr_linear
    model2 = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=777)
    model2.fit(X_tr, residuals)

    # Validation
    pred_val = model1.predict(X_val[['days']]) + model2.predict(X_val)
    val_scores.append(np.sqrt(mean_squared_error(y_val, pred_val)))

    # Test Prediction
    final_preds += (model1.predict(X_test[['days']]) + model2.predict(X_test)) / 5

print(f"Average CV RMSE: {np.mean(val_scores):.4f}")

# [Submit]
sample[1] = final_preds
sample.to_csv("submission_hybrid.csv", index=False, header=False)
print("Hybrid submission file created!")