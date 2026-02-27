# ==========================================
# データエンジニアの実装：前処理と評価の統合
# ==========================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# --- 昨日の成果：前処理ロジック ---
def preprocess(df):
    """お弁当プロジェクト：備考欄から特定メニューを抽出と欠損値処理"""
    # 'お楽しみメニュー'が含まれる場合にフラグを立てる（タイポ修正済み）
    df['fun_menu'] = df['remarks'].apply(lambda x: 1 if x == 'お楽しみメニュー' else 0)
    
    # 欠損値処理（降水量がNaNなら0にする）
    df['precipitation'] = df['precipitation'].fillna(0)
    return df

# --- 本日の成果：モデル評価ロジック ---
def evaluate_model(y_true, y_pred):
    """予測値と実績値の差（RMSE）を計算する"""
    # MSEを計算してから平方根(sqrt)をとってRMSEにする
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # 小数点2桁で表示（タイポ修正済み）
    print(f"RMSE: {rmse:.2f}")
    return rmse

# --- 実行イメージ ---
print("Preprocess and Evaluation logic ready!")