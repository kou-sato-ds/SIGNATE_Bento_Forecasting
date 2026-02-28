import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging

# --- 1. Logging Setup (実務の作法) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BentoForecaster:
    def __init__(self):
        # 実務で標準的なRandomForestを採用
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    # --- 2. Read & Preprocessing ---
    def prepare_data(self, df):
        # 時系列特徴量の抽出
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        
        # 欠損値処理
        df['remarks'] = df['remarks'].fillna('None')
        # ドメイン知識：お楽しみメニューのフラグ化
        df['fun_flag'] = df['remarks'].apply(lambda x: 1 if 'お楽しみ' in x else 0)
        
        # 不要な列の削除とダミー変数化（DS検定頻出：One-Hot Encoding）
        df = pd.get_dummies(df.drop(['datetime', 'remarks'], axis=1))
        return df

    # --- 3. Features & Align (聖域の1行) ---
    def align_features(self, train, test):
        # 訓練とテストの列（次元）を一致させ、欠落を0で埋める
        train_aligned, test_aligned = train.align(test, join='left', axis=1, fill_value=0)
        return train_aligned, test_aligned

    # --- 4. K-fold Cross Validation ---
    def train_and_evaluate(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(