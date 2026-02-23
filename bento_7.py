import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import KFold
import lightgbm as lgb

# --- Logging設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BentoForecaster:
    def __init__(self):
        """モデルの初期化"""
        self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42)

    def load_data(self, train_path, test_path):
        """[Read] データの読み込み"""
        logger.info(f"Loading data: {train_path}, {test_path}")
        # ファイルが存在するか確認
        if not os.path.exists(train_path):
            logger.error(f"File not found: {train_path}")
            return None, None
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test

    def create_features(self, df):
        """[Features] 日付から特徴量を抽出"""
        logger.info("Creating features (month, day, weekday)...")
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.weekday
        return df.drop(['datetime'], axis=1, errors='ignore')

    def train(self, X, y, X_test):
        """[K-fold] 聖域の X.align を実行して学習"""
        logger.info("Aligning data and starting training...")
        # 訓練データとテストデータの列を一致させる
        X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
        
        self.model.fit(X, y)
        logger.info("Model training completed successfully.")
        return X_test

# --- Execution (実行ブロック) ---
if __name__ == "__main__":
    # TODO: 自分の環境にあるCSVのパスに合わせて書き換えてください
    # 例: 'train.csv', 'test.csv' など
    TRAIN_PATH = 'train.csv' 
    TEST_PATH = 'test.csv'
    
    forecaster = BentoForecaster()
    
    logger.info("🚀 Starting Bento Forecasting Pipeline (bento_7)...")
    
    # テスト実行用のロジック（ファイルがある場合のみ動く）
    if os.path.exists(TRAIN_PATH):
        train_df, test_df = forecaster.load_data(TRAIN_PATH, TEST_PATH)
        # ここに続きの工程を書いていく...
    else:
        logger.warning(f"CSV file not found at {TRAIN_PATH}. Skipping actual data processing.")
    
    logger.info("✅ Pipeline structure check complete.")