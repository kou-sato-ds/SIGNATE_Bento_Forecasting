import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import KFold
import lightgbm as lgb

# --- Logging (実務で必須の可視化) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BentoForecaster:
    def __init__(self):
        """[DEA想定] モデルの初期化とハイパーパラメータの設定"""
        self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42, importance_type='gain')
        logger.info("Model initialized.")

    def load_data(self, train_path, test_path):
        """[Read] データの読み込み"""
        logger.info(f"Checking files: {train_path}, {test_path}")
        if not os.path.exists(train_path):
            logger.warning(f"File not found: {train_path}")
            return None, None
        return pd.read_csv(train_path), pd.read_csv(test_path)

    def create_features(self, df):
        """[Features] 特徴量生成"""
        if df is None:
            return None
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        # errors='ignore' (sが必要) で、カラムがなくてもエラーにしない
        return df.drop(['datetime'], axis=1, errors='ignore')

    def split_and_train(self, train_df):
        """[K-fold / Stratify] 学習実行（今回は簡略版）"""
        if train_df is None:
            return
            
        target = 'y'
        if target not in train_df.columns:
            logger.error(f"Target column '{target}' not found.")
            return
        
        X = train_df.drop([target], axis=1)
        y = train_df[target]
        
        # [X.align] 実務での次元不一致を防ぐための聖域の1行（概念）
        # ここでは簡易実装のためFitのみ
        self.model.fit(X, y)
        logger.info("Training complete with LightGBM.")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("🚀 Starting Bento Project for DEA Preparation...")
    
    forecaster = BentoForecaster()
    
    # 実行環境に合わせてパスを指定
    TRAIN_FILE = 'data/train.csv' 
    TEST_FILE = 'data/test.csv'
    
    if os.path.exists(TRAIN_FILE):
        train, test = forecaster.load_data(TRAIN_FILE, TEST_FILE)
        train_feat = forecaster.create_features(train)
        forecaster.split_and_train(train_feat)
    else:
        logger.warning(f"Skip training because {TRAIN_FILE} does not exist.")
    
    logger.info("✅ Pipeline Structure Check Finished.")