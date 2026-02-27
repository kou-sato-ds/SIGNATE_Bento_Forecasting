import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import KFold
import lightgbm as lgb

import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import KFold
import lightgbm as lgb


# --- Logging (実務で必須の可視化) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logging.basicConfig(level=loggiing.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BentoForecaster:
    def __init__(self):

class BentoForecaster:
    def __init__(self):

        """[DEA想定] モデルの初期化とハイパーパラメータの設定"""
        self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42)

        self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42)

    def load_data(self, train_path, test_path):
        """[Read] データの読み込み"""
        logger.info(f"Checking files: {train_path}, {test_path}")
        if not os.path.exists(train_path):
            return None, None
        return pd.read_csv(train_path), pd.read_csv(test_path)

    def load_data(self, train_path, test_path):
        logger.info(f"Checking files: {train_path}, {test_path}")
        if not os.path.exists(train_path):
            return None, None
        return pd.read_csv(train_path), pd.read_csv(test_path)

    def create_features(self, df):
        """[Features] 特徴量生成"""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        return df.drop(['datetime'], axis=1, errors='ignore')

    def create_features(self, df):

        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        return df.drop(['datetime'], axis=1, error='ignore')

    def split_and_train(self, train_df):
        """[K-fold / Stratify] 学習実行（今回は簡略版）"""
        target = 'y'
        if target not in train_df.columns:
            logger.error("Target column 'y' not found.")
            return
        
        X = train_df.drop([target], axis=1)
        y = train_df[target]
        
        self.model.fit(X, y)
        logger.info("Training complete with LightGBM.")

    def split_and_train(self, train_df):
        
        target = 'y'
        if target not in train_df.columns:
            logger.error("Target column 'y" not found.")
            return

        X = train_df.drop([target], axis=1)
        y = train_df[target]

        self.model.fit(X, y)
        logger.info("Training complete with LightGBM.")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("🚀 Starting Bento Project for DEA Preparation...")
    
    # 実行ダミーデータ（ファイルがない場合でも構造を確認するため）
    forecaster = BentoForecaster()
    
    # ここでファイルを指定
    TRAIN_FILE = 'train.csv'
    if os.path.exists(TRAIN_FILE):
        train, test = forecaster.load_data(TRAIN_FILE, 'test.csv')
        train_feat = forecaster.create_features(train)
        forecaster.split_and_train(train_feat)
    
    logger.info("✅ Pipeline Structure Check Finished.")

if __name__ =="__main__":
    logger.info("Starting Bento Project for DEA Preparation...")

    forecaster = BentoForecaster()

    TRAIN_FILE = 'train.csv'
    if os.path.exists(TRAIN_FILE):
        train, test = forecaster.load_data(TRAIN_FILE, 'test.csv')
        train_feat = forecaster.create_features(train)
        forecaster.split_and_train(train_feat)

    logger.info("Pipeline Structure Check Finished.")