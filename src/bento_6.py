import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import KFold
import lightgbm as lgb

# --- [Master型] Logging設定 ---
# 実行時の状況を記録する「レコーダー」の準備
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BentoForecaster:
    def __init__(self):
        """ロボットの初期設定：持ち物リスト"""
        self.model = None
        self.features = []

    def load_data(self, train_path, test_path):
        """[Read] データの読み込み：os.path.joinを使うのがプロの作法"""
        logger.info("Loading data from CSV...")
        # 明日の復習：ここで pd.read_csv(train_path) を書く
        pass

    def create_features(self, df):
        """[Features] 特徴量エンジニアリング：AIに『ヒント』を与える工程"""
        logger.info("Creating features (Date, Weather, Menu)...")
        df = df.copy()
        # 明日の復習：ここで日付処理（dt.monthなど）を書く
        return df

    def train(self, X, y):
        """[K-fold] 交差検証と学習：『聖域』X.alignの出番"""
        logger.info("Starting K-fold validation and training...")
        # 明日の復習：ここで StratifiedKFold や X.align を書く
        pass

    def generate_submission(self, test_df):
        """[Submit] 予測とCSV保存：最後にしっかり書き出す"""
        logger.info("Generating final submission file...")
        # 明日の復習：ここで model.predict と to_csv を書く
        pass

# --- 実行スイッチ ---
if __name__ == "__main__":
    logger.info("🚀 Bento Demand Forecasting Pipeline Started.")
    
    # ここに「黄金の5工程」を並べて、ロボットを動かす
    # forecaster = BentoForecaster()
    # forecaster.load_data(...)