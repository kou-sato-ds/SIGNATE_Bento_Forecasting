# ==========================================
# 今日の「草」：前処理ロジックの写経
# ==========================================
import pandas as pd

def preprocess(df):
    """お弁当プロジェクト：備考欄から特定メニューを抽出"""
    # 'お楽しみメニュー'が含まれる場合にフラグを立てる
    df['fun_menu'] = df['remarks'].apply(lambda x: 1 if x == 'お楽しみメニュー' else 0)
    
    # 欠損値処理（例：降水量がNaNなら0にする）
    df['precipitation'] = df['precipitation'].fillna(0)
    return df

print("Preprocess function created and pushed!")