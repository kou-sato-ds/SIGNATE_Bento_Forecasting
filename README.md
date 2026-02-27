# SIGNATE Bento Demand Forecasting (Master Model Implementation)

2026年11月の社内外の転身を見据えた、お弁当需要予測プロジェクトです。
単なるスコアアップを目的とせず、実務に耐えうる**「堅牢な機械学習パイプライン」**と**「保守性の高いコード設計」**の構築を主眼に置いています。

## 📌 プロジェクトの背景
私はIT企業の事務職として働く傍ら、2025年8月の「マナビDXクエスト」をきっかけにAIエンジニアへの道を決意しました。
12月から開始した連続学習の中で、かつて経験した「スコア0.5（ランダム予測）」というサイレント・バグの絶望を乗り越え、現在は**「データの整合性と再現性を保証する実装」**を信条としています。

## 📂 ディレクトリ構成 (Directory Structure)
```text
.
├── src/                # 実装コード (bento_3.py ~ bento_10.py)
│   └── main.py         # 最終的な予測パイプライン
├── data/               # (Git管理対象外) 学習用・予測用CSVデータ
├── docs/               # プロジェクト資料・実行結果
│   └── images/         # スコア証明および可視化グラフ
├── notebooks/          # 探索的データ分析 (EDA)
├── .gitignore          # セキュリティ設定 (Data/Terraform関連)
└── README.md           # 本ドキュメント

```

## 🏆 達成実績

* **平均交差検証誤差 (Average CV RMSE)**: **11.2777**
* **SIGNATE 暫定スコア**: **9.6301** (RMSE)

*図：ハイブリッドモデルによるRMSE 11.27の達成とSIGNATEへの投稿完了証明*

## 💡 実装のこだわり（Master型設計）

### 1. 黄金のフローによるカプセル化

`BentoForecaster`クラスを定義し、`import -> read -> features -> K-fold -> submit` の流れをメソッド化。誰がどこから動かしても同じ結果が得られる「職人芸に頼らない」設計です。

### 2. 聖域の1行：`X.align` による次元保証

訓練データとテストデータの列のズレを完封するため、学習直前に必ず `X.align` を実行。カテゴリ変数の数値化過程で生じる予期せぬ次元不一致を自動で修正し、モデルの堅牢性を確保しています。

### 3. ハイブリッド予測戦略

* **Linear Regression**: 長期的な売上の減少トレンドをキャプチャ。
* **Random Forest / LightGBM**: 天候や「お楽しみメニュー」などの非線形な要因（残差）を学習。
* **データフィルタリング**: トレンドが変化した2014年5月以降のデータに特化して学習。

## 🛠 使用技術

* **Language**: Python 3.10+
* **Library**: Pandas, Scikit-learn, LightGBM
* **Validation**: K-Fold Cross Validation (5-folds)
* **Infrastructure**: Terraform (AWS S3 Data Lake)

## 🔄 モデル改善の試行錯誤 (A/B Testing)

| モデル名 | 手法・特徴量 | スコア (RMSE) | 考察 |
| --- | --- | --- | --- |
| **bento_3 (Best)** | **Linear Trend + RF 残差学習** | **11.27** | **トレンドと日次変動を分けたのが正解。** |
| bento_5 | GBDT + メニュー名特徴量 | 14.20 | データ件数が少ないため過学習が発生。 |

---

### 🔗 関連リンク

* [Qiita: モデルを疑う前に「データの形」を疑え。Kaggleスコア0.5の絶望を救った「聖域の1行」](https://qiita.com/yoshirin1989k/items/589fb64ffd970c88faea)
* [GitHub: AWS_IaC_Terraform](https://github.com/kou-sato-ds/AWS_IaC_Terraform)