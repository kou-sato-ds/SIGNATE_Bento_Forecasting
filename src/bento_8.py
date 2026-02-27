# ==========================================
# 1. import
# ==========================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# ==========================================
# 2. read
# ==========================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submit.csv', header=None)

# ==========================================
# 3. features (簡略化した特徴量作成)
# ==========================================
# お弁当プロジェクトを想定した「お楽しみメニュー」フラグ作成など
train['fun_menu'] = train['remarks'].apply(lambda x: 1 if x == 'お楽しみメニュー' else 0)
test['fun_menu'] = test['remarks'].apply(lambda x: 1 if x == 'お楽しみメニュー' else 0)

# 使用するカラムを選択
select_cols = ['week', 'temp', 'precipitation', 'fun_menu']
X = pd.get_dummies(train[select_cols])
y = train['y']
X_test = pd.get_dummies(test[select_cols])

# ==========================================
# 4. K-fold / Stratified split
# ==========================================
# ※回帰問題の場合は本来KFoldですが、あえて流れを重視します
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 5. submit
# ==========================================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
pred = model.predict(X_test)

sample[1] = pred
sample.to_csv('submit_obento.csv', index=None, header=None)

print("Processing Complete!")