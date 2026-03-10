# 第 6 週作業：KNN 與支持向量機

## 作業資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch7 K 最近鄰、Ch8 支持向量機 |
| 繳交方式 | 在 Fork 的 week06/ 資料夾中建立三個檔案，發 PR 繳交 |
| 繳交期限 | 下週上課前 |
| PR 標題格式 | 學號_姓名_week06 |

---

## 第 1 題：KNN 分類與 K 值選擇（40 分）

### 任務說明

使用 KNN 對 Scikit-learn 內建的 wine 資料集進行分類，觀察不同 K 值與標準化對準確率的影響。

### Python 程式要求

撰寫程式碼完成以下工作：

1. 載入 wine 資料集（sklearn.datasets.load_wine）
2. 印出資料的基本資訊：幾筆資料、幾個特徵、幾個類別
3. 切割資料（test_size=0.3, random_state=42）
4. **未標準化**：使用 KNN（k=1 到 k=25，間隔 2），記錄每個 K 值的測試集準確率
5. **有標準化**：建立 Pipeline（StandardScaler → KNN），同樣測試 k=1 到 k=25
6. 繪製折線圖比較：X 軸為 K 值，Y 軸為準確率，兩條線分別代表有/無標準化
7. 找出標準化後最佳的 K 值與對應準確率

### 作答內容

請建立 `week06/q1_knn.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 資料基本資訊 ===
資料筆數：???
特徵數量：???
類別數量：???

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== K 值準確率比較（標準化 vs 未標準化）===
（貼上部分代表性的 K 值與準確率，或貼上完整表格）

=== 最佳結果 ===
標準化後最佳 K 值：???
對應測試集準確率：???

=== 折線圖觀察 ===
（描述圖表呈現的趨勢，標準化前後差異有多大）
```

### 提示

```python
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

wine = load_wine()
X, y = wine.data, wine.target

k_range = range(1, 26, 2)
scores_raw = []
scores_scaled = []

for k in k_range:
    # 未標準化
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores_raw.append(knn.score(X_test, y_test))

    # 有標準化
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipe.fit(X_train, y_train)
    scores_scaled.append(pipe.score(X_test, y_test))
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 正確載入資料並印出基本資訊 | 5 分 |
| 未標準化的 KNN 多 K 值測試正確 | 10 分 |
| 標準化 Pipeline 的多 K 值測試正確 | 10 分 |
| 繪製比較折線圖 | 10 分 |
| 找出最佳 K 值，觀察描述合理 | 5 分 |

---

## 第 2 題：SVM 與 ColumnTransformer 實作（40 分）

### 任務說明

使用鐵達尼號資料集，建立包含數值與類別欄位處理的完整管道器，搭配 SVM 進行生存預測。

### 測試資料

請使用以下程式碼建立簡化版鐵達尼號資料：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300

data = {
    'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    'sex': np.random.choice(['male', 'female'], n, p=[0.6, 0.4]),
    'age': np.round(np.random.uniform(1, 75, n), 0),
    'fare': np.round(np.random.exponential(30, n), 2),
    'embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.7, 0.2, 0.1]),
    'sibsp': np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05]),
}
df = pd.DataFrame(data)

# 加入遺漏值
df.loc[np.random.choice(n, 15, replace=False), 'age'] = np.nan
df.loc[np.random.choice(n, 8, replace=False), 'embarked'] = np.nan

# 生存機率受多個因素影響
prob = 1 / (1 + np.exp(-(
    -0.5 * df['pclass'] +
    1.5 * (df['sex'] == 'female').astype(int) -
    0.01 * df['age'].fillna(30) +
    0.005 * df['fare'] - 0.5
)))
df['survived'] = (np.random.random(n) < prob).astype(int)

df.to_csv('titanic_simple.csv', index=False)
print(f"資料形狀：{df.shape}")
print(f"\n遺漏值：\n{df.isnull().sum()}")
print(f"\n生存比例：\n{df['survived'].value_counts(normalize=True)}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 讀取資料，區分數值型與類別型欄位
2. 建立數值管道：SimpleImputer(mean) → StandardScaler
3. 建立類別管道：SimpleImputer(most_frequent) → OneHotEncoder(sparse_output=False)
4. 使用 ColumnTransformer 合併兩條管道
5. 建立完整 Pipeline：ColumnTransformer → SVC(kernel='rbf')
6. 切割資料（test_size=0.3, random_state=42），訓練模型
7. 印出測試集的準確率與 Classification Report
8. 額外建立一個使用 LogisticRegression 的 Pipeline，比較兩個模型的準確率

### 作答內容

請建立 `week06/q2_svm.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 欄位分類 ===
數值型欄位：???
類別型欄位：???
目標欄位：???

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== SVM 結果 ===
測試集準確率：???
（貼上 Classification Report）

=== LogisticRegression 結果 ===
測試集準確率：???

=== 比較 ===
（哪個模型表現較好？差距大嗎？）
```

### 提示

```python
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

num_cols = ['age', 'fare', 'sibsp', 'pclass']
cat_cols = ['sex', 'embarked']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='mean')), ('scl', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(sparse_output=False))]), cat_cols)
])

pipe_svm = Pipeline([('pre', preprocessor), ('clf', SVC(kernel='rbf'))])
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 正確區分欄位並建立數值/類別管道 | 10 分 |
| ColumnTransformer 與完整 Pipeline 正確 | 10 分 |
| SVM 模型訓練且有 Classification Report | 10 分 |
| LogisticRegression 比較且有觀察 | 10 分 |

---

## 第 3 題：KNN 與 SVM 觀念題（20 分）

### 作答內容

請建立 `week06/q3_concept.txt`，回答以下問題：

```
姓名：
學號：

Q1：為什麼 KNN 和 SVM 在使用前需要對資料做標準化，
    而決策樹不需要？請從演算法的運作原理來解釋。
A1：???

Q2：KNN 的 K 值如果設太小（例如 K=1）或太大（例如 K=100），
    各自會造成什麼問題？這跟過擬合/欠擬合有什麼關係？
A2：???

Q3：SVM 的 kernel 參數可以選擇 'linear' 或 'rbf'。
    請用簡單的方式說明這兩種 kernel 的差異，
    以及什麼時候適合用 linear、什麼時候適合用 rbf。
A3：???
```

### 評分標準

| 項目 | 配分 |
|------|------|
| Q1 從演算法原理解釋標準化的必要性 | 7 分 |
| Q2 正確說明 K 值過大/過小的問題 | 7 分 |
| Q3 合理說明兩種 kernel 的差異與適用情境 | 6 分 |

---

## 繳交 Checklist

- [ ] week06/q1_knn.txt 包含完整程式碼、K 值比較圖與最佳 K 值
- [ ] week06/q2_svm.txt 包含完整程式碼、SVM 與 LogisticRegression 比較
- [ ] week06/q3_concept.txt 包含三題觀念回答
- [ ] 已 push 到自己的 Fork
- [ ] 已發 PR，標題格式：學號_姓名_week06
