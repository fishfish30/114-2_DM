# 第 5 週作業：羅吉斯迴歸與分類評估

## 作業資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch6 羅吉斯迴歸 |
| 繳交方式 | 在 Fork 的 week05/ 資料夾中建立三個檔案，發 PR 繳交 |
| 繳交期限 | 下週上課前 |
| PR 標題格式 | 學號_姓名_week05 |

---

## 第 1 題：羅吉斯迴歸分類與混淆矩陣（40 分）

### 任務說明

使用 Scikit-learn 的鳶尾花資料集，訓練羅吉斯迴歸模型進行分類，並輸出完整的分類評估報告。

### Python 程式要求

撰寫程式碼完成以下工作：

1. 載入鳶尾花資料集（sklearn.datasets.load_iris）
2. 使用 pairplot 或散布圖探索資料分布
3. 切割資料（test_size=0.3, random_state=42）
4. 建立 Pipeline：StandardScaler → LogisticRegression(max_iter=200)
5. 訓練模型並進行預測
6. 印出混淆矩陣
7. 印出 Classification Report（包含 precision、recall、f1-score）
8. 印出訓練集與測試集的準確率

### 作答內容

請建立 `week05/q1_logistic.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 資料探索 ===
（描述鳶尾花資料集的基本資訊：幾筆資料、幾個特徵、幾個類別）

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 混淆矩陣 ===
（貼上混淆矩陣）

=== Classification Report ===
（貼上完整的分類報告）

=== 準確率 ===
訓練集準確率：???
測試集準確率：???
```

### 提示

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=200))
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 正確載入資料並完成探索 | 8 分 |
| Pipeline 建立正確且模型訓練成功 | 10 分 |
| 混淆矩陣輸出正確 | 10 分 |
| Classification Report 輸出完整 | 12 分 |

---

## 第 2 題：ROC 曲線與門檻調整（40 分）

### 任務說明

使用二元分類情境，繪製 ROC 曲線、計算 AUC，並觀察調整門檻值對精確率與召回率的影響。

### 測試資料

請使用以下程式碼建立二元分類資料：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

# 模擬船舶延遲預測資料
data = {
    'wind_speed': np.round(np.random.uniform(5, 40, n), 1),
    'wave_height': np.round(np.random.uniform(0.5, 5.0, n), 1),
    'cargo_weight': np.random.randint(10000, 80000, n),
    'port_congestion': np.round(np.random.uniform(0, 1, n), 2)
}
df = pd.DataFrame(data)

# 延遲與否（1=延遲, 0=準時）
prob = 1 / (1 + np.exp(-(0.05*df['wind_speed'] + 0.3*df['wave_height']
            + 0.00002*df['cargo_weight'] + 2*df['port_congestion'] - 3)))
df['is_delayed'] = (np.random.random(n) < prob).astype(int)

df.to_csv('ship_delay.csv', index=False)
print(f"資料形狀：{df.shape}")
print(f"延遲比例：\n{df['is_delayed'].value_counts(normalize=True)}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 讀取資料，切割訓練/測試集（test_size=0.3, random_state=42）
2. 建立 Pipeline：StandardScaler → LogisticRegression
3. 訓練模型，取得測試集的預測機率（predict_proba）
4. 使用 roc_curve 和 roc_auc_score 繪製 ROC 曲線，標示 AUC 值
5. 嘗試三個不同門檻值（0.3、0.5、0.7），分別計算精確率和召回率
6. 印出門檻值比較表

### 作答內容

請建立 `week05/q2_roc_threshold.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== AUC 分數 ===
AUC：???

=== 門檻值比較表 ===
門檻=0.3：精確率=???  召回率=???
門檻=0.5：精確率=???  召回率=???
門檻=0.7：精確率=???  召回率=???

=== 觀察 ===
（說明門檻值變化對精確率和召回率的影響趨勢）
```

### 提示

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score

y_prob = pipe.predict_proba(X_test)[:, 1]

# ROC 曲線
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# 不同門檻
for t in [0.3, 0.5, 0.7]:
    y_pred_t = (y_prob >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    print(f"門檻={t}：精確率={p:.4f}  召回率={r:.4f}")
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 模型訓練正確，取得預測機率 | 10 分 |
| ROC 曲線繪製正確，AUC 計算正確 | 10 分 |
| 三個門檻值的精確率/召回率計算正確 | 10 分 |
| 觀察門檻值影響的描述合理 | 10 分 |

---

## 第 3 題：分類評估觀念題（20 分）

### 作答內容

請建立 `week05/q3_concept.txt`，回答以下問題：

```
姓名：
學號：

Q1：在以下情境中，你會更重視精確率還是召回率？請說明理由。
    情境：一套系統用來偵測船舶是否有走私嫌疑，海關人員會根據系統的預測結果
    決定是否對船舶進行人工檢查。
A1：???

Q2：ROC 曲線的 X 軸和 Y 軸分別代表什麼？
    AUC = 0.5 代表什麼意義？AUC = 1.0 又代表什麼？
A2：???

Q3：羅吉斯迴歸和線性迴歸的輸出有什麼不同？
    為什麼羅吉斯迴歸適合用來做分類，而不是直接用線性迴歸做分類？
A3：???
```

### 評分標準

| 項目 | 配分 |
|------|------|
| Q1 正確判斷精確率/召回率的優先順序並說明理由 | 8 分 |
| Q2 正確解釋 ROC 曲線軸意義與 AUC 值 | 6 分 |
| Q3 正確說明兩種迴歸的差異與適用場景 | 6 分 |

---

## 繳交 Checklist

- [ ] week05/q1_logistic.txt 包含完整程式碼、混淆矩陣與 Classification Report
- [ ] week05/q2_roc_threshold.txt 包含完整程式碼、ROC 曲線、門檻值比較
- [ ] week05/q3_concept.txt 包含三題觀念回答
- [ ] 已 push 到自己的 Fork
- [ ] 已發 PR，標題格式：學號_姓名_week05
