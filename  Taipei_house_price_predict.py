import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 顯示資料基本資訊
print("資料前幾筆：")
print(df.head())
print("\n欄位名稱：", df.columns.tolist())

# 假設我們選擇部分欄位作為特徵
# 以下是範例，實際需要依據你的資料欄位名稱做調整
# 假設資料有 'area', 'room', 'age', 'price' 等欄位
features = ['行政區', '土地面積', '屋齡']  # 這些是自變數（特徵）
target = '總價'  # 這是因變數（房價）

# 移除有缺失值的資料列
df = df.dropna(subset=features + [target])

# 分割訓練與測試資料
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 模型評估
mse = mean_squared_error(y_test, y_pred)
print(f"\n測試資料的均方誤差 (MSE): {mse:.2f}")

# 顯示實際與預測的比較圖
plt.scatter(y_test, y_pred)
plt.xlabel("實際房價")
plt.ylabel("預測房價")
plt.title("實際 vs 預測房價")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid()
plt.show()