# ==========================================
# Rainfall Prediction using Machine Learning
# Complete Pipeline Script
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. Load Dataset
# ==========================================

data_path = Path("data/rainfall.csv")
if not data_path.exists():
    data_path = Path("rainfall.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "Could not find rainfall dataset. Please place rainfall.csv in the project root or data/ folder."
    )

with data_path.open() as f:
    header_row = None
    for idx, line in enumerate(f):
        if line.startswith("YEAR,DOY"):
            header_row = idx
            break

if header_row is None:
    raise ValueError(
        f"Could not find the header row in {data_path}. Ensure the file begins with the CSV header line 'YEAR,DOY,...'."
    )

# Skip metadata lines before the CSV header row and read the actual dataset.
df = pd.read_csv(data_path, skiprows=header_row)
print("Dataset shape:", df.shape)

# ==========================================
# 2. Data Cleaning
# ==========================================

df.replace(-999, np.nan, inplace=True)

df = df.interpolate(method="linear")
df = df.ffill().bfill()

df = df[df["PRECTOTCORR"] >= 0]

# ==========================================
# 3. Feature Engineering
# ==========================================

df["DATE"] = pd.to_datetime(df["YEAR"].astype(str)) + pd.to_timedelta(df["DOY"] - 1, unit="D")

df["MONTH"] = df["DATE"].dt.month
df["DAY"] = df["DATE"].dt.day

# Lag Features
df["RAIN_LAG1"] = df["PRECTOTCORR"].shift(1)
df["RAIN_LAG2"] = df["PRECTOTCORR"].shift(2)
df["RAIN_LAG3"] = df["PRECTOTCORR"].shift(3)

# Change Feature
df["RAIN_CHANGE"] = df["PRECTOTCORR"] - df["RAIN_LAG1"]

# Weather Interaction
df["TEMP_DEW_DIFF"] = df["T2M"] - df["T2MDEW"]

# Rolling Features
df["RAIN_ROLLING3"] = df["PRECTOTCORR"].rolling(window=3).mean()
df["RAIN_ROLLING7"] = df["PRECTOTCORR"].rolling(window=7).mean()

df["RAIN_INTENSITY"] = df["PRECTOTCORR"] / (df["RAIN_ROLLING7"] + 1)

df = df.dropna().reset_index(drop=True)

# ==========================================
# 4. Save Cleaned Dataset
# ==========================================

df.to_csv("cleaned_rainfall_dataset.csv", index=False)
print("Cleaned dataset saved.")

# ==========================================
# 5. Exploratory Data Analysis
# ==========================================

plt.figure(figsize=(6,4))
sns.histplot(df["PRECTOTCORR"], bins=50)
plt.title("Rainfall Distribution")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(df["DATE"], df["PRECTOTCORR"])
plt.title("Rainfall Over Time")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x=df["MONTH"], y=df["PRECTOTCORR"])
plt.title("Monthly Rainfall Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# 6. Dataset Splitting
# ==========================================

train_test_df = df[df["YEAR"] <= 2020]
future_df = df[df["YEAR"] > 2020]

print("Train/Test dataset:", train_test_df.shape)
print("Future dataset:", future_df.shape)

# ==========================================
# 7. Feature Selection
# ==========================================

features = [
"RH2M",
"GWETTOP",
"T2MDEW",
"T2M_MIN",
"WS2M",
"RAIN_LAG1",
"RAIN_LAG2",
"RAIN_LAG3",
"RAIN_ROLLING3",
"RAIN_ROLLING7",
"RAIN_CHANGE",
"RAIN_INTENSITY",
"TEMP_DEW_DIFF"
]

target = "PRECTOTCORR"

X = train_test_df[features]
y = train_test_df[target]

# ==========================================
# 8. Train Test Split (80/20)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

# ==========================================
# 9. Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 10. PCA
# ==========================================

pca = PCA(0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Number of PCA Components:", pca.n_components_)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Components")
plt.ylabel("Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

# ==========================================
# 11. Model Evaluation Function
# ==========================================

def evaluate_model(model, X_train, X_test, y_train, y_test, name):

    model.fit(X_train, y_train)

    y_train_pred = np.clip(model.predict(X_train), a_min=0, a_max=None)
    y_test_pred  = np.clip(model.predict(X_test),  a_min=0, a_max=None)

    return {
        "Model": name,
        "R2 Train": r2_score(y_train, y_train_pred),
        "R2 Test": r2_score(y_test, y_test_pred),
        "RMSE Train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "RMSE Test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "MAE Train": mean_absolute_error(y_train, y_train_pred),
        "MAE Test": mean_absolute_error(y_test, y_test_pred),
        "MSE Train": mean_squared_error(y_train, y_train_pred),
        "MSE Test": mean_squared_error(y_test, y_test_pred)
    }

# ==========================================
# 12. Train Models
# ==========================================

results = []

results.append(evaluate_model(
LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression"))

results.append(evaluate_model(
DecisionTreeRegressor(max_depth=10, random_state=42),
X_train, X_test, y_train, y_test, "Decision Tree"))

results.append(evaluate_model(
RandomForestRegressor(n_estimators=300, random_state=42),
X_train, X_test, y_train, y_test, "Random Forest"))

results.append(evaluate_model(
GradientBoostingRegressor(n_estimators=200),
X_train, X_test, y_train, y_test, "Gradient Boosting"))

xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)

results.append(evaluate_model(
xgb, X_train, X_test, y_train, y_test, "XGBoost"))

results.append(evaluate_model(
LGBMRegressor(n_estimators=500),
X_train, X_test, y_train, y_test, "LightGBM"))

results.append(evaluate_model(
SVR(kernel='rbf'), X_train_pca, X_test_pca, y_train, y_test, "SVR"))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE Test")

print(results_df)

# ==========================================
# 13. Model Comparison Plot
# ==========================================

plt.figure(figsize=(8,5))
sns.barplot(x="RMSE Test", y="Model", data=results_df)
plt.title("Model Comparison")
plt.show()

# ==========================================
# 14. Best Model (XGBoost)
# ==========================================

best_model = xgb
best_model.fit(X_train, y_train)

# ==========================================
# 15. Predictions for Future Data (2021–2025)
# ==========================================

X_future = future_df[features]

future_predictions = best_model.predict(X_future)
future_predictions = np.clip(future_predictions, a_min=0, a_max=None)  
future_df["Predicted_Rainfall"] = future_predictions

prediction_table = future_df[["DATE","PRECTOTCORR","Predicted_Rainfall"]]

prediction_table.rename(columns={
"DATE":"Date",
"PRECTOTCORR":"Actual Rainfall (mm)",
"Predicted_Rainfall":"Predicted Rainfall (mm)"
}, inplace=True)

print(prediction_table.head())

prediction_table.to_csv("rainfall_predictions_2021_2025.csv", index=False)

# ==========================================
# 16. Actual vs Predicted Plot
# ==========================================

y_test_pred = np.clip(best_model.predict(X_test), a_min=0, a_max=None)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.plot([y_test.min(), y_test.max()],
[y_test.min(), y_test.max()], color="red")
plt.show()

# ==========================================
# 17. Residual Plot
# ==========================================

residuals = y_test - y_test_pred

plt.scatter(y_test_pred, residuals)
plt.axhline(0,color="red")
plt.title("Residual Plot")
plt.show()

# ==========================================
# 18. Feature Importance
# ==========================================

importance = pd.DataFrame({
"Feature": features,
"Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")
plt.show()

print("Pipeline Completed Successfully.")