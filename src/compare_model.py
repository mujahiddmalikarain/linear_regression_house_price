import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # 👈 Add this


# Load data
data = pd.read_csv("data/data.csv")
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# --- Train Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_preds = lr_model.predict(X)

# --- Train Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_preds = rf_model.predict(X)

# Save models
joblib.dump(lr_model, "models/model_linear.pkl")
joblib.dump(rf_model, "models/model_rf.pkl")
# --- Evaluation Metrics ---
def evaluate(name, y_true, y_pred):
    print(f"📊 {name} Performance")
    print(f"  MAE  : {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
    print(f"  R²   : {r2_score(y_true, y_pred):.4f}")
    print("-" * 40)

evaluate("Linear Regression", y, lr_preds)
evaluate("Random Forest", y, rf_preds)

# --- Plot Comparison (Area vs Price) ---
plt.figure(figsize=(10, 6))
plt.scatter(data['Area'], y, label="Actual", color='blue')
plt.scatter(data['Area'], lr_preds, label="Linear Predicted", color='green', marker='x')
plt.scatter(data['Area'], rf_preds, label="RF Predicted", color='red', marker='^')

plt.title("Model Comparison: Area vs Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)

# Save plot
import os
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/model_comparison.png")
print("📈 Saved comparison chart: visuals/model_comparison.png")
