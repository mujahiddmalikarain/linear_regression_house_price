import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Step 1: Load dataset
df = pd.read_csv("data/house_price_1000.csv")

# Step 2: Encode categorical column (location)
le = LabelEncoder()
df["location_encoded"] = le.fit_transform(df["location"])

# Step 3: Define features and target
X = df[["area", "bedrooms", "bathrooms", "location_encoded"]]
y = df["price"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📏 Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"📈 R² Score (Model Accuracy): {r2:.2f}")

# Step 7: Real-time prediction (example)
new_data = pd.DataFrame([{
    "area": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "location_encoded": le.transform(["City"])[0]
}])
predicted_price = model.predict(new_data)
print(f"💰 Predicted price for new house: ${predicted_price[0]:,.0f}")
