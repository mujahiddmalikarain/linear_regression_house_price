import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load data from CSV
data = pd.read_csv('data/data.csv')  # Or 'multivariable_data.csv' if renamed

# Prepare features (X) and target (y)
X = data[['Area', 'Bedrooms', 'Bathrooms']].values
y = data['Price'].values

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

# Predict prices
predicted_prices = model.predict(X)

# Plot Area vs Price
plt.figure(figsize=(8, 5))
plt.scatter(data['Area'].values, y, color='blue', label='Actual Prices')
plt.plot(data['Area'].values, predicted_prices, color='red', label='Predicted Prices')

plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price Prediction (Multi-variable Linear Regression)")
plt.legend()
plt.grid(True)

# Save plot
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/regression_plot.png")
