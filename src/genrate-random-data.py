import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 rows
data = {
    "area": np.random.randint(600, 4000, 1000),
    "bedrooms": np.random.randint(1, 6, 1000),
    "bathrooms": np.random.randint(1, 4, 1000),
    "location": np.random.choice(["City", "Suburb", "Rural"], 1000),
}

# Generate realistic price based on features + noise
data["price"] = (
    data["area"] * 100 +
    np.array(data["bedrooms"]) * 5000 +
    np.array(data["bathrooms"]) * 7000 +
    np.array([15000 if loc == "City" else 8000 if loc == "Suburb" else 3000 for loc in data["location"]]) +
    np.random.normal(0, 10000, 1000)  # random noise
).astype(int)

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("house_price_1000.csv", index=False)

print("✅ house_price_1000.csv generated successfully!")
