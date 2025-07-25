import joblib

# Load trained model
model = joblib.load("models/model.pkl")

try:
    area = float(input("📏 Enter house area (sq ft): "))
    bedrooms = int(input("🛏️ Enter number of bedrooms: "))
    bathrooms = int(input("🛁 Enter number of bathrooms: "))

    features = [[area, bedrooms, bathrooms]]
    prediction = model.predict(features)

    print(f"\n💰 Predicted house price: ${prediction[0]:,.2f}")
except Exception as e:
    print("❌ Error:", e)
