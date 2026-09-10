import joblib
import os

model_path = 'models/car_price_model.pkl'

print(f"Original size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

# Load model
model = joblib.load(model_path)

# Save with compression (3 is a good balance between size and load speed)
joblib.dump(model, model_path, compress=3)

print(f"Compressed size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
