import tensorflow as tf
import numpy as np
import joblib

MODEL_PATH = "models/motor_model.keras"
SCALER_PATH = "models/scaler.pkl"

# Load the model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = joblib.load(f)

input_shape = model.input_shape[1:]
print(f"Model input shape: {input_shape}")

data = np.random.rand(1, *input_shape).astype(np.float32)

original_shape = data.shape
num_features = original_shape[-1]
reshaped_data = data.reshape(-1, num_features)
scaled_data_reshaped = scaler.transform(reshaped_data)
scaled_data = scaled_data_reshaped.reshape(original_shape)


pred = np.argmax(model.predict(scaled_data))

print(f"Prediction: {pred}")