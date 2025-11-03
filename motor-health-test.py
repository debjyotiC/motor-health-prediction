import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load

# --- 1. Configuration ---
TIME_STEPS, STEP = 40, 20
N_FEATURES = 6

## IMPROVEMENT: Use a clear mapping for classes to avoid confusion
CLASS_MAP = {
    0: 'MotorOn',
    1: 'MotorOff',
    2: 'MotorBadFan',
    3: 'MotorNoFan'
}

class_names = [CLASS_MAP[i] for i in sorted(CLASS_MAP.keys())]

MODEL_PATH = "models/motor_health.keras"
TEST_FILE_PATH = "dataset/digested/train.csv"
IMAGE_PATH = "images/blind_test_cnn_confusion_matrix.png"

# --- 2. Load and Preprocess Test Data ---
print(f"--- Loading test data from: {TEST_FILE_PATH} ---")
df_test = pd.read_csv(TEST_FILE_PATH, skiprows=1)
signals = df_test.iloc[:, :N_FEATURES].values.astype(np.float32)

labels = df_test.iloc[:, N_FEATURES].values.astype(int)

print("Unique labels found in test file:", np.unique(labels, return_counts=True))

# --- 3. Create Windows from Test Data ---
windows, window_labels = [], []
for i in range(0, len(signals) - TIME_STEPS + 1, STEP):
    windows.append(signals[i: i + TIME_STEPS])
    window_labels.append(labels[i + TIME_STEPS - 1])

X_test = np.array(windows)
y_test = np.array(window_labels)

if len(X_test) == 0:
    print(f"Error: Not enough data in '{TEST_FILE_PATH}' to create a single window.")
    exit()

# --- 4. Reshape Data for 2D CNN Input ---
X_test_reshaped = X_test.reshape((X_test.shape[0], TIME_STEPS, N_FEATURES, 1))
print(f"Reshaped test data for 2D CNN input: {X_test_reshaped.shape}")

# --- 5. Load Model and Run Inference ---
print(f"--- Loading model from: {MODEL_PATH} ---")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# Get model predictions
y_probs = model.predict(X_test_reshaped)

# Option 1: Use this line if model has a SOFTMAX output layer (2 units)
y_pred = np.argmax(y_probs, axis=1)

# --- 6. Evaluate and Report Performance ---
print("\n--- Final Performance Report on Unseen Data ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
if cm.shape[0] > 1:
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        if i < len(per_class_accuracy):
            print(f"{name:<10}: {per_class_accuracy[i]:.2%}")
else:
    print("\nWarning: Only one class present in the test labels.")

# --- 7. Plot Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names[:cm.shape[1]],
            yticklabels=class_names[:cm.shape[0]])
plt.title("Confusion Matrix - Blind Test Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(IMAGE_PATH)
plt.show()