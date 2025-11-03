import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    Flatten, Activation
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


# --- Configuration ---
FILE_PATH = "dataset/digested/train.csv"
TIME_STEPS, STEP = 40, 20
N_FEATURES = 6

# --- Load and Preprocess Data ---
df = pd.read_csv(FILE_PATH, skiprows=1)
signals = df.iloc[:, :N_FEATURES].values.astype(np.float32)
labels = df.iloc[:, 6].values.astype(int)

print("Sample signal:", signals[0])

windows, window_labels = [], []
for i in range(0, len(signals) - TIME_STEPS + 1, STEP):
    windows.append(signals[i: i + TIME_STEPS])
    window_labels.append(labels[i + TIME_STEPS - 1])

X = np.array(windows)
y = np.array(window_labels)
print(f"Total windows created: {len(X)}")

# --- Train-Validation-Test Split ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)
print(f"Training samples: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# --- Data Augmentation (training set only) ---
def augment_data(X, y, noise_std=0.02, scale_range=(0.9, 1.1)):
    X_aug, y_aug = list(X), list(y)
    for i in range(len(X)):
        noise = np.random.normal(0, noise_std, X[i].shape)
        scaled = np.random.uniform(*scale_range)
        X_aug.append((X[i] + noise) * scaled)
        y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)

X_train_aug, y_train_aug = augment_data(X_train, y_train)
print(f"Augmented training samples: {len(X_train_aug)}")

# --- One-Hot Encode Labels ---
num_classes = len(np.unique(y_train_aug))
y_train_aug = to_categorical(y_train_aug, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)
print(f"Number of classes: {num_classes}")

# --- Reshape Data ---
X_train_reshaped = X_train_aug.reshape((X_train_aug.shape[0], TIME_STEPS, N_FEATURES, 1))
X_val_reshaped = X_val.reshape((X_val.shape[0], TIME_STEPS, N_FEATURES, 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], TIME_STEPS, N_FEATURES, 1))
print(f"Reshaped training data: {X_train_reshaped.shape}")

# --- Compute Class Weights ---
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.arange(num_classes),
    y=np.argmax(y_train_aug, axis=1)
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class Weights: {class_weight_dict}")

# --- Build Optimized CNN Model ---
def build_simple_cnn(input_shape=(TIME_STEPS, N_FEATURES, 1), num_classes=4):
    l2_rate = 1e-4
    model = Sequential([
        Conv2D(24, (7, 3), padding='same', kernel_regularizer=l2(l2_rate),
               input_shape=input_shape, name="conv1"),
        BatchNormalization(name="bn1"), Activation('relu', name="relu1"),
        MaxPooling2D((2, 1), name="pool1"),

        Conv2D(22, (7, 3), padding='same', kernel_regularizer=l2(l2_rate), name="conv2"),
        BatchNormalization(name="bn2"), Activation('relu', name="relu2"),
        MaxPooling2D((2, 2), name="pool2"),

        Conv2D(20, (3, 3), padding='same', kernel_regularizer=l2(l2_rate), name="conv3"),
        BatchNormalization(name="bn3"), Activation('relu', name="relu3"),
        MaxPooling2D((2, 1), name="pool3"),

        Conv2D(10, (3, 3), padding='same', kernel_regularizer=l2(l2_rate), name="conv4"),
        BatchNormalization(name="bn4"), Activation('relu', name="relu4"),
        MaxPooling2D((2, 1), name="pool4"),

        Flatten(name="flatten"),
        Dense(40, activation='relu', kernel_regularizer=l2(l2_rate), name="dense1"),
        Dropout(0.5, name="dropout"),
        Dense(num_classes, activation='softmax', name="output")
    ])
    return model

# --- Compile and Train ---
model = build_simple_cnn(num_classes=num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5, verbose=1)

history = model.fit(
    X_train_reshaped, y_train_aug,
    validation_data=(X_val_reshaped, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# --- Evaluate on Blind Test Set ---
print("\n--- Evaluating on Blind Test Set ---")
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=1)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# --- Save Model ---
os.makedirs("models", exist_ok=True)
model.save("models/motor_health.keras")
print("Model saved successfully.")

# --- Plot Metrics ---
os.makedirs("images", exist_ok=True)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("images/training_curves_optimized_cnn.png")
plt.show()

# --- Confusion Matrix ---
y_pred_probs = model.predict(X_test_reshaped)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["MotorOn", "MotorOff", "MotorBadFan", "MotorNoFan"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.show()
