import os
import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "drowsiness_model.keras"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ==========================================
# TEST IMAGE
# ==========================================

image_path = input(
    "Enter image path: "
).strip().strip('"')

if not os.path.exists(image_path):

    print("Image not found:")
    print(image_path)
    exit()

# ==========================================
# READ IMAGE
# ==========================================

image = cv2.imread(image_path)

if image is None:

    print("Could not read image.")
    exit()

# ==========================================
# PREPROCESS
# ==========================================

image = cv2.resize(
    image,
    (96, 96)
)

# Convert to grayscale
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Normalize
image = image.astype("float32") / 255.0

# Add dimensions
image = np.expand_dims(
    image,
    axis=0
)

image = np.expand_dims(
    image,
    axis=-1
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    image,
    verbose=0
)[0][0]

print()
print("Prediction value:", prediction)

# ==========================================
# RESULT
# ==========================================

if prediction >= 0.5:

    result = "SLEEPING"
    confidence = prediction * 100

else:

    result = "AWAKE"
    confidence = (1 - prediction) * 100

print()
print("RESULT:", result)
print(
    f"Confidence: {confidence:.2f}%"
)