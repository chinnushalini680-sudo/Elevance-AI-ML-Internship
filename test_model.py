import cv2
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "models/sign_language_model.keras"
LABEL_PATH = "models/labels.json"

IMG_SIZE = 128

# ==============================
# CHECK TIME
# ==============================

current_hour = datetime.now().hour

if current_hour < 18 or current_hour >= 22:
    print("====================================")
    print("SIGN LANGUAGE DETECTION")
    print("====================================")
    print("The system works only from 6 PM to 10 PM.")
    print("Current time is outside the allowed time.")
    print("====================================")
    exit()

# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# ==============================
# OPEN CAMERA
# ==============================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("\nCamera started.")
print("Show one sign to the camera.")
print("Press Q to quit.")

# ==============================
# REAL-TIME PREDICTION
# ==============================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    # Resize image for model
    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Prediction
    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index] * 100

    predicted_sign = class_names[predicted_index]

    # ==============================
    # DISPLAY
    # ==============================

    text = f"{predicted_sign.upper()}  {confidence:.1f}%"

    cv2.rectangle(
        frame,
        (10, 10),
        (500, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Sign Language Detection", frame)

    # ==============================
    # QUIT
    # ==============================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ==============================
# CLOSE
# ==============================

camera.release()
cv2.destroyAllWindows()

print("Test completed.")