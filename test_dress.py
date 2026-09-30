import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/dress_colour_model.keras"
CLASS_FILE = "models/dress_colour_classes.txt"

model = load_model(MODEL_PATH)

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

image_path = input("Enter image path: ").strip()

img = cv2.imread(image_path)

if img is None:
    print("Could not read image.")
    exit()

img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)[0]

index = np.argmax(prediction)
confidence = prediction[index] * 100

print("\n==============================")
print("DRESS COLOUR RESULT")
print("==============================")
print("Predicted colour:", classes[index])
print("Confidence:", round(confidence, 2), "%")
print("==============================")