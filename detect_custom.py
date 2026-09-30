import cv2
import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

IMAGE_FOLDER = "dataset/images"
LABEL_FOLDER = "dataset/labels"

# IMPROVED MODEL
MODEL_PATH = "m/car_colour_model_improved.keras"

CLASS_NAMES = {
    0: "blue",
    1: "other",
    2: "person"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading improved model...")

model = load_model(MODEL_PATH)

print("Improved model loaded successfully.")


# ============================================================
# FIND IMAGES
# ============================================================

images = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    print("No images found.")
    exit()

image_name = images[0]

image_path = os.path.join(
    IMAGE_FOLDER,
    image_name
)

label_path = os.path.join(
    LABEL_FOLDER,
    os.path.splitext(image_name)[0] + ".json"
)

print()
print("Testing image:", image_name)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:
    print("Could not read image.")
    exit()


# ============================================================
# LOAD ANNOTATION
# ============================================================

if not os.path.exists(label_path):
    print("Label file not found:", label_path)
    exit()

with open(label_path, "r") as f:
    data = json.load(f)

annotations = data["annotations"]


# ============================================================
# COUNTERS
# ============================================================

blue_count = 0
other_count = 0
person_count = 0


print()
print("==============================")
print("INDIVIDUAL PREDICTIONS")
print("==============================")


# ============================================================
# PROCESS EACH OBJECT
# ============================================================

for number, ann in enumerate(annotations, start=1):

    x1 = int(ann["x1"])
    y1 = int(ann["y1"])
    x2 = int(ann["x2"])
    y2 = int(ann["y2"])

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        continue


    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    resized = cv2.resize(
        crop,
        (224, 224)
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    arr = np.expand_dims(
        rgb.astype(np.float32),
        axis=0
    )

    arr = preprocess_input(arr)


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        arr,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(prediction)
    )

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    blue_conf = float(
        prediction[0]
    ) * 100

    other_conf = float(
        prediction[1]
    ) * 100

    person_conf = float(
        prediction[2]
    ) * 100


    # Actual annotation
    actual_class = int(
        ann.get("class_id", -1)
    )

    actual_name = CLASS_NAMES.get(
        actual_class,
        "unknown"
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print()
    print(f"Object {number}")
    print("Actual    :", actual_name)
    print("Predicted :", predicted_name)

    print(
        f"Blue      : {blue_conf:.2f}%"
    )

    print(
        f"Other     : {other_conf:.2f}%"
    )

    print(
        f"Person    : {person_conf:.2f}%"
    )


    # ========================================================
    # DRAW RECTANGLE
    # ========================================================

    if predicted_class == 0:

        # BLUE CAR → RED RECTANGLE
        rectangle_color = (0, 0, 255)

        label = "Blue Car"

        blue_count += 1


    elif predicted_class == 1:

        # OTHER CAR → BLUE RECTANGLE
        rectangle_color = (255, 0, 0)

        label = "Other Car"

        other_count += 1


    else:

        # PERSON → GREEN RECTANGLE
        rectangle_color = (0, 255, 0)

        label = "Person"

        person_count += 1


    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        rectangle_color,
        3
    )


    confidence = max(
        blue_conf,
        other_conf,
        person_conf
    )


    cv2.putText(
        image,
        f"{label} {confidence:.1f}%",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        rectangle_color,
        2
    )


# ============================================================
# COUNTS
# ============================================================

total_cars = (
    blue_count +
    other_count
)


print()
print("==============================")
print("DETECTION RESULT")
print("==============================")

print(
    "Blue Cars :",
    blue_count
)

print(
    "Other Cars:",
    other_count
)

print(
    "Total Cars:",
    total_cars
)

print(
    "People    :",
    person_count
)

print("==============================")


# ============================================================
# DISPLAY
# ============================================================

cv2.imshow(
    "Custom Car Colour Detection",
    image
)

print()
print("Press any key on the image window to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()