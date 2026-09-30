import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
import os


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "multi_animal_detector.keras"

IMG_SIZE = 224
MAX_OBJECTS = 10

CLASSES = [
    "cat",
    "cow",
    "dog",
    "elephant",
    "lion"
]

CARNIVORES = {
    "cat",
    "dog",
    "lion"
}

RESULTS_FOLDER = "results"


# ==========================================
# CREATE RESULTS FOLDER
# ==========================================

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# ==========================================
# LOSS FUNCTION
# ==========================================

def detector_loss(y_true, y_pred):

    true_presence = y_true[:, :, 0:1]
    pred_presence = y_pred[:, :, 0:1]

    presence_loss = tf.keras.backend.binary_crossentropy(
        true_presence,
        pred_presence
    )

    true_class = y_true[:, :, 1:6]
    pred_class = y_pred[:, :, 1:6]

    class_loss = tf.keras.backend.categorical_crossentropy(
        true_class,
        pred_class
    )

    object_mask = tf.squeeze(
        true_presence,
        axis=-1
    )

    class_loss = class_loss * object_mask

    true_box = y_true[:, :, 6:10]
    pred_box = y_pred[:, :, 6:10]

    box_loss = tf.reduce_sum(
        tf.abs(
            true_box - pred_box
        ),
        axis=-1
    )

    box_loss = box_loss * object_mask

    presence_loss = tf.reduce_mean(
        presence_loss,
        axis=[1, 2]
    )

    class_loss = tf.reduce_mean(
        class_loss,
        axis=1
    )

    box_loss = tf.reduce_mean(
        box_loss,
        axis=1
    )

    total_loss = (
        presence_loss
        + class_loss
        + (5.0 * box_loss)
    )

    return tf.reduce_mean(
        total_loss
    )


# ==========================================
# LOAD MODEL
# ==========================================

print()
print("==============================")
print("LOADING MODEL")
print("==============================")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "detector_loss": detector_loss
    }
)

print("Model loaded successfully!")


# ==========================================
# DETECT ANIMALS
# ==========================================

def detect_animals(image):

    original_height = image.shape[0]
    original_width = image.shape[1]

    img = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(
        np.float32
    )

    img = tf.keras.applications.mobilenet_v2.preprocess_input(
        img
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(
        img,
        verbose=0
    )[0]

    detections = []

    for i in range(MAX_OBJECTS):

        presence = float(
            prediction[i, 0]
        )

        if presence < 0.35:
            continue

        class_scores = prediction[
            i,
            1:6
        ]

        class_id = int(
            np.argmax(
                class_scores
            )
        )

        class_confidence = float(
            class_scores[class_id]
        )

        confidence = (
            presence *
            class_confidence
        )

        if confidence < 0.25:
            continue

        animal = CLASSES[
            class_id
        ]

        cx = float(
            prediction[i, 6]
        )

        cy = float(
            prediction[i, 7]
        )

        bw = float(
            prediction[i, 8]
        )

        bh = float(
            prediction[i, 9]
        )

        x1 = int(
            (cx - bw / 2)
            * original_width
        )

        y1 = int(
            (cy - bh / 2)
            * original_height
        )

        x2 = int(
            (cx + bw / 2)
            * original_width
        )

        y2 = int(
            (cy + bh / 2)
            * original_height
        )

        x1 = max(
            0,
            min(
                x1,
                original_width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                original_height - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                original_width - 1
            )
        )

        y2 = max(
            0,
            min(
                y2,
                original_height - 1
            )
        )

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "animal": animal,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        )

    return detections


# ==========================================
# REMOVE DUPLICATES
# ==========================================

def remove_duplicates(detections):

    if len(detections) <= 1:
        return detections

    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    final_detections = []

    for detection in detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        duplicate = False

        for existing in final_detections:

            ox1 = existing["x1"]
            oy1 = existing["y1"]
            ox2 = existing["x2"]
            oy2 = existing["y2"]

            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)

            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            intersection = (
                (ix2 - ix1)
                *
                (iy2 - iy1)
            )

            area1 = (
                (x2 - x1)
                *
                (y2 - y1)
            )

            area2 = (
                (ox2 - ox1)
                *
                (oy2 - oy1)
            )

            union = (
                area1
                +
                area2
                -
                intersection
            )

            if union <= 0:
                continue

            iou = (
                intersection /
                union
            )

            if iou > 0.50:

                duplicate = True
                break

        if not duplicate:

            final_detections.append(
                detection
            )

    return final_detections


# ==========================================
# OPEN IMAGE
# ==========================================

def open_image():

    path = filedialog.askopenfilename(
        title="Select Animal Image",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp"
            ),
            (
                "All files",
                "*.*"
            )
        ]
    )

    if not path:
        return

    image = cv2.imread(
        path
    )

    if image is None:

        messagebox.showerror(
            "Error",
            "Unable to open image."
        )

        return

    print()
    print("==============================")
    print("DETECTING ANIMALS")
    print("==============================")

    # ======================================
    # DETECT
    # ======================================

    detections = detect_animals(
        image
    )

    detections = remove_duplicates(
        detections
    )

    # ======================================
    # COUNT
    # ======================================

    total_animals = len(
        detections
    )

    carnivore_count = 0

    animal_counts = {
        animal: 0
        for animal in CLASSES
    }

    for detection in detections:

        animal = detection[
            "animal"
        ]

        animal_counts[
            animal
        ] += 1

        if animal in CARNIVORES:

            carnivore_count += 1

    # ======================================
    # DRAW BOXES
    # ======================================

    for detection in detections:

        animal = detection[
            "animal"
        ]

        confidence = detection[
            "confidence"
        ]

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        if animal in CARNIVORES:

            color = (
                0,
                0,
                255
            )

        else:

            color = (
                0,
                255,
                0
            )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = (
            f"{animal.upper()} "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    30,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ======================================
    # INFORMATION PANEL
    # ======================================

    panel_height = 150

    cv2.rectangle(
        image,
        (0, 0),
        (400, panel_height),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        f"Total animals: {total_animals}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        f"Carnivores: {carnivore_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    # Show individual counts

    count_text = (
        f"Cat:{animal_counts['cat']} "
        f"Cow:{animal_counts['cow']} "
        f"Dog:{animal_counts['dog']}"
    )

    cv2.putText(
        image,
        count_text,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    count_text2 = (
        f"Elephant:{animal_counts['elephant']} "
        f"Lion:{animal_counts['lion']}"
    )

    cv2.putText(
        image,
        count_text2,
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # ======================================
    # SAVE RESULT
    # ======================================

    filename = os.path.basename(
        path
    )

    output_path = os.path.join(
        RESULTS_FOLDER,
        "result_" + filename
    )

    cv2.imwrite(
        output_path,
        image
    )

    print()
    print("==============================")
    print("RESULT")
    print("==============================")

    for animal in CLASSES:

        print(
            f"{animal}: "
            f"{animal_counts[animal]}"
        )

    print()
    print(
        "Total animals:",
        total_animals
    )

    print(
        "Carnivores:",
        carnivore_count
    )

    print()
    print(
        "Result saved to:"
    )

    print(
        output_path
    )

    print("==============================")

    # ======================================
    # SHOW RESULT
    # ======================================

    cv2.imshow(
        "MULTI-ANIMAL DETECTION",
        image
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # ======================================
    # MESSAGE
    # ======================================

    messagebox.showinfo(
        "Detection Complete",
        f"Animals detected: {total_animals}\n"
        f"Carnivores: {carnivore_count}\n\n"
        f"Result saved in:\n"
        f"{RESULTS_FOLDER}"
    )


# ==========================================
# GUI
# ==========================================

root = tk.Tk()

root.title(
    "Multi-Animal Detection System"
)

root.geometry(
    "550x400"
)

root.resizable(
    False,
    False
)


title = tk.Label(
    root,
    text="MULTI-ANIMAL DETECTION SYSTEM",
    font=("Arial", 18, "bold")
)

title.pack(
    pady=50
)


description = tk.Label(
    root,
    text="Select an image containing multiple animals",
    font=("Arial", 12)
)

description.pack(
    pady=10
)


button = tk.Button(
    root,
    text="OPEN IMAGE",
    command=open_image,
    width=25,
    height=2,
    font=("Arial", 12, "bold")
)

button.pack(
    pady=30
)


exit_button = tk.Button(
    root,
    text="EXIT",
    command=root.destroy,
    width=15,
    height=1,
    font=("Arial", 10)
)

exit_button.pack(
    pady=10
)


# ==========================================
# START
# ==========================================

print()
print("==============================")
print("MULTI-ANIMAL DETECTION SYSTEM")
print("==============================")
print("Ready!")
print("==============================")

root.mainloop()