import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk


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
        + 5.0 * box_loss
    )

    return tf.reduce_mean(
        total_loss
    )


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

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

    original_height, original_width = image.shape[:2]

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

        animal = CLASSES[class_id]

        cx = float(prediction[i, 6])
        cy = float(prediction[i, 7])
        bw = float(prediction[i, 8])
        bh = float(prediction[i, 9])

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

        detections.append({
            "animal": animal,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return remove_duplicates(
        detections
    )


# ==========================================
# REMOVE DUPLICATES
# ==========================================

def remove_duplicates(detections):

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    final = []

    for detection in detections:

        duplicate = False

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        for old in final:

            ox1 = old["x1"]
            oy1 = old["y1"]
            ox2 = old["x2"]
            oy2 = old["y2"]

            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            intersection = (
                ix2 - ix1
            ) * (
                iy2 - iy1
            )

            area1 = (
                x2 - x1
            ) * (
                y2 - y1
            )

            area2 = (
                ox2 - ox1
            ) * (
                oy2 - oy1
            )

            union = (
                area1
                + area2
                - intersection
            )

            if union > 0:

                iou = (
                    intersection /
                    union
                )

                if iou > 0.50:

                    duplicate = True
                    break

        if not duplicate:

            final.append(
                detection
            )

    return final


# ==========================================
# GLOBAL RESULT
# ==========================================

current_photo = None


# ==========================================
# OPEN IMAGE
# ==========================================

def open_image():

    global current_photo

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

    image = cv2.imread(path)

    if image is None:

        messagebox.showerror(
            "Error",
            "Unable to open image."
        )

        return

    # ======================================
    # DETECTION
    # ======================================

    detections = detect_animals(
        image
    )

    # ======================================
    # COUNTS
    # ======================================

    counts = {
        animal: 0
        for animal in CLASSES
    }

    carnivore_count = 0

    for detection in detections:

        animal = detection["animal"]

        counts[animal] += 1

        if animal in CARNIVORES:

            carnivore_count += 1

    total = len(detections)

    # ======================================
    # DRAW BOXES
    # ======================================

    for detection in detections:

        animal = detection["animal"]

        confidence = detection["confidence"]

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
                200,
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

    # ======================================
    # UPDATE GUI
    # ======================================

    result_text.set(
        f"Total Animals: {total}\n"
        f"Carnivores: {carnivore_count}"
    )

    count_text.set(
        f"CAT       : {counts['cat']}\n"
        f"COW       : {counts['cow']}\n"
        f"DOG       : {counts['dog']}\n"
        f"ELEPHANT  : {counts['elephant']}\n"
        f"LION      : {counts['lion']}"
    )

    saved_text.set(
        f"Saved: {output_path}"
    )

    # ======================================
    # DISPLAY IMAGE
    # ======================================

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    pil_image.thumbnail(
        (700, 500)
    )

    current_photo = ImageTk.PhotoImage(
        pil_image
    )

    image_label.config(
        image=current_photo
    )


# ==========================================
# CLEAR RESULT
# ==========================================

def clear_result():

    global current_photo

    current_photo = None

    image_label.config(
        image=""
    )

    result_text.set(
        "Total Animals: 0\n"
        "Carnivores: 0"
    )

    count_text.set(
        "CAT       : 0\n"
        "COW       : 0\n"
        "DOG       : 0\n"
        "ELEPHANT  : 0\n"
        "LION      : 0"
    )

    saved_text.set(
        ""
    )


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title(
    "Multi-Animal Detection System"
)

root.geometry(
    "1050x700"
)

root.resizable(
    False,
    False
)


# ==========================================
# HEADER
# ==========================================

header = tk.Label(
    root,
    text="MULTI-ANIMAL DETECTION SYSTEM",
    font=("Arial", 24, "bold")
)

header.pack(
    pady=20
)


subtitle = tk.Label(
    root,
    text="Cat • Cow • Dog • Elephant • Lion",
    font=("Arial", 12)
)

subtitle.pack()


# ==========================================
# MAIN FRAME
# ==========================================

main_frame = tk.Frame(
    root
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=20
)


# ==========================================
# IMAGE AREA
# ==========================================

image_frame = tk.Frame(
    main_frame,
    width=720,
    height=500
)

image_frame.pack(
    side="left",
    padx=10
)

image_frame.pack_propagate(
    False
)


image_label = tk.Label(
    image_frame,
    text="Select an image to begin",
    font=("Arial", 14)
)

image_label.pack(
    expand=True
)


# ==========================================
# RESULT AREA
# ==========================================

result_frame = tk.Frame(
    main_frame,
    width=280,
    height=500
)

result_frame.pack(
    side="right",
    fill="y",
    padx=10
)


# ==========================================
# RESULT TITLE
# ==========================================

result_title = tk.Label(
    result_frame,
    text="DETECTION RESULT",
    font=("Arial", 16, "bold")
)

result_title.pack(
    pady=10
)


# ==========================================
# TOTAL RESULT
# ==========================================

result_text = tk.StringVar()

result_text.set(
    "Total Animals: 0\n"
    "Carnivores: 0"
)

result_label = tk.Label(
    result_frame,
    textvariable=result_text,
    font=("Arial", 14, "bold"),
    justify="left"
)

result_label.pack(
    pady=15
)


# ==========================================
# ANIMAL COUNTS
# ==========================================

count_title = tk.Label(
    result_frame,
    text="Animal Counts",
    font=("Arial", 13, "bold")
)

count_title.pack(
    pady=5
)


count_text = tk.StringVar()

count_text.set(
    "CAT       : 0\n"
    "COW       : 0\n"
    "DOG       : 0\n"
    "ELEPHANT  : 0\n"
    "LION      : 0"
)

count_label = tk.Label(
    result_frame,
    textvariable=count_text,
    font=("Courier New", 12),
    justify="left"
)

count_label.pack(
    pady=10
)


# ==========================================
# BUTTONS
# ==========================================

open_button = tk.Button(
    result_frame,
    text="OPEN IMAGE",
    command=open_image,
    width=22,
    height=2,
    font=("Arial", 11, "bold")
)

open_button.pack(
    pady=15
)


clear_button = tk.Button(
    result_frame,
    text="CLEAR",
    command=clear_result,
    width=22,
    height=2,
    font=("Arial", 11)
)

clear_button.pack(
    pady=5
)


exit_button = tk.Button(
    result_frame,
    text="EXIT",
    command=root.destroy,
    width=22,
    height=2,
    font=("Arial", 11)
)

exit_button.pack(
    pady=5
)


# ==========================================
# SAVED RESULT
# ==========================================

saved_text = tk.StringVar()

saved_label = tk.Label(
    result_frame,
    textvariable=saved_text,
    font=("Arial", 9),
    wraplength=250,
    justify="left"
)

saved_label.pack(
    pady=20
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