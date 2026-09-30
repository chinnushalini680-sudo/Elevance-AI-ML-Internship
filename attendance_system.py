import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/drowsiness_model.keras"
FACE_CASCADE_PATH = "../haarcascade_frontalface_default.xml"

IMG_SIZE = 160

# Your model:
# prediction >= 0.50 = SLEEPING
# prediction <  0.50 = AWAKE
SLEEP_THRESHOLD = 0.50


# ============================================================
# CHECK FILES
# ============================================================

print("\n==============================")
print("DROWSINESS DETECTION")
print("==============================")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found!")
    print("Looking for:", MODEL_PATH)
    input("Press Enter to close...")
    exit()

if not os.path.exists(FACE_CASCADE_PATH):
    print("ERROR: Haar cascade not found!")
    print("Looking for:", FACE_CASCADE_PATH)
    input("Press Enter to close...")
    exit()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

print("Loading face detector...")

face_detector = cv2.CascadeClassifier(
    FACE_CASCADE_PATH
)

if face_detector.empty():
    print("ERROR: Face detector could not be loaded!")
    input("Press Enter to close...")
    exit()

print("Face detector loaded successfully!")


# ============================================================
# SELECT IMAGE
# ============================================================

root = tk.Tk()
root.withdraw()

print("\nSelect an image...")

image_path = filedialog.askopenfilename(
    title="Select Test Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
        ("All Files", "*.*")
    ]
)

root.destroy()

if not image_path:
    print("No image selected.")
    exit()


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not read image!")
    input("Press Enter to close...")
    exit()

output = image.copy()


# ============================================================
# FACE DETECTION
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Improve contrast
gray = cv2.equalizeHist(gray)


faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=4,
    minSize=(40, 40)
)


print("\n==============================")
print("RESULT")
print("==============================")

print("Faces detected:", len(faces))


# ============================================================
# COUNTERS
# ============================================================

total_people = len(faces)
sleeping_people = 0
awake_people = 0


# ============================================================
# PROCESS EACH PERSON
# ============================================================

for person_number, (x, y, w, h) in enumerate(
    faces,
    start=1
):

    # --------------------------------------------------------
    # GET FACE
    # --------------------------------------------------------

    face = image[
        y:y + h,
        x:x + w
    ]

    if face.size == 0:
        continue


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face_resized = cv2.resize(
        face_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    face_array = np.asarray(
        face_resized,
        dtype=np.float32
    )


    # IMPORTANT:
    # This normalization must match your training code.
    face_array = face_array / 127.5 - 1.0

    face_array = np.expand_dims(
        face_array,
        axis=0
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        face_array,
        verbose=0
    )[0][0]

    prediction = float(prediction)


    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    if prediction >= SLEEP_THRESHOLD:

        status = "SLEEPING"

        sleeping_people += 1

        # RED
        color = (0, 0, 255)

        confidence = prediction

    else:

        status = "AWAKE"

        awake_people += 1

        # GREEN
        color = (0, 255, 0)

        confidence = 1.0 - prediction


    confidence *= 100


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Person {person_number}: "
        f"{status} | "
        f"Prediction = {prediction:.4f} | "
        f"Confidence = {confidence:.1f}%"
    )


    # --------------------------------------------------------
    # DRAW FACE BOX
    # --------------------------------------------------------

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        color,
        3
    )


    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    label = (
        f"Person {person_number}: {status}"
    )

    confidence_label = (
        f"{confidence:.1f}%"
    )


    # Background
    label_height = 60

    top = max(
        0,
        y - label_height
    )

    cv2.rectangle(
        output,
        (x, top),
        (x + w, y),
        color,
        -1
    )


    # Status
    cv2.putText(
        output,
        label,
        (x + 5, y - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # Confidence
    cv2.putText(
        output,
        confidence_label,
        (x + 5, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


# ============================================================
# FINAL COUNTS
# ============================================================

print("\n==============================")
print("FINAL COUNTS")
print("==============================")

print("Total people :", total_people)
print("Sleeping     :", sleeping_people)
print("Awake        :", awake_people)


# ============================================================
# SUMMARY BOX
# ============================================================

cv2.rectangle(
    output,
    (10, 10),
    (370, 145),
    (0, 0, 0),
    -1
)

cv2.putText(
    output,
    f"Total People: {total_people}",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    f"Sleeping: {sleeping_people}",
    (20, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)

cv2.putText(
    output,
    f"Awake: {awake_people}",
    (20, 130),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

result_path = os.path.join(
    "results",
    "drowsiness_result.jpg"
)

cv2.imwrite(
    result_path,
    output
)

print("\nResult saved to:")
print(result_path)


# ============================================================
# SHOW RESULT
# ============================================================

cv2.namedWindow(
    "Drowsiness Detection",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Drowsiness Detection",
    1000,
    700
)

cv2.imshow(
    "Drowsiness Detection",
    output
)


# ============================================================
# POPUP
# ============================================================

root = tk.Tk()
root.withdraw()

if sleeping_people > 0:

    messagebox.showwarning(
        "Drowsiness Alert",
        f"Total people: {total_people}\n"
        f"Sleeping people: {sleeping_people}\n"
        f"Awake people: {awake_people}\n\n"
        f"WARNING: {sleeping_people} "
        f"sleeping person(s) detected!"
    )

else:

    messagebox.showinfo(
        "Drowsiness Result",
        f"Total people: {total_people}\n"
        f"Sleeping people: {sleeping_people}\n"
        f"Awake people: {awake_people}\n\n"
        "No sleeping people detected."
    )

root.destroy()


# ============================================================
# WAIT
# ============================================================

print("\nPress any key on the result window to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()