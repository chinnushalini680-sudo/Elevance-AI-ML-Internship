import cv2
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


# ============================================================
# PATHS
# ============================================================

DROWSINESS_MODEL = "models/drowsiness_model.keras"
AGE_MODEL = "age_model.keras"
AGE_CLASSES_FILE = "age_classes.txt"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(DROWSINESS_MODEL):
    messagebox.showerror(
        "Error",
        f"Drowsiness model not found:\n{DROWSINESS_MODEL}"
    )
    raise SystemExit

if not os.path.exists(AGE_MODEL):
    messagebox.showerror(
        "Error",
        f"Age model not found:\n{AGE_MODEL}"
    )
    raise SystemExit

if not os.path.exists(AGE_CLASSES_FILE):
    messagebox.showerror(
        "Error",
        f"Age classes file not found:\n{AGE_CLASSES_FILE}"
    )
    raise SystemExit


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading drowsiness model...")
drowsiness_model = tf.keras.models.load_model(
    DROWSINESS_MODEL
)

print("Loading age model...")
age_model = tf.keras.models.load_model(
    AGE_MODEL
)

with open(AGE_CLASSES_FILE, "r") as f:
    AGE_CLASSES = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

print()
print("Age classes:")
for i, age in enumerate(AGE_CLASSES):
    print(i, "=", age)

print()
print("Models loaded successfully.")


# ============================================================
# SETTINGS
# ============================================================

DROWSINESS_CLASSES = [
    "awake",
    "sleeping"
]


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# DROWSINESS PREDICTION
# ============================================================

def predict_drowsiness(face):

    # Convert BGR to grayscale
    gray_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to drowsiness model size
    resized = cv2.resize(
        gray_face,
        (128, 128)
    )

    # Normalize
    resized = resized.astype(
        "float32"
    ) / 255.0

    # Add channel dimension
    # 128 x 128 -> 128 x 128 x 1
    resized = np.expand_dims(
        resized,
        axis=-1
    )

    # Add batch dimension
    # 128 x 128 x 1 -> 1 x 128 x 128 x 1
    input_image = np.expand_dims(
        resized,
        axis=0
    )

    # Prediction
    prediction = drowsiness_model.predict(
        input_image,
        verbose=0
    )

    index = np.argmax(
        prediction[0]
    )

    confidence = (
        prediction[0][index] * 100
    )

    label = DROWSINESS_CLASSES[index]

    return label, confidence


# ============================================================
# AGE PREDICTION
# ============================================================

def predict_age(face):

    # OpenCV uses BGR.
    # Convert to RGB because the age model
    # was trained/tested using RGB images.
    rgb_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Resize to age model input size
    resized = cv2.resize(
        rgb_face,
        (160, 160)
    )

    # Convert to float
    resized = resized.astype(
        "float32"
    )

    # Normalize
    resized = resized / 255.0

    # Add batch dimension
    input_image = np.expand_dims(
        resized,
        axis=0
    )

    # Prediction
    prediction = age_model.predict(
        input_image,
        verbose=0
    )

    # Find highest probability class
    index = np.argmax(
        prediction[0]
    )

    confidence = (
        prediction[0][index] * 100
    )

    # Get age range
    age_range = AGE_CLASSES[index]

    return age_range, confidence


# ============================================================
# DETECT FACES
# ============================================================

def detect_faces(frame):

    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect multiple faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    sleeping_count = 0

    # --------------------------------------------------------
    # PROCESS EACH PERSON
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:
            continue

        # ====================================================
        # DROWSINESS
        # ====================================================

        try:

            drowsiness, drowsiness_conf = \
                predict_drowsiness(face)

        except Exception as e:

            print(
                "Drowsiness prediction error:",
                e
            )

            continue

        # ====================================================
        # AGE
        # ====================================================

        try:

            age_range, age_conf = \
                predict_age(face)

        except Exception as e:

            print(
                "Age prediction error:",
                e
            )

            age_range = "Unknown"
            age_conf = 0

        # ====================================================
        # BOX COLOR
        # ====================================================

        if drowsiness.lower() == "sleeping":

            sleeping_count += 1

            # RED BOX
            box_color = (0, 0, 255)

        else:

            # GREEN BOX
            box_color = (0, 255, 0)

        # ====================================================
        # FACE BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )

        # ====================================================
        # TEXT BACKGROUND
        # ====================================================

        text_start_y = y + h + 5

        # Keep text inside image
        if text_start_y + 55 > frame.shape[0]:

            text_start_y = max(
                y - 60,
                5
            )

        # ====================================================
        # DROWSINESS TEXT
        # ====================================================

        cv2.putText(
            frame,
            f"{drowsiness.upper()} "
            f"{drowsiness_conf:.1f}%",
            (x, text_start_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

        # ====================================================
        # AGE TEXT
        # ====================================================

        cv2.putText(
            frame,
            f"Age: {age_range}",
            (x, text_start_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

    # ========================================================
    # SUMMARY BOX
    # ========================================================

    total_people = len(faces)

    cv2.rectangle(
        frame,
        (10, 10),
        (300, 85),
        (0, 0, 0),
        -1
    )

    # People count
    cv2.putText(
        frame,
        f"People: {total_people}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Sleeping count
    cv2.putText(
        frame,
        f"Sleeping: {sleeping_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    return frame


# ============================================================
# SHOW IMAGE IN TKINTER
# ============================================================

def show_result(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_pil = Image.fromarray(
        image_rgb
    )

    image_pil.thumbnail(
        (750, 500)
    )

    photo = ImageTk.PhotoImage(
        image_pil
    )

    result_label.configure(
        image=photo
    )

    result_label.image = photo


# ============================================================
# UPLOAD IMAGE
# ============================================================

def detect_image():

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp"
            )
        ]
    )

    if not file_path:
        return

    image = cv2.imread(
        file_path
    )

    if image is None:

        messagebox.showerror(
            "Error",
            "Unable to open image."
        )

        return

    result = detect_faces(
        image
    )

    show_result(
        result
    )


# ============================================================
# UPLOAD VIDEO
# ============================================================

def detect_video():

    file_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[
            (
                "Video files",
                "*.mp4 *.avi *.mov *.mkv"
            )
        ]
    )

    if not file_path:
        return

    cap = cv2.VideoCapture(
        file_path
    )

    if not cap.isOpened():

        messagebox.showerror(
            "Error",
            "Unable to open video."
        )

        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detect_faces(
            frame
        )

        cv2.imshow(
            "Drowsiness + Age Detection",
            result
        )

        key = cv2.waitKey(1) & 0xFF

        # Press Q to stop
        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# LIVE CAMERA
# ============================================================

def detect_camera():

    cap = cv2.VideoCapture(
        0
    )

    if not cap.isOpened():

        messagebox.showerror(
            "Error",
            "Camera could not be opened."
        )

        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detect_faces(
            frame
        )

        cv2.imshow(
            "Live Drowsiness + Age Detection",
            result
        )

        key = cv2.waitKey(1) & 0xFF

        # Press Q to stop
        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Drowsiness and Age Detection"
)

root.geometry(
    "900x700"
)

root.configure(
    bg="#eeeeee"
)


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    root,
    text="DROWSINESS + AGE DETECTION",
    font=("Arial", 24, "bold"),
    bg="#eeeeee"
)

title.pack(
    pady=20
)


# ============================================================
# DESCRIPTION
# ============================================================

info = tk.Label(
    root,
    text="Multiple People | Sleeping Detection | Age Range",
    font=("Arial", 12),
    bg="#eeeeee"
)

info.pack()


# ============================================================
# RESULT AREA
# ============================================================

result_label = tk.Label(
    root,
    bg="#eeeeee"
)

result_label.pack(
    pady=20,
    expand=True
)


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#eeeeee"
)

button_frame.pack(
    pady=20
)


# ============================================================
# UPLOAD IMAGE BUTTON
# ============================================================

image_button = tk.Button(
    button_frame,
    text="Upload Image",
    command=detect_image,
    font=("Arial", 13, "bold"),
    padx=20,
    pady=10
)

image_button.grid(
    row=0,
    column=0,
    padx=10
)


# ============================================================
# UPLOAD VIDEO BUTTON
# ============================================================

video_button = tk.Button(
    button_frame,
    text="Upload Video",
    command=detect_video,
    font=("Arial", 13, "bold"),
    padx=20,
    pady=10
)

video_button.grid(
    row=0,
    column=1,
    padx=10
)


# ============================================================
# LIVE CAMERA BUTTON
# ============================================================

camera_button = tk.Button(
    button_frame,
    text="Live Camera",
    command=detect_camera,
    font=("Arial", 13, "bold"),
    padx=20,
    pady=10
)

camera_button.grid(
    row=0,
    column=2,
    padx=10
)


# ============================================================
# EXIT BUTTON
# ============================================================

exit_button = tk.Button(
    root,
    text="Exit",
    command=root.destroy,
    font=("Arial", 12),
    padx=30,
    pady=8
)

exit_button.pack(
    pady=15
)


# ============================================================
# START
# ============================================================

root.mainloop()