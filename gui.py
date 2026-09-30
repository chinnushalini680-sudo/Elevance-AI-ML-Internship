import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import json
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "m/car_colour_model_improved.keras"

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "blue",
    1: "other",
    2: "person"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

selected_image = None
display_image = None


# ============================================================
# PREDICT OBJECT
# ============================================================

def predict_object(crop):

    resized = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE)
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

    prediction = model.predict(
        arr,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_class]
    )

    return predicted_class, confidence


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image():

    global selected_image

    if selected_image is None:
        messagebox.showwarning(
            "No Image",
            "Please select an image first."
        )
        return

    image = cv2.imread(selected_image)

    if image is None:
        messagebox.showerror(
            "Error",
            "Unable to read image."
        )
        return


    # --------------------------------------------------------
    # Find matching annotation JSON
    # --------------------------------------------------------

    image_name = os.path.basename(
        selected_image
    )

    base_name = os.path.splitext(
        image_name
    )[0]

    label_path = os.path.join(
        "dataset",
        "labels",
        base_name + ".json"
    )


    if not os.path.exists(label_path):

        messagebox.showerror(
            "No Annotation",
            "This image does not have a matching annotation file."
        )

        return


    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    with open(label_path, "r") as f:
        data = json.load(f)

    annotations = data.get(
        "annotations",
        []
    )


    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    blue_count = 0
    other_count = 0
    person_count = 0


    # --------------------------------------------------------
    # Process every annotated object
    # --------------------------------------------------------

    for ann in annotations:

        x1 = int(ann["x1"])
        y1 = int(ann["y1"])
        x2 = int(ann["x2"])
        y2 = int(ann["y2"])


        # Keep coordinates inside image

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            image.shape[1],
            x2
        )

        y2 = min(
            image.shape[0],
            y2
        )


        crop = image[
            y1:y2,
            x1:x2
        ]


        if crop.size == 0:
            continue


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predicted_class, confidence = predict_object(
            crop
        )


        # ----------------------------------------------------
        # Select rectangle colour
        # ----------------------------------------------------

        if predicted_class == 0:

            # Blue car
            rectangle_color = (
                0,
                0,
                255
            )

            label = "BLUE CAR"

            blue_count += 1


        elif predicted_class == 1:

            # Other car
            rectangle_color = (
                255,
                0,
                0
            )

            label = "OTHER CAR"

            other_count += 1


        else:

            # Person
            rectangle_color = (
                0,
                255,
                0
            )

            label = "PERSON"

            person_count += 1


        # ----------------------------------------------------
        # Draw rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            rectangle_color,
            3
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        text = (
            f"{label} "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            image,
            text,
            (
                x1,
                max(
                    y1 - 10,
                    25
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            rectangle_color,
            2
        )


    # ========================================================
    # COUNTS
    # ========================================================

    total_cars = (
        blue_count +
        other_count
    )


    # ========================================================
    # ADD RESULT INFORMATION
    # ========================================================

    cv2.rectangle(
        image,
        (10, 10),
        (330, 175),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        image,
        f"TOTAL CARS: {total_cars}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    cv2.putText(
        image,
        f"BLUE CARS: {blue_count}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )


    cv2.putText(
        image,
        f"OTHER CARS: {other_count}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )


    cv2.putText(
        image,
        f"PEOPLE: {person_count}",
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    output_folder = "results"

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    output_path = os.path.join(
        output_folder,
        "result_" + image_name
    )


    cv2.imwrite(
        output_path,
        image
    )


    # ========================================================
    # SHOW RESULT
    # ========================================================

    show_result(image)

    status_label.config(
        text=(
            f"Cars: {total_cars}   |   "
            f"Blue: {blue_count}   |   "
            f"Other: {other_count}   |   "
            f"People: {person_count}"
        )
    )


# ============================================================
# SELECT IMAGE
# ============================================================

def select_image():

    global selected_image

    file_path = filedialog.askopenfilename(
        title="Select Traffic Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png"
            )
        ]
    )


    if not file_path:
        return


    selected_image = file_path


    # --------------------------------------------------------
    # Show original image
    # --------------------------------------------------------

    image = Image.open(
        file_path
    )

    image.thumbnail(
        (850, 500)
    )

    photo = ImageTk.PhotoImage(
        image
    )


    image_label.config(
        image=photo
    )

    image_label.image = photo


    status_label.config(
        text="Image selected. Click Detect."
    )


# ============================================================
# SHOW RESULT
# ============================================================

def show_result(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    pil_image.thumbnail(
        (850, 500)
    )

    photo = ImageTk.PhotoImage(
        pil_image
    )

    image_label.config(
        image=photo
    )

    image_label.image = photo


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Car Colour Detection System"
)

root.geometry(
    "1000x700"
)

root.configure(
    bg="#eeeeee"
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="CAR COLOUR DETECTION SYSTEM",
    font=(
        "Arial",
        22,
        "bold"
    ),
    bg="#eeeeee"
)

title_label.pack(
    pady=20
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

image_frame = tk.Frame(
    root,
    bg="white",
    width=900,
    height=500
)

image_frame.pack(
    padx=30,
    pady=10
)

image_frame.pack_propagate(
    False
)


image_label = tk.Label(
    image_frame,
    text="Select a traffic image",
    font=(
        "Arial",
        16
    ),
    bg="white"
)

image_label.pack(
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
    pady=15
)


select_button = tk.Button(
    button_frame,
    text="SELECT IMAGE",
    font=(
        "Arial",
        12,
        "bold"
    ),
    command=select_image,
    width=18,
    height=2
)

select_button.pack(
    side=tk.LEFT,
    padx=10
)


detect_button = tk.Button(
    button_frame,
    text="DETECT",
    font=(
        "Arial",
        12,
        "bold"
    ),
    command=process_image,
    width=18,
    height=2
)

detect_button.pack(
    side=tk.LEFT,
    padx=10
)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    root,
    text="Select an image to begin.",
    font=(
        "Arial",
        12
    ),
    bg="#eeeeee"
)

status_label.pack(
    pady=10
)


# ============================================================
# START GUI
# ============================================================

root.mainloop()