import cv2
import os
import tkinter as tk
from tkinter import simpledialog

SOURCE_DIR = "age_raw"
OUTPUT_DIR = "age_dataset"

AGE_RANGES = {
    "1": "0-10",
    "2": "11-19",
    "3": "20-30",
    "4": "31-40",
    "5": "41-50",
    "6": "51-60",
    "7": "61-70",
    "8": "71-80",
    "9": "81-90",
    "10": "91-100"
}

# Create folders
for folder in AGE_RANGES.values():
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not files:
    print("No images found in age_raw!")
    exit()

# Tkinter popup
root = tk.Tk()
root.withdraw()

# Mouse variables
drawing = False
start_x = 0
start_y = 0
current_box = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x = x
        start_y = y
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box = (
            start_x,
            start_y,
            x,
            y
        )

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box = (
            start_x,
            start_y,
            x,
            y
        )


for image_number, filename in enumerate(files, 1):

    image_path = os.path.join(SOURCE_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        continue

    result = image.copy()
    boxes = []

    window_name = "AGE ANNOTATOR"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n===================================")
    print(f"IMAGE {image_number}/{len(files)}")
    print(filename)
    print("===================================")

    while True:

        display = result.copy()

        # Draw current box while dragging
        if current_box is not None and drawing:

            x1, y1, x2, y2 = current_box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # --------------------------------
        # ENTER = accept drawn box
        # --------------------------------

        if key == 13:

            if current_box is None:
                continue

            x1, y1, x2, y2 = current_box

            # Fix coordinates
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)

            width = right - left
            height = bottom - top

            if width < 10 or height < 10:
                print("Box too small.")
                current_box = None
                continue

            # Ask age
            age = simpledialog.askstring(
                "Age Range",
                "Enter age range:\n\n"
                "1 = 0-10\n"
                "2 = 11-19\n"
                "3 = 20-30\n"
                "4 = 31-40\n"
                "5 = 41-50\n"
                "6 = 51-60\n"
                "7 = 61-70\n"
                "8 = 71-80\n"
                "9 = 81-90\n"
                "10 = 91-100\n\n"
                "Enter number:",
                parent=root
            )

            if age is None:
                current_box = None
                continue

            age = age.strip()

            if age not in AGE_RANGES:
                print("Invalid age range.")
                current_box = None
                continue

            age_folder = AGE_RANGES[age]

            # Crop face/person
            crop = image[top:bottom, left:right]

            # Unique filename
            count = len(
                os.listdir(
                    os.path.join(
                        OUTPUT_DIR,
                        age_folder
                    )
                )
            )

            crop_name = (
                os.path.splitext(filename)[0]
                + "_"
                + str(count + 1)
                + ".jpg"
            )

            crop_path = os.path.join(
                OUTPUT_DIR,
                age_folder,
                crop_name
            )

            cv2.imwrite(crop_path, crop)

            # Save box
            boxes.append(
                (left, top, right, bottom, age_folder)
            )

            # Draw permanent box
            cv2.rectangle(
                result,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                result,
                "Age: " + age_folder,
                (left, max(top - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print(
                "Saved:",
                crop_path
            )

            current_box = None

        # --------------------------------
        # S = SAVE CURRENT ANNOTATED IMAGE
        # --------------------------------

        elif key == ord("s"):

            save_path = os.path.join(
                OUTPUT_DIR,
                "annotated_" + filename
            )

            cv2.imwrite(
                save_path,
                result
            )

            print("\nANNOTATED IMAGE SAVED:")
            print(save_path)

        # --------------------------------
        # N = NEXT IMAGE
        # --------------------------------

        elif key == ord("n"):

            print("Next image...")
            break

        # --------------------------------
        # Q = QUIT
        # --------------------------------

        elif key == ord("q"):

            cv2.destroyAllWindows()
            root.destroy()
            print("Stopped.")
            exit()

    cv2.destroyAllWindows()

root.destroy()

print("\n===================================")
print("ANNOTATION COMPLETE")
print("===================================")
print("Dataset:")
print(os.path.abspath(OUTPUT_DIR))