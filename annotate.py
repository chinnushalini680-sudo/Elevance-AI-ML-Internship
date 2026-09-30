import cv2
import os
import json

# ==========================================
# SETTINGS
# ==========================================

IMAGE_FOLDER = "dataset/images"
LABEL_FOLDER = "dataset/labels"

os.makedirs(LABEL_FOLDER, exist_ok=True)

CLASS_NAMES = {
    0: "blue",
    1: "other",
    2: "person"
}

# ==========================================
# GLOBAL VARIABLES
# ==========================================

image = None
display_image = None

boxes = []
current_class = None

drawing = False
start_x = 0
start_y = 0


# ==========================================
# DRAW EXISTING BOXES
# ==========================================

def draw_boxes():

    global display_image

    display_image = image.copy()

    for box in boxes:

        class_id = box["class_id"]
        class_name = box["class_name"]

        # Blue car = RED rectangle
        if class_id == 0:
            color = (0, 0, 255)

        # Other car = BLUE rectangle
        elif class_id == 1:
            color = (255, 0, 0)

        # Person = GREEN rectangle
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            display_image,
            (box["x1"], box["y1"]),
            (box["x2"], box["y2"]),
            color,
            2
        )

        cv2.putText(
            display_image,
            class_name,
            (
                box["x1"],
                max(20, box["y1"] - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )


# ==========================================
# MOUSE FUNCTION
# ==========================================

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x, start_y
    global display_image
    global boxes
    global current_class

    if event == cv2.EVENT_LBUTTONDOWN:

        if current_class is None:

            print()
            print("Select a class first:")
            print("1 = Blue car")
            print("2 = Other car")
            print("3 = Person")
            return

        drawing = True

        start_x = x
        start_y = y

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            draw_boxes()

            cv2.rectangle(
                display_image,
                (start_x, start_y),
                (x, y),
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Car Colour Annotation",
                display_image
            )

    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False

            end_x = x
            end_y = y

            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)

            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            # Ignore tiny boxes
            if (x2 - x1) > 10 and (y2 - y1) > 10:

                new_box = {
                    "class_id": current_class,
                    "class_name": CLASS_NAMES[current_class],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }

                boxes.append(new_box)

                print(
                    "Added:",
                    CLASS_NAMES[current_class],
                    "=>",
                    (x1, y1),
                    (x2, y2)
                )

            draw_boxes()

            cv2.imshow(
                "Car Colour Annotation",
                display_image
            )


# ==========================================
# LOAD EXISTING ANNOTATIONS
# ==========================================

def load_annotations(filename):

    label_filename = os.path.splitext(filename)[0] + ".json"

    label_path = os.path.join(
        LABEL_FOLDER,
        label_filename
    )

    if os.path.exists(label_path):

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            print(
                "Existing annotations loaded:",
                len(data.get("annotations", []))
            )

            return data.get("annotations", [])

        except Exception as e:

            print(
                "Could not read annotation:",
                e
            )

    return []


# ==========================================
# SAVE ANNOTATIONS
# ==========================================

def save_annotations(filename):

    label_filename = os.path.splitext(filename)[0] + ".json"

    label_path = os.path.join(
        LABEL_FOLDER,
        label_filename
    )

    data = {
        "image": filename,
        "classes": {
            "0": "blue",
            "1": "other",
            "2": "person"
        },
        "annotations": boxes
    }

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    print(
        "Saved:",
        label_filename
    )

    print(
        "Total boxes:",
        len(boxes)
    )


# ==========================================
# GET IMAGES
# ==========================================

image_files = [
    f
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

image_files.sort()

if len(image_files) == 0:

    print(
        "No images found in:",
        IMAGE_FOLDER
    )

    exit()


# ==========================================
# INFORMATION
# ==========================================

print()
print("==========================================")
print("       CAR COLOUR ANNOTATION TOOL")
print("==========================================")
print()
print("TOTAL IMAGES:", len(image_files))
print()
print("1 = BLUE CAR")
print("2 = OTHER COLOUR CAR")
print("3 = PERSON")
print()
print("LEFT MOUSE + DRAG = Draw box")
print()
print("U = Undo last box")
print("S = Save")
print("N = Next image")
print("Q = Quit")
print()
print("Existing annotations will be loaded.")
print("==========================================")


# ==========================================
# PROCESS IMAGES
# ==========================================

for image_index, filename in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:

        print(
            "Could not open:",
            filename
        )

        continue

    # IMPORTANT:
    # Load previous annotations
    boxes = load_annotations(filename)

    current_class = None

    draw_boxes()

    cv2.namedWindow(
        "Car Colour Annotation"
    )

    cv2.setMouseCallback(
        "Car Colour Annotation",
        mouse_callback
    )

    print()
    print("------------------------------------------")
    print(
        f"IMAGE {image_index + 1}/{len(image_files)}"
    )
    print(
        "FILE:",
        filename
    )
    print(
        "Existing boxes:",
        len(boxes)
    )
    print("------------------------------------------")

    while True:

        draw_boxes()

        # Instructions
        cv2.putText(
            display_image,
            "1 Blue | 2 Other | 3 Person",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_image,
            "U Undo | S Save | N Next | Q Quit",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Car Colour Annotation",
            display_image
        )

        key = cv2.waitKey(50) & 0xFF

        # ==================================
        # BLUE CAR
        # ==================================

        if key == ord("1"):

            current_class = 0

            print(
                "Selected: BLUE CAR"
            )

        # ==================================
        # OTHER CAR
        # ==================================

        elif key == ord("2"):

            current_class = 1

            print(
                "Selected: OTHER COLOUR CAR"
            )

        # ==================================
        # PERSON
        # ==================================

        elif key == ord("3"):

            current_class = 2

            print(
                "Selected: PERSON"
            )

        # ==================================
        # UNDO
        # ==================================

        elif key == ord("u"):

            if len(boxes) > 0:

                removed = boxes.pop()

                print(
                    "Removed:",
                    removed["class_name"]
                )

            else:

                print(
                    "Nothing to undo."
                )

        # ==================================
        # SAVE
        # ==================================

        elif key == ord("s"):

            save_annotations(filename)

        # ==================================
        # NEXT IMAGE
        # ==================================

        elif key == ord("n"):

            save_annotations(filename)

            print(
                "Moving to next image..."
            )

            break

        # ==================================
        # QUIT
        # ==================================

        elif key == ord("q"):

            save_annotations(filename)

            cv2.destroyAllWindows()

            print()
            print(
                "Annotation stopped."
            )

            print(
                "Your saved annotations are safe."
            )

            exit()


# ==========================================
# FINISHED
# ==========================================

cv2.destroyAllWindows()

print()
print("==========================================")
print("       ALL IMAGES COMPLETED")
print("==========================================")
print()
print(
    "Images:",
    len(image_files)
)
print(
    "Annotations saved in:",
    LABEL_FOLDER
)
