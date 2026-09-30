import cv2
import os
import glob

# ==========================================
# SETTINGS
# ==========================================

FOLDER = "multiple_animals"

ANIMALS = {
    ord("1"): "cat",
    ord("2"): "cow",
    ord("3"): "dog",
    ord("4"): "elephant",
    ord("5"): "lion"
}

CARNIVORES = {
    "cat",
    "dog",
    "lion"
}

# ==========================================
# FIND IMAGES
# ==========================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

image_files = []

for ext in extensions:
    image_files.extend(
        glob.glob(os.path.join(FOLDER, ext))
    )

image_files = sorted(list(set(image_files)))

if len(image_files) == 0:
    print("No images found!")
    print("Check the multiple_animals folder.")
    exit()

print()
print("==============================")
print("MULTIPLE ANIMAL ANNOTATOR")
print("==============================")
print("Images found:", len(image_files))
print("==============================")


# ==========================================
# VARIABLES
# ==========================================

current_index = 0

boxes = []

drawing = False

start_x = 0
start_y = 0

temp_box = None
waiting_for_label = False

original = None
image = None


# ==========================================
# REDRAW IMAGE
# ==========================================

def redraw():

    global image

    image = original.copy()

    # --------------------------------------
    # Draw saved boxes
    # --------------------------------------

    for animal, x1, y1, x2, y2 in boxes:

        if animal in CARNIVORES:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            image,
            animal.upper(),
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # --------------------------------------
    # Temporary box
    # --------------------------------------

    if temp_box is not None:

        x1, y1, x2, y2 = temp_box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

    # --------------------------------------
    # Image number
    # --------------------------------------

    cv2.putText(
        image,
        f"Image {current_index + 1}/{len(image_files)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------
    # Instructions
    # --------------------------------------

    cv2.putText(
        image,
        "1 Cat  2 Cow  3 Dog  4 Elephant  5 Lion",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        "S Save | N Next | U Undo | R Reset | Q Quit",
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    if waiting_for_label:

        cv2.putText(
            image,
            "PRESS 1-5 TO SELECT ANIMAL",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "Multiple Animal Annotator",
        image
    )


# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(index):

    global original
    global image
    global boxes
    global temp_box
    global waiting_for_label

    original = cv2.imread(
        image_files[index]
    )

    if original is None:

        print(
            "Could not open:",
            image_files[index]
        )

        return False

    boxes = []

    temp_box = None

    waiting_for_label = False

    print()
    print("==============================")
    print(
        f"IMAGE {index + 1} OF {len(image_files)}"
    )
    print(
        os.path.basename(
            image_files[index]
        )
    )
    print("==============================")

    print("Draw a box around an animal.")
    print("Then press 1-5.")
    print()
    print("1 = Cat")
    print("2 = Cow")
    print("3 = Dog")
    print("4 = Elephant")
    print("5 = Lion")
    print()
    print("S = Save")
    print("N = Next")
    print("U = Undo")
    print("R = Reset")
    print("Q = Quit")

    redraw()

    return True


# ==========================================
# SAVE CURRENT IMAGE
# ==========================================

def save_current():

    if len(boxes) == 0:

        print()
        print("No boxes to save!")

        return False

    image_path = image_files[
        current_index
    ]

    txt_path = (
        os.path.splitext(image_path)[0]
        + ".txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as file:

        for animal, x1, y1, x2, y2 in boxes:

            file.write(
                f"{animal} "
                f"{x1} {y1} "
                f"{x2} {y2}\n"
            )

    print()
    print("==============================")
    print("SAVED SUCCESSFULLY!")
    print("==============================")

    print(
        "File:",
        txt_path
    )

    print(
        "Animals:",
        len(boxes)
    )

    for box in boxes:
        print(box)

    print("==============================")

    return True


# ==========================================
# MOUSE CALLBACK
# ==========================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global drawing
    global start_x
    global start_y
    global temp_box
    global waiting_for_label

    # --------------------------------------
    # START BOX
    # --------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        # Don't start another box
        # while choosing animal label

        if waiting_for_label:
            return

        drawing = True

        start_x = x
        start_y = y

        temp_box = (
            x,
            y,
            x,
            y
        )

        redraw()

    # --------------------------------------
    # DRAW BOX
    # --------------------------------------

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            temp_box = (
                min(start_x, x),
                min(start_y, y),
                max(start_x, x),
                max(start_y, y)
            )

            redraw()

    # --------------------------------------
    # FINISH BOX
    # --------------------------------------

    elif event == cv2.EVENT_LBUTTONUP:

        if not drawing:
            return

        drawing = False

        x1 = min(
            start_x,
            x
        )

        y1 = min(
            start_y,
            y
        )

        x2 = max(
            start_x,
            x
        )

        y2 = max(
            start_y,
            y
        )

        # Check box size

        if x2 - x1 < 10 or y2 - y1 < 10:

            print(
                "Box too small. Draw again."
            )

            temp_box = None

            redraw()

            return

        # Store temporary box

        temp_box = (
            x1,
            y1,
            x2,
            y2
        )

        waiting_for_label = True

        print()
        print("==============================")
        print("BOX CREATED")
        print("==============================")
        print("Press:")
        print("1 = Cat")
        print("2 = Cow")
        print("3 = Dog")
        print("4 = Elephant")
        print("5 = Lion")

        redraw()


# ==========================================
# CREATE WINDOW
# ==========================================

cv2.namedWindow(
    "Multiple Animal Annotator",
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    "Multiple Animal Annotator",
    mouse_callback
)


# ==========================================
# LOAD FIRST IMAGE
# ==========================================

if not load_image(current_index):
    exit()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    key = cv2.waitKey(30) & 0xFF

    # Ignore no key

    if key == 255:
        continue

    # ======================================
    # ANIMAL SELECTION
    # ======================================

    if waiting_for_label:

        if key in ANIMALS:

            animal = ANIMALS[key]

            x1, y1, x2, y2 = temp_box

            boxes.append(
                (
                    animal,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            print(
                f"Added {animal}: "
                f"{x1} {y1} {x2} {y2}"
            )

            temp_box = None

            waiting_for_label = False

            redraw()

        # Don't process S/N while
        # waiting for animal selection

        continue

    # ======================================
    # SAVE
    # ======================================

    if key == ord("s"):

        save_current()

    # ======================================
    # NEXT
    # ======================================

    elif key == ord("n"):

        if len(boxes) > 0:

            save_current()

        if current_index < len(image_files) - 1:

            current_index += 1

            load_image(
                current_index
            )

        else:

            print()
            print("==============================")
            print("ALL IMAGES COMPLETED!")
            print("==============================")
            print("Press Q to quit.")

    # ======================================
    # UNDO
    # ======================================

    elif key == ord("u"):

        if len(boxes) > 0:

            removed = boxes.pop()

            print(
                "Removed:",
                removed
            )

            redraw()

        else:

            print(
                "Nothing to undo."
            )

    # ======================================
    # RESET
    # ======================================

    elif key == ord("r"):

        boxes.clear()

        temp_box = None

        waiting_for_label = False

        redraw()

        print(
            "All boxes removed."
        )

    # ======================================
    # QUIT
    # ======================================

    elif key == ord("q"):

        print(
            "Closing annotator..."
        )

        break


cv2.destroyAllWindows()