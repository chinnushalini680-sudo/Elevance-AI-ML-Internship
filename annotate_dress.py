import cv2
import os
import csv

# ==========================================
# SETTINGS
# ==========================================

DATASET_DIR = "images"
OUTPUT_FILE = "dress_labels.csv"

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

# ==========================================
# 20 COLOURS
# ==========================================

COLOURS = [
    "Red",
    "Pink",
    "Blue",
    "Yellow",
    "Purple",
    "Violet",
    "Brown",
    "Grey",
    "Beige",
    "Silver",
    "Gold",
    "Black",
    "White",
    "Green",
    "Orange",
    "Maroon",
    "Mauve",
    "Cream",
    "Multicolour",
    "Lavender"
]

# ==========================================
# KEY MAPPING
# ==========================================

KEY_TO_COLOUR = {
    ord("1"): "Red",
    ord("2"): "Pink",
    ord("3"): "Blue",
    ord("4"): "Yellow",
    ord("5"): "Purple",
    ord("6"): "Violet",
    ord("7"): "Brown",
    ord("8"): "Grey",
    ord("9"): "Beige",
    ord("0"): "Silver",

    ord("a"): "Gold",
    ord("A"): "Gold",

    ord("b"): "Black",
    ord("B"): "Black",

    ord("c"): "White",
    ord("C"): "White",

    ord("d"): "Green",
    ord("D"): "Green",

    ord("e"): "Orange",
    ord("E"): "Orange",

    ord("f"): "Maroon",
    ord("F"): "Maroon",

    ord("g"): "Mauve",
    ord("G"): "Mauve",

    ord("h"): "Cream",
    ord("H"): "Cream",

    ord("i"): "Multicolour",
    ord("I"): "Multicolour",

    ord("j"): "Lavender",
    ord("J"): "Lavender"
}

# ==========================================
# LOAD ALL IMAGES
# ==========================================

image_list = []

for root, dirs, files in os.walk(DATASET_DIR):

    for file in files:

        if file.lower().endswith(VALID_EXTENSIONS):

            full_path = os.path.join(root, file)

            category = os.path.basename(root)

            image_list.append(
                (full_path, category)
            )

# Sort images
image_list.sort()

print()
print("======================================")
print("       DRESS COLOUR ANNOTATOR")
print("======================================")

print("Total images:", len(image_list))

print()
print("COLOUR KEYS")
print("--------------------------------------")

print("1 = Red")
print("2 = Pink")
print("3 = Blue")
print("4 = Yellow")
print("5 = Purple")
print("6 = Violet")
print("7 = Brown")
print("8 = Grey")
print("9 = Beige")
print("0 = Silver")

print("A = Gold")
print("B = Black")
print("C = White")
print("D = Green")
print("E = Orange")
print("F = Maroon")
print("G = Mauve")
print("H = Cream")
print("I = Multicolour")
print("J = Lavender")

print()
print("S = Skip")
print("Q = Quit")

print("======================================")

# ==========================================
# LOAD EXISTING LABELS
# ==========================================

labels = {}

if os.path.exists(OUTPUT_FILE):

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            labels[row["image"]] = row["dress_colour"]

# ==========================================
# START ANNOTATION
# ==========================================

index = 0

while index < len(image_list):

    image_path, category = image_list[index]

    # Skip already labelled images
    if image_path in labels:

        index += 1
        continue

    image = cv2.imread(image_path)

    if image is None:

        print(
            "Could not open:",
            image_path
        )

        index += 1
        continue

    display = image.copy()

    # ======================================
    # RESIZE IMAGE
    # ======================================

    height, width = display.shape[:2]

    max_width = 1000
    max_height = 700

    scale = min(
        max_width / width,
        max_height / height,
        1
    )

    if scale < 1:

        display = cv2.resize(
            display,
            (
                int(width * scale),
                int(height * scale)
            )
        )

    # ======================================
    # DISPLAY INFORMATION
    # ======================================

    cv2.putText(
        display,
        f"Category: {category}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Image {index + 1} / {len(image_list)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Dress Colour Annotation",
        display
    )

    key = cv2.waitKey(0)

    # ======================================
    # QUIT
    # ======================================

    if key == ord("q") or key == ord("Q"):

        print()
        print("Annotation stopped.")
        break

    # ======================================
    # SKIP
    # ======================================

    if key == ord("s") or key == ord("S"):

        print(
            "SKIPPED:",
            os.path.basename(image_path)
        )

        index += 1
        continue

    # ======================================
    # SAVE COLOUR
    # ======================================

    if key in KEY_TO_COLOUR:

        colour = KEY_TO_COLOUR[key]

        labels[image_path] = colour

        print(
            "SAVED:",
            os.path.basename(image_path),
            "->",
            colour
        )

        index += 1

    else:

        print()
        print("Invalid key.")
        print("Please press a valid colour key.")

# ==========================================
# CLOSE WINDOW
# ==========================================

cv2.destroyAllWindows()

# ==========================================
# SAVE CSV
# ==========================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "image",
            "category",
            "dress_colour"
        ]
    )

    for image_path, category in image_list:

        if image_path in labels:

            writer.writerow(
                [
                    image_path,
                    category,
                    labels[image_path]
                ]
            )

# ==========================================
# COMPLETE
# ==========================================

print()
print("======================================")
print("       ANNOTATION COMPLETE")
print("======================================")

print("Labels saved to:")
print(OUTPUT_FILE)

print("Total labelled:", len(labels))

print("======================================")