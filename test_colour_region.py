import cv2
import numpy as np

# ==============================
# COLOUR DEFINITIONS
# ==============================

COLOURS = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (255, 0, 0),
    "Orange": (255, 165, 0),
    "Yellow": (255, 255, 0),
    "Green": (0, 128, 0),
    "Blue": (0, 0, 255),
    "Purple": (128, 0, 128),
    "Pink": (255, 105, 180),
    "Brown": (139, 69, 19),
    "Grey": (128, 128, 128)
}


def get_dress_region(image):
    """
    Take the central/lower portion of the image.
    This is a simple approximation of the clothing region.
    """

    h, w = image.shape[:2]

    # Ignore the upper part where the face is usually located
    y1 = int(h * 0.35)
    y2 = int(h * 0.90)

    x1 = int(w * 0.20)
    x2 = int(w * 0.80)

    return image[y1:y2, x1:x2]


def predict_colour(region):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Ignore extremely dark and extremely bright pixels
    # because they can be background/shadows.
    pixels = hsv.reshape(-1, 3)

    valid = []

    for h, s, v in pixels:

        # Remove very dark pixels
        if v < 30:
            continue

        # Remove very low saturation pixels
        # only when they are not clearly white/grey.
        valid.append((h, s, v))

    if len(valid) == 0:
        return "Unknown"

    valid = np.array(valid)

    h = valid[:, 0]
    s = valid[:, 1]
    v = valid[:, 2]

    # OpenCV hue range = 0-179

    # Black
    black = v < 60

    # White
    white = (v > 180) & (s < 50)

    # Grey
    grey = (s < 50) & (v >= 60) & (v <= 180)

    # Strong colour pixels
    coloured = s > 70

    if np.sum(black) > len(valid) * 0.35:
        return "Black"

    if np.sum(white) > len(valid) * 0.35:
        return "White"

    if np.sum(grey) > len(valid) * 0.35:
        return "Grey"

    if np.sum(coloured) == 0:
        return "Unknown"

    colour_hues = h[coloured]

    # Hue ranges
    red = (
        (colour_hues < 10) |
        (colour_hues > 170)
    )

    orange = (
        (colour_hues >= 10) &
        (colour_hues < 22)
    )

    yellow = (
        (colour_hues >= 22) &
        (colour_hues < 35)
    )

    green = (
        (colour_hues >= 35) &
        (colour_hues < 85)
    )

    blue = (
        (colour_hues >= 85) &
        (colour_hues < 130)
    )

    purple = (
        (colour_hues >= 130) &
        (colour_hues < 155)
    )

    pink = (
        (colour_hues >= 155) &
        (colour_hues <= 170)
    )

    counts = {
        "Red": np.sum(red),
        "Orange": np.sum(orange),
        "Yellow": np.sum(yellow),
        "Green": np.sum(green),
        "Blue": np.sum(blue),
        "Purple": np.sum(purple),
        "Pink": np.sum(pink)
    }

    # Brown detection
    brown = (
        (colour_hues >= 5) &
        (colour_hues < 25) &
        (s > 60) &
        (v < 170)
    )

    counts["Brown"] = np.sum(brown)

    result = max(
        counts,
        key=counts.get
    )

    return result


# ==============================
# MAIN
# ==============================

image_path = input(
    "Enter image path: "
).strip()

image = cv2.imread(image_path)

if image is None:
    print("Could not read image.")
    exit()

region = get_dress_region(image)

result = predict_colour(region)

print("\n==============================")
print("DRESS COLOUR RESULT")
print("==============================")
print("Predicted colour:", result)
print("==============================")


# Show the region being analysed
cv2.imshow(
    "Dress Region",
    region
)

cv2.waitKey(0)
cv2.destroyAllWindows()