import cv2
import os

DATASET_PATH = "dataset"

signs = {
    "1": "hello",
    "2": "thank_you",
    "3": "yes",
    "4": "no",
    "5": "help"
}

# Create folders
for sign in signs.values():
    os.makedirs(os.path.join(DATASET_PATH, sign), exist_ok=True)

print("====================================")
print("     SIGN LANGUAGE DATA COLLECTION")
print("====================================")
print("1 - Hello")
print("2 - Thank You")
print("3 - Yes")
print("4 - No")
print("5 - Help")
print("Q - Quit")
print("====================================")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "1:Hello  2:ThankYou  3:Yes  4:No  5:Help",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Sign Language Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if chr(key) in signs:
        sign_name = signs[chr(key)]
        folder = os.path.join(DATASET_PATH, sign_name)

        existing_images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        image_number = len(existing_images) + 1

        filename = os.path.join(
            folder,
            f"{sign_name}_{image_number}.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

camera.release()
cv2.destroyAllWindows()

print("Data collection finished.")