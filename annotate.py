import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "face_dataset")

AWAKE_DIR = os.path.join(DATASET_DIR, "awake")
SLEEP_DIR = os.path.join(DATASET_DIR, "sleep")

SAVE_AWAKE_DIR = os.path.join(OUTPUT_DIR, "awake")
SAVE_SLEEP_DIR = os.path.join(OUTPUT_DIR, "sleep")

os.makedirs(SAVE_AWAKE_DIR, exist_ok=True)
os.makedirs(SAVE_SLEEP_DIR, exist_ok=True)


def annotate_folder(input_dir, output_dir, label):

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print()
    print("================================")
    print("ANNOTATING:", label.upper())
    print("================================")
    print("Images:", len(files))
    print()
    print("Drag around each FACE.")
    print("ENTER = save all boxes")
    print("U = undo last box")
    print("R = remove all boxes")
    print("N = skip image")
    print("Q = quit")
    print()

    total_saved = 0

    for index, filename in enumerate(files):

        path = os.path.join(input_dir, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        boxes = []
        drawing = False
        start_x = 0
        start_y = 0

        window_name = "Drowsiness Annotation"

        display = image.copy()

        def mouse(event, x, y, flags, param):

            nonlocal drawing
            nonlocal start_x
            nonlocal start_y
            nonlocal display

            if event == cv2.EVENT_LBUTTONDOWN:

                drawing = True
                start_x = x
                start_y = y

            elif event == cv2.EVENT_MOUSEMOVE:

                if drawing:

                    display = image.copy()

                    for box in boxes:

                        x1, y1, x2, y2 = box

                        cv2.rectangle(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                    cv2.rectangle(
                        display,
                        (start_x, start_y),
                        (x, y),
                        (0, 255, 255),
                        2
                    )

            elif event == cv2.EVENT_LBUTTONUP:

                drawing = False

                x1 = min(start_x, x)
                y1 = min(start_y, y)

                x2 = max(start_x, x)
                y2 = max(start_y, y)

                if x2 - x1 >= 20 and y2 - y1 >= 20:

                    boxes.append(
                        (x1, y1, x2, y2)
                    )

                    print(
                        "Box added:",
                        len(boxes)
                    )

                display = image.copy()

                for box in boxes:

                    x1, y1, x2, y2 = box

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse)

        print(
            f"[{index + 1}/{len(files)}] {filename}"
        )

        while True:

            screen = display.copy()

            cv2.putText(
                screen,
                f"Faces: {len(boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                screen,
                "ENTER=SAVE  U=UNDO  R=RESET  N=SKIP  Q=QUIT",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.imshow(window_name, screen)

            key = cv2.waitKey(1) & 0xFF

            # ENTER
            if key == 13:

                if len(boxes) == 0:
                    print("No boxes drawn.")
                    continue

                for box in boxes:

                    x1, y1, x2, y2 = box

                    face = image[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    total_saved += 1

                    save_name = (
                        f"{label}_{total_saved:04d}.jpg"
                    )

                    save_path = os.path.join(
                        output_dir,
                        save_name
                    )

                    cv2.imwrite(
                        save_path,
                        face
                    )

                    print("SAVED:", save_name)

                cv2.destroyWindow(window_name)

                break

            # UNDO
            elif key == ord("u"):

                if boxes:

                    boxes.pop()

                    display = image.copy()

                    for box in boxes:

                        x1, y1, x2, y2 = box

                        cv2.rectangle(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                    print("Last box removed.")

            # RESET
            elif key == ord("r"):

                boxes.clear()

                display = image.copy()

                print("All boxes removed.")

            # SKIP
            elif key == ord("n"):

                print("Skipped:", filename)

                cv2.destroyWindow(window_name)

                break

            # QUIT
            elif key == ord("q"):

                cv2.destroyAllWindows()

                print("Annotation stopped.")

                return

    cv2.destroyAllWindows()

    print()
    print("Completed:", label.upper())
    print("Faces saved:", total_saved)


# ==========================================
# AWAKE
# ==========================================

annotate_folder(
    AWAKE_DIR,
    SAVE_AWAKE_DIR,
    "awake"
)


# ==========================================
# SLEEP
# ==========================================

annotate_folder(
    SLEEP_DIR,
    SAVE_SLEEP_DIR,
    "sleep"
)


print()
print("================================")
print("ANNOTATION COMPLETE")
print("================================")