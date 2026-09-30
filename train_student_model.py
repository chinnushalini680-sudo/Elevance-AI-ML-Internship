import cv2
import os
import numpy as np

# Path to student dataset
dataset_path = "student_dataset"

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Store training data
faces = []
labels = []

# Student names and numeric labels
student_names = {}
label_id = 0

# Read each student's folder
for student_name in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.isdir(student_folder):
        continue

    student_names[label_id] = student_name

    print(f"Processing: {student_name}")

    for image_name in os.listdir(student_folder):

        image_path = os.path.join(student_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            faces.append(face)
            labels.append(label_id)

    label_id += 1


# Check whether faces were found
if len(faces) == 0:
    print("No faces found in the dataset.")
    exit()

print()
print("Total face images used:", len(faces))
print("Students:", student_names)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train model
recognizer.train(faces, np.array(labels))

# Save model
recognizer.write("student_face_model.yml")

# Save student names
with open("student_names.txt", "w") as file:
    for label, name in student_names.items():
        file.write(f"{label},{name}\n")

print()
print("Student face training completed successfully!")
print("Model saved as student_face_model.yml")
print("Student names saved as student_names.txt")