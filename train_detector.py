import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

DATASET = "dataset"
IMG_SIZE = 160

classes = ["cat", "cow", "dog", "elephant", "lion"]

images = []
labels = []

print("Loading animal crops...")

for class_id, animal in enumerate(classes):

    folder = os.path.join(DATASET, animal)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(folder, filename)
        txt_path = os.path.splitext(image_path)[0] + ".txt"

        if not os.path.exists(txt_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        h, w = image.shape[:2]

        with open(txt_path, "r") as f:
            line = f.readline().strip()

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            x1 = int(parts[1])
            y1 = int(parts[2])
            x2 = int(parts[3])
            y2 = int(parts[4])
        except:
            continue

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(
            crop,
            (IMG_SIZE, IMG_SIZE)
        )

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        images.append(crop)
        labels.append(class_id)

        # Add extra augmented copies for ELEPHANTS
        if animal == "elephant":

            # Horizontal flip
            flipped = cv2.flip(crop, 1)

            images.append(flipped)
            labels.append(class_id)

            # Brightness variation
            brighter = cv2.convertScaleAbs(
                crop,
                alpha=1.15,
                beta=15
            )

            images.append(brighter)
            labels.append(class_id)

            # Slightly darker
            darker = cv2.convertScaleAbs(
                crop,
                alpha=0.85,
                beta=-10
            )

            images.append(darker)
            labels.append(class_id)


X = np.array(images, dtype=np.float32)
Y = np.array(labels, dtype=np.int32)

print()
print("==============================")
print("DATASET")
print("==============================")
print("Total crops:", len(X))

for i, name in enumerate(classes):
    print(
        name + ":",
        np.sum(Y == i)
    )

print("==============================")


# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("Training:", len(X_train))
print("Testing:", len(X_test))


# MobileNet preprocessing
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)


# ------------------------------------------------
# MOBILE NET V2
# ------------------------------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False


inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    5,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print()
print("==============================")
print("TRAINING STARTED")
print("==============================")


model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_test,
        Y_test
    ),
    epochs=35,
    batch_size=8,
    shuffle=True
)


loss, accuracy = model.evaluate(
    X_test,
    Y_test,
    verbose=0
)


print()
print("==============================")
print("TRAINING COMPLETED")
print("==============================")
print(
    "Test Accuracy:",
    f"{accuracy * 100:.2f}%"
)
print("==============================")


model.save(
    "animal_crop_classifier.keras"
)


with open(
    "animal_classes.txt",
    "w"
) as f:

    for animal in classes:
        f.write(animal + "\n")


print()
print("Saved:")
print("animal_crop_classifier.keras")
print("animal_classes.txt")