import os
import json
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# SETTINGS
# ============================================================

IMAGE_FOLDER = "dataset/images"
LABEL_FOLDER = "dataset/labels"

MODEL_FOLDER = "m"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30

CLASS_NAMES = {
    0: "blue",
    1: "other",
    2: "person"
}


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# LOAD OBJECT CROPS FROM ANNOTATIONS
# ============================================================

X = []
y = []

print()
print("==========================================")
print("LOADING ANNOTATED OBJECTS")
print("==========================================")

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for image_name in image_files:

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    label_name = os.path.splitext(image_name)[0] + ".json"
    label_path = os.path.join(LABEL_FOLDER, label_name)

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(image_path)

    if image is None:
        continue

    with open(label_path, "r") as file:
        data = json.load(file)

    for annotation in data.get("annotations", []):

        class_id = int(annotation["class_id"])

        if class_id not in CLASS_NAMES:
            continue

        x1 = int(annotation["x1"])
        y1 = int(annotation["y1"])
        x2 = int(annotation["x2"])
        y2 = int(annotation["y2"])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(
            crop,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        X.append(crop)
        y.append(class_id)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)


print()
print("Total object crops:", len(X))

for class_id, class_name in CLASS_NAMES.items():
    count = np.sum(y == class_id)
    print(f"{class_name.capitalize()}: {count}")


# ============================================================
# PREPROCESS
# ============================================================

X = preprocess_input(X)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("Training objects:", len(X_train))
print("Validation objects:", len(X_val))


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print()
print("==========================================")
print("CLASS WEIGHTS")
print("==========================================")

for class_id in sorted(class_weights):
    print(
        f"{CLASS_NAMES[class_id].capitalize()}: "
        f"{class_weights[class_id]:.2f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True
)

datagen.fit(X_train)


# ============================================================
# MOBILE NET V2
# ============================================================

print()
print("Loading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze base model
base_model.trainable = False


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

output = Dense(
    3,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_FOLDER,
    "car_colour_model_improved.keras"
)

callbacks = [

    ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print()
print("==========================================")
print("STARTING TRAINING")
print("==========================================")

history = model.fit(
    datagen.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE
    ),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# VALIDATION
# ============================================================

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print()
print("==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)

print()
print("Best model saved at:")
print(best_model_path)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_FOLDER,
    "car_colour_model_improved_final.keras"
)

model.save(final_model_path)

print()
print("Final model saved at:")
print(final_model_path)

print()
print("==========================================")
print("DONE")
print("==========================================")