import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset"
MODEL_PATH = "models/sign_language_model.keras"
LABEL_PATH = "models/labels.json"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 40

# ==============================
# CREATE MODEL FOLDERS
# ==============================

os.makedirs("models", exist_ok=True)

# ==============================
# LOAD DATASET
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)

# Save class names
with open(LABEL_PATH, "w") as f:
    json.dump(class_names, f)

# ==============================
# IMPROVE DATA PERFORMANCE
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.08, 0.08)
])

# ==============================
# CLASS WEIGHTS
# ==============================

class_counts = {}

for class_index, class_name in enumerate(class_names):
    folder = os.path.join(DATASET_PATH, class_name)

    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    class_counts[class_index] = count

print("\nImage counts:")
for index, count in class_counts.items():
    print(class_names[index], ":", count)

y = []

for class_index, count in class_counts.items():
    y.extend([class_index] * count)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=np.array(y)
)

class_weights = {
    i: float(class_weights_array[i])
    for i in range(len(class_names))
}

print("\nClass weights:")
print(class_weights)

# ==============================
# CNN MODEL
# ==============================

model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(len(class_names), activation="softmax")
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# CALLBACKS
# ==============================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ==============================
# TRAIN
# ==============================

print("\n==============================")
print("STARTING SIGN LANGUAGE TRAINING")
print("==============================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# ==============================
# FINAL RESULTS
# ==============================

loss, accuracy = model.evaluate(validation_dataset)

print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")

print("Validation Accuracy:", round(accuracy * 100, 2), "%")
print("Validation Loss:", round(loss, 4))

print("\nModel saved to:")
print(MODEL_PATH)

print("\nLabels saved to:")
print(LABEL_PATH)