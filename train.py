import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "face_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = 96
BATCH_SIZE = 8
EPOCHS = 60

# ==========================================
# COUNTS
# ==========================================

awake_dir = os.path.join(DATASET_DIR, "awake")
sleep_dir = os.path.join(DATASET_DIR, "sleep")

awake_count = len([
    f for f in os.listdir(awake_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

sleep_count = len([
    f for f in os.listdir(sleep_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print()
print("================================")
print("DROWSINESS CNN FROM SCRATCH")
print("================================")
print()

print("Awake:", awake_count)
print("Sleep:", sleep_count)
print("Total:", awake_count + sleep_count)

# ==========================================
# DATA GENERATOR
# ==========================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20,

    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    brightness_range=(0.85, 1.15)
)

# ==========================================
# TRAIN DATA
# ==========================================

train_data = datagen.flow_from_directory(
    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    classes=["awake", "sleep"],

    subset="training",

    shuffle=True,

    seed=42
)

# ==========================================
# VALIDATION DATA
# ==========================================

validation_data = datagen.flow_from_directory(
    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    classes=["awake", "sleep"],

    subset="validation",

    shuffle=False,

    seed=42
)

print()
print("Class mapping:")
print(train_data.class_indices)
print()

# ==========================================
# CNN FROM SCRATCH
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    ),

    # BLOCK 1

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.20),

    # BLOCK 2

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.25),

    # BLOCK 3

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.30),

    # CLASSIFIER

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.40),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

print("Model created from scratch.")
print()

# ==========================================
# CALLBACKS
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "drowsiness_model.keras"
)

checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.000001,
    verbose=1
)

# ==========================================
# TRAIN
# ==========================================

print("================================")
print("STARTING TRAINING")
print("================================")
print()

history = model.fit(
    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ==========================================
# RESULTS
# ==========================================

best_train = max(
    history.history["accuracy"]
)

best_val = max(
    history.history["val_accuracy"]
)

print()
print("================================")
print("TRAINING COMPLETE")
print("================================")
print()

print(
    f"Best training accuracy: "
    f"{best_train * 100:.2f}%"
)

print(
    f"Best validation accuracy: "
    f"{best_val * 100:.2f}%"
)

print()

print("Model saved at:")
print(model_path)