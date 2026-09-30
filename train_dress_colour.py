import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# SETTINGS
# ==========================================

CSV_FILE = "dress_labels.csv"
IMAGE_SIZE = 128


# ==========================================
# MAP ORIGINAL COLOURS TO BROADER COLOURS
# ==========================================

COLOUR_MAP = {

    "Red": "Red",
    "Maroon": "Red",

    "Pink": "Pink",
    "Mauve": "Pink",

    "Blue": "Blue",
    "Lavender": "Purple",
    "Purple": "Purple",
    "Violet": "Purple",

    "Green": "Green",

    "Yellow": "Yellow",
    "Gold": "Yellow",

    "Orange": "Orange",

    "Brown": "Brown",

    "Black": "Black",

    "White": "White",
    "Cream": "White",
    "Beige": "White",

    "Grey": "Neutral",
    "Silver": "Neutral",

    "Multicolour": "Multicolour"
}


# ==========================================
# LOAD CSV
# ==========================================

df = pd.read_csv(CSV_FILE)

print("Original labelled images:", len(df))

# Create broader colour label
df["broad_colour"] = df["dress_colour"].map(COLOUR_MAP)

# Check for unmapped labels
if df["broad_colour"].isna().any():

    print("\nWARNING: Unmapped colours found:")

    print(
        df.loc[
            df["broad_colour"].isna(),
            "dress_colour"
        ].unique()
    )

    raise SystemExit


print("\nBroad colour counts:")
print(df["broad_colour"].value_counts())


# ==========================================
# LOAD IMAGES
# ==========================================

images = []
labels = []

for _, row in df.iterrows():

    image_path = row["image"]
    colour = row["broad_colour"]

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        continue

    img = cv2.imread(image_path)

    if img is None:
        print("Could not read:", image_path)
        continue

    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype("float32") / 255.0

    images.append(img)
    labels.append(colour)


X = np.array(images)
labels = np.array(labels)

print("\nImages loaded:", len(X))
print("Image shape:", X.shape)


# ==========================================
# ENCODE LABELS
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(labels)

class_names = encoder.classes_

print("\nFinal classes:")

for i, name in enumerate(class_names):
    print(i, "=", name)


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print("\nTraining images:", len(X_train))
print("Validation images:", len(X_val))


# ==========================================
# ONE-HOT ENCODING
# ==========================================

y_train = to_categorical(
    y_train_encoded,
    num_classes=len(class_names)
)

y_val = to_categorical(
    y_val_encoded,
    num_classes=len(class_names)
)


# ==========================================
# CNN FROM SCRATCH
# ==========================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        len(class_names),
        activation="softmax"
    )
])


# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True
)


# ==========================================
# TRAIN
# ==========================================

print("\n======================================")
print("STARTING TRAINING")
print("======================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=16,

    callbacks=[
        early_stop
    ]
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/dress_colour_model.keras"
)


# Save the NEW broad colour classes
with open(
    "models/dress_colour_classes.txt",
    "w",
    encoding="utf-8"
) as f:

    for name in class_names:

        f.write(
            name + "\n"
        )


# ==========================================
# EVALUATE
# ==========================================

loss, accuracy = model.evaluate(

    X_val,
    y_val,

    verbose=0
)


print("\n======================================")
print("TRAINING COMPLETE")
print("======================================")

print(
    "Validation accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClasses:")

for name in class_names:
    print("-", name)

print("\nModel saved to:")
print("models/dress_colour_model.keras")

print("\nClasses saved to:")
print("models/dress_colour_classes.txt")

print("======================================")