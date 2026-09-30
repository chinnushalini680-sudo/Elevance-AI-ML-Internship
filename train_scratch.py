import os
import cv2
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Path matching your directory structure shown in the screenshot
DATASET_DIR = "images"
CLASSES = ["indian", "United_States", "African", "Other"]
IMAGE_SIZE = (64, 64)

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    return resized.flatten()

def load_data():
    X, y = [], []
    for class_idx, class_name in enumerate(CLASSES):
        folder_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(class_idx)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Loading image dataset...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No images found!")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training model on {len(X_train)} images...")
        
        clf = SVC(kernel='linear', C=1.0, probability=True)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(f"Training Complete! Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        with open("nationality_classifier.pkl", "wb") as f:
            pickle.dump(clf, f)
        print("Saved model as 'nationality_classifier.pkl'.")