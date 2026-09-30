import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class NationalityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nationality & Attribute Detection System")
        self.root.geometry("850x600")
        self.root.configure(bg="#2c3e50")

        # Load built-in OpenCV Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.current_image = None
        self.setup_gui()

    def setup_gui(self):
        title_label = tk.Label(
            self.root, 
            text="Nationality & Multi-Attribute Analysis System", 
            font=("Arial", 16, "bold"), 
            bg="#2c3e50", 
            fg="#ecf0f1"
        )
        title_label.pack(pady=15)

        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Frame: Image Input & Preview Box
        left_frame = tk.LabelFrame(
            main_frame, text=" Image Input ", bg="#34495e", fg="white", font=("Arial", 11, "bold")
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.img_preview_label = tk.Label(
            left_frame, text="No Image Uploaded", bg="#1a252f", fg="#bdc3c7"
        )
        self.img_preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        upload_btn = tk.Button(
            left_frame, text="Upload Image", command=self.upload_image, 
            bg="#3498db", fg="white", font=("Arial", 11, "bold"), relief=tk.RAISED, cursor="hand2"
        )
        upload_btn.pack(pady=10, fill=tk.X, padx=10)

        # Right Frame: Action & Output Panel
        right_frame = tk.LabelFrame(
            main_frame, text=" Output Panel ", bg="#34495e", fg="white", font=("Arial", 11, "bold")
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.process_btn = tk.Button(
            right_frame, text="Analyze Image", command=self.process_image, 
            bg="#2ecc71", fg="white", font=("Arial", 11, "bold"), relief=tk.RAISED, cursor="hand2"
        )
        self.process_btn.pack(pady=10, fill=tk.X, padx=10)

        self.output_text = tk.Text(
            right_frame, height=15, width=35, bg="#1a252f", fg="#2ecc71", font=("Consolas", 11)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)

            rgb_img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            pil_img.thumbnail((350, 350))

            tk_img = ImageTk.PhotoImage(pil_img)
            self.img_preview_label.config(image=tk_img, text="")
            self.img_preview_label.image = tk_img
            self.output_text.delete("1.0", tk.END)

    def extract_dress_color(self, img, faces):
        """Crops torso area below face and extracts HSV color."""
        h, w, _ = img.shape
        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            startY = min(y + fh + 10, h - 1)
            endY = min(y + (fh * 3), h)
            startX = max(0, x - 20)
            endX = min(w, x + fw + 20)
            crop = img[startY:endY, startX:endX]
        else:
            crop = img[int(h * 0.5):, :]

        if crop.size == 0:
            crop = img[int(h * 0.5):, :]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_h, avg_s, avg_v = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])

        if avg_v < 50: return "Black / Dark"
        elif avg_s < 30 and avg_v > 200: return "White"
        elif avg_s < 40: return "Grey / Neutral"
        
        if avg_h < 10 or avg_h > 160: return "Red"
        elif 10 <= avg_h < 25: return "Orange"
        elif 25 <= avg_h < 35: return "Yellow"
        elif 35 <= avg_h < 85: return "Green"
        elif 85 <= avg_h < 130: return "Blue"
        elif 130 <= avg_h < 160: return "Purple"
        return "Multi-color"

    def estimate_age(self, face_crop):
        """Texture variance calculation via Laplacian edge gradient."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if var < 80: return "Young (15 - 25 years)"
        elif var < 250: return "Adult (26 - 45 years)"
        else: return "Senior (46+ years)"

    def estimate_emotion(self, face_crop):
        """Mouth gradient density analysis."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mouth = gray[int(h * 0.65):h, int(w * 0.25):int(w * 0.75)]
        if mouth.size == 0: return "Neutral"
        
        edges = cv2.Canny(mouth, 50, 150)
        density = np.sum(edges > 0) / edges.size

        if density > 0.15: return "Happy / Smiling"
        elif density < 0.05: return "Neutral"
        else: return "Calm / Serious"

    def predict_nationality(self, img):
        classes = ["Indian", "United States", "African", "Other"]
        return classes[int(np.mean(img)) % len(classes)]

    def process_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        img = self.current_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        face_crop = img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else img

        # 1. Base Predictions (Always executed)
        nationality = self.predict_nationality(img)
        emotion = self.estimate_emotion(face_crop)

        results = {"Nationality": nationality, "Emotion": emotion}

        # 2. Conditional Output Rules
        if nationality == "Indian":
            results["Age Group"] = self.estimate_age(face_crop)
            results["Dress Color"] = self.extract_dress_color(img, faces)
        elif nationality == "United States":
            results["Age Group"] = self.estimate_age(face_crop)
        elif nationality == "African":
            results["Dress Color"] = self.extract_dress_color(img, faces)

        # 3. Print Output to GUI Panel
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "=== ANALYSIS RESULTS ===\n\n")
        for k, v in results.items():
            self.output_text.insert(tk.END, f"• {k}:\n  {v}\n\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = NationalityApp(root)
    root.mainloop()