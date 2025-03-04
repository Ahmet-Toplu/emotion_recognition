# GUI to show pictures from test dataset and/or live camera
# If showing pictures display the guessed emotion and the labeled emotion
# If the live camera, display the guessed emotion only with a text box.
# When displaying the live camera divide the screen to two with one half show camera,
# and the other half for text inputs that is going to be implemented later.
# The model is not going to be trained here but the trained model will be used.
# The GUI is going to be implemented using tkinter library.

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
import os
import csv
import torchvision.transforms as transforms

from facenet_classifier import FaceNetClassifier

# Mock function to load your trained PyTorch model.
def load_model():
    # Choose the best model from models folder
    # The names of the models are divided with "_"
    # And the last item after division contains the val accuracy
    # Choose the model automatically
    models = os.listdir("./models")
    model_accuracies = {}

    for model in models:
        parts = model.split('_')
        if parts:
            try:
                acc_part = parts[-1].replace('.h5', '').replace('.pth', '')
                acc = float(acc_part)
                model_accuracies[model] = acc
            except ValueError:
                continue
    if model_accuracies:
        best_model = max(model_accuracies, key=model_accuracies.get)
        full_path = os.path.join('./models', best_model)
        print("Best model found:", best_model)
    else:
        print("No models found in the models folder.")

    # Branch by extension
    ext = os.path.splitext(best_model)[1].lower()
    if ext == ".pth":
        # This must be a PyTorch model or a pickled object
        model = FaceNetClassifier()
        if torch.cuda.is_available():
            state_dict = torch.load(full_path, map_location=torch.device('cuda'))
        else:
            state_dict = torch.load(full_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        # If it's a full PyTorch model, it should have .eval()
        try:
            model.eval()
            print("Training mode:", model.training)                # should be False if eval() was called
            print("FaceNet Training mode:", model.facenet.training)        # also check submodules
            print("Classifier Training mode:", model.classifier.training)
        except AttributeError:
            print("Loaded object has no eval(). Perhaps it's a state_dict? Construct a model and load_state_dict instead.")
            return None
        return model

    elif ext == ".h5":
        # This must be a Keras model
        print("Keras is not supported yet.")
        return None

    else:
        print(f"Unsupported extension: {ext}")
        return None


def preprocess_image(img_bgr):
    """
    Convert BGR image from OpenCV to whatever your model expects 
    (e.g. transform to tensor, normalize, resize, etc.).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = Image.fromarray(img_rgb)
    
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = preprocess(img_rgb)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_emotion(model, image_tensor):
    """
    Inference step, returning e.g. the predicted emotion string.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
    return int(pred.item())

# Placeholder: map label index to string emotion
EMOTION_MAP = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neutral",
    6: "Sad",
    7: "Surprise",
}

class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition GUI")

        # Load the model
        self.model = load_model()
        self.test_labels = self._load_test_labels("./cropped_mixed/test/test_labels.csv")

        # Create frames: 
        # left side for images / camera feed, right side for chat
        self.left_frame = tk.Frame(self.root, width=600, height=600, bg="gray")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.root, width=300, height=600, bg="white")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        # In the right frame, we can place a scrolledtext to simulate chat
        self.chat_box = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD, width=40, height=25)
        self.chat_box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5, expand=True)
        self.chat_box.config(state=tk.DISABLED)

        # A button or menu to switch modes (test images / live camera)
        self.btn_frame = tk.Frame(self.right_frame)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.test_images_btn = tk.Button(self.btn_frame, text="Test Images", command=self.open_test_images)
        self.test_images_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.live_camera_btn = tk.Button(self.btn_frame, text="Live Camera", command=self.start_live_camera)
        self.live_camera_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Label for displaying images / camera
        self.display_label = tk.Label(self.left_frame, bg="black")
        self.display_label.pack(fill=tk.BOTH, expand=True)

        self.cap = None  # for camera capture

        # For test images
        self.test_images_folder = './cropped_mixed/test'
        self.test_image_files = sorted(os.listdir(self.test_images_folder))
        self.test_index = 0

    def _load_test_labels(self, csv_path):
        """Loads a CSV file of (filename, label) pairs into a dict."""
        label_dict = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            # If your CSV has a header, skip it:
            # next(reader, None)
            for row in reader:
                # Adjust indices based on your CSV format
                filename, label = row[0], row[1]
                label_dict[filename] = label
        return label_dict

    def open_test_images(self):
        """Show test images in a loop. Press Next to move forward, or do it automatically."""
        self.stop_live_camera()  # in case camera is running

        # Clear left frame and add a "Next image" button
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        self.display_label = tk.Label(self.left_frame, bg="black")
        self.display_label.pack(fill=tk.BOTH, expand=True)

        next_btn = tk.Button(self.left_frame, text="Next Image", command=self.show_next_test_image)
        next_btn.pack(side=tk.BOTTOM, pady=5)

        self.test_index = 0
        self.show_next_test_image()

    def show_next_test_image(self):
        if self.test_index >= len(self.test_image_files):
            self.test_index = 0  # loop around or do nothing

        img_file = self.test_image_files[self.test_index]
        self.test_index += 1

        # Load the image with cv2
        img_path = os.path.join(self.test_images_folder, img_file)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return  # skip broken image

        # Suppose your test image's label is in the filename or a separate label file
        # We'll pretend the filename is "happy_001.jpg" => ground truth "happy"
        # This is just a placeholder
        ground_truth = self.test_labels.get(img_file, "Unknown")

        # Preprocess and predict
        input_tensor = preprocess_image(img_bgr)
        # input_tensor should be shape [1, ...] if it's a single image
        # input_tensor = input_tensor.unsqueeze(0)  # if needed

        pred_idx = predict_emotion(self.model, input_tensor)
        predicted_emotion = EMOTION_MAP[pred_idx]

        # Convert to PIL for display
        disp_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        disp_img = Image.fromarray(disp_img)
        disp_img = disp_img.resize((400, 400), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(disp_img)

        # Put text overlay in the label or just display as text
        self.display_label.config(image=tk_img)
        self.display_label.image = tk_img  # keep ref
        self.display_label_text = f"Pred: {predicted_emotion}\nActual: {ground_truth}"
        self.display_label.config(text=self.display_label_text, compound=tk.TOP, fg="white", font=("Arial", 14))

        # Also add a "chat" style message about it
        self._add_chat_message(f"Tested image: {img_file}\nPredicted: {predicted_emotion}, Actual: {ground_truth}", sender="System")

    def start_live_camera(self):
        """Start capturing from webcam and show in left_frame."""
        # If already capturing, do nothing
        if self.cap is not None:
            return

        self.cap = cv2.VideoCapture(0)  # or other device index
        # Clear left frame
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        self.display_label = tk.Label(self.left_frame, bg="black")
        self.display_label.pack(fill=tk.BOTH, expand=True)

        self._update_camera_frame()

    def _update_camera_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._add_chat_message("Failed to read from camera.", sender="System")
            return

        # Predict emotion on this frame
        input_tensor = preprocess_image(frame)
        # input_tensor = input_tensor.unsqueeze(0)  # if needed

        pred_idx = predict_emotion(self.model, input_tensor)
        predicted_emotion = EMOTION_MAP[pred_idx]

        # Draw the predicted emotion on the frame
        cv2.putText(
            frame, 
            f"{predicted_emotion}", 
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            (0, 255, 0), 
            2
        )

        # Convert for display
        disp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp_img = Image.fromarray(disp_img)
        disp_img = disp_img.resize((400, 400), Image.ANTIALIAS)
        tk_img = ImageTk.PhotoImage(disp_img)

        self.display_label.config(image=tk_img)
        self.display_label.image = tk_img  # keep ref

        # Callback again after 30ms
        self.display_label.after(30, self._update_camera_frame)

    def stop_live_camera(self):
        """Release the camera and remove reference."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _add_chat_message(self, text, sender="User"):
        """Add a message to the chat box on the right side."""
        self.chat_box.config(state=tk.NORMAL)

        if sender == "User":
            tag_color = "blue"
        else:
            tag_color = "red"

        if not tag_color in self.chat_box.tag_names():
            self.chat_box.tag_config(tag_color, foreground=tag_color)

        # Insert message with color
        self.chat_box.insert(tk.END, f"{sender}: ", tag_color)
        self.chat_box.insert(tk.END, text + "\n\n")

        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

def main():
    root = tk.Tk()
    gui = EmotionGUI(root)

    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(gui, root))
    root.mainloop()

def on_closing(gui, root):
    # Make sure camera is released before exit
    gui.stop_live_camera()
    root.destroy()

if __name__ == "__main__":
    main()
