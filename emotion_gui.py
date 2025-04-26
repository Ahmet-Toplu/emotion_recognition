# Import required libraries
import tkinter as tk
from tkinter import scrolledtext  # For creating a scrolling chat box
import cv2                      # OpenCV for image and camera handling
from PIL import Image, ImageTk   # PIL for converting images for Tkinter
import torch                    # PyTorch for model handling
import numpy as np
import os                       # For file system operations
import csv                      # To read CSV files (for test labels)
import torchvision.transforms as transforms  # For image pre-processing

# Import your custom FaceNetClassifier model
from facenet_classifier import FaceNetClassifier

# Import MTCNN for face detection and landmarks
from facenet_pytorch import MTCNN

# Import a text classification pipeline from Hugging Face Transformers
from transformers import pipeline


def load_model():
    """
    Loads the best PyTorch model from the "./models" folder.
    The best model is determined by its validation accuracy, which is assumed to be part of the filename.
    Only models with a '.pth' extension (PyTorch state_dicts) are supported.
    """
    models = os.listdir("./models")
    model_accuracies = {}

    # Iterate over each model file in the directory
    for model in models:
        parts = model.split('_')
        if parts:
            try:
                # Extract the accuracy from the filename (assumes it's the last part before the extension)
                acc_part = parts[-1].replace('.h5', '').replace('.pth', '')
                acc = float(acc_part)
                model_accuracies[model] = acc
            except ValueError:
                # Skip files that do not follow the expected naming convention
                continue

    if model_accuracies:
        # Choose the model with the highest accuracy
        best_model = max(model_accuracies, key=model_accuracies.get)
        full_path = os.path.join('./models', best_model)
        print("Best model found:", best_model)
    else:
        print("No models found in the models folder.")

    # Determine the extension to handle different frameworks
    ext = os.path.splitext(best_model)[1].lower()
    if ext == ".pth":
        # For a PyTorch model (state_dict)
        model = FaceNetClassifier()  # Instantiate your model architecture
        # Load the state_dict with safety: load only weights (weights_only=True)
        if torch.cuda.is_available():
            state_dict = torch.load(full_path, map_location=torch.device('cuda'), weights_only=True)
        else:
            state_dict = torch.load(full_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)  # Load the weights into the model

        # Set the model to evaluation mode
        try:
            model.eval()
            print("Training mode:", model.training)  # Should print False after calling eval()
            print("FaceNet Training mode:", model.facenet.training)
            print("Classifier Training mode:", model.classifier.training)
        except AttributeError:
            print("Loaded object has no eval(). Perhaps it's a state_dict? Construct a model and load_state_dict instead.")
            return None
        return model

    elif ext == ".h5":
        # Keras model support is not implemented
        print("Keras is not supported yet.")
        return None
    else:
        print(f"Unsupported extension: {ext}")
        return None


def preprocess_image(img_bgr):
    """
    Preprocesses the image so that it is in the correct format for the model.
    Converts from BGR (OpenCV's default) to RGB, resizes the image to 160x160,
    converts it to a PyTorch tensor, and normalizes the pixel values.
    """
    # Convert the image from BGR to RGB format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Convert the numpy array to a PIL Image object
    img_rgb = Image.fromarray(img_rgb)
    
    # Define the preprocessing steps: resizing, tensor conversion, and normalization
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply the transformations
    img_tensor = preprocess(img_rgb)
    # Add an extra dimension to match the expected input shape [batch_size, channels, height, width]
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict_emotion(model, image_tensor):
    """
    Uses the model to predict the emotion for the given image tensor.
    Returns an integer corresponding to the predicted emotion label.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        # Get the index of the maximum value (predicted class)
        _, pred = torch.max(outputs, 1)
    return int(pred.item())


# A mapping from numeric labels to human-readable emotion strings
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
        # Initialize MTCNN for face detection; reuse this instance for each frame
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize the text classifier pipeline for emotion recognition in text
        self.text_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

        self.root = root
        self.root.title("Emotion Recognition GUI")

        # Load the trained image-based emotion model
        self.model = load_model()
        # Load the test labels (ground truth) from a CSV file
        self.test_labels = self._load_test_labels("./cropped_mixed/test/test_labels.csv")
        self.image_prediction = None  # Store the last image-based prediction

        # Create the left frame for displaying images (camera feed or test images)
        self.left_frame = tk.Frame(self.root, width=600, height=600, bg="gray")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create the right frame for chat and text inputs
        self.right_frame = tk.Frame(self.root, width=300, height=600, bg="white")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        # Create a scrolled text widget for the chat display in the right frame
        self.chat_box = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD, width=40, height=25)
        self.chat_box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5, expand=True)
        self.chat_box.config(state=tk.DISABLED)

        # Create a frame for the user input field and send button at the bottom of the right frame
        self.entry_frame = tk.Frame(self.right_frame)
        self.entry_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Create an entry widget for user text input
        self.user_entry = tk.Entry(self.entry_frame, width=30)
        self.user_entry.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        # Bind the "Enter" key to trigger processing of the user input
        self.user_entry.bind("<Return>", self.process_user_input)

        # Create a send button for user input
        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.process_user_input)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        # Create a button frame for switching between test images and live camera modes
        self.btn_frame = tk.Frame(self.right_frame)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Button to display test images
        self.test_images_btn = tk.Button(self.btn_frame, text="Test Images", command=self.open_test_images)
        self.test_images_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Button to start the live camera feed
        self.live_camera_btn = tk.Button(self.btn_frame, text="Live Camera", command=self.start_live_camera)
        self.live_camera_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Label widget in the left frame to display images or camera feed
        self.display_label = tk.Label(self.left_frame, bg="black")
        self.display_label.pack(fill=tk.BOTH, expand=True)

        self.cap = None  # This will hold the video capture object for the camera

        # Load test images from the specified folder
        self.test_images_folder = './cropped_mixed/test'
        self.test_image_files = sorted(os.listdir(self.test_images_folder))
        self.test_index = 0  # To keep track of which test image to display next

    def _load_test_labels(self, csv_path):
        """
        Reads a CSV file containing filenames and their corresponding emotion labels,
        and returns a dictionary mapping filenames to labels.
        """
        label_dict = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Uncomment the next line if the CSV has a header row
            # next(reader, None)
            for row in reader:
                filename, label = row[0], row[1]
                label_dict[filename] = label
        return label_dict

    def open_test_images(self):
        """
        Display test images one at a time in the left frame.
        Each image is processed and its predicted emotion is shown.
        """
        self.stop_live_camera()  # Stop live camera if it is running

        # Clear the left frame before displaying test images
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        # Create a new label in the left frame to display images
        self.display_label = tk.Label(self.left_frame, bg="black")
        self.display_label.pack(fill=tk.BOTH, expand=True)

        # Add a "Next Image" button to cycle through test images
        next_btn = tk.Button(self.left_frame, text="Next Image", command=self.show_next_test_image)
        next_btn.pack(side=tk.BOTTOM, pady=5)

        self.test_index = 0
        self.show_next_test_image()

    def show_next_test_image(self):
        """
        Loads the next test image, processes it through the model,
        and displays the image with predicted and actual emotions.
        """
        if self.test_index >= len(self.test_image_files):
            self.test_index = 0  # Loop back to the first image

        img_file = self.test_image_files[self.test_index]
        self.test_index += 1

        # Load image using OpenCV
        img_path = os.path.join(self.test_images_folder, img_file)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return  # Skip if the image is not found or is corrupted

        # Get the ground truth label from the CSV file (using EMOTION_MAP for conversion)
        ground_truth = EMOTION_MAP[int(self.test_labels.get(img_file))]

        # Preprocess the image and predict the emotion using the loaded model
        input_tensor = preprocess_image(img_bgr)
        pred_idx = predict_emotion(self.model, input_tensor)
        predicted_emotion = EMOTION_MAP[pred_idx]
        self.image_prediction = predicted_emotion

        # Convert image to a format Tkinter can display (PIL ImageTk)
        disp_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        disp_img = Image.fromarray(disp_img)
        disp_img = disp_img.resize((400, 400), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(disp_img)

        # Display the image along with the prediction text
        self.display_label.config(image=tk_img)
        self.display_label.image = tk_img  # Keep a reference to avoid garbage collection
        self.display_label_text = f"Pred: {predicted_emotion}\nActual: {ground_truth}"
        self.display_label.config(text=self.display_label_text, compound=tk.TOP, fg="white", font=("Arial", 14))

        # Also log the prediction in the chat box
        self._add_chat_message(f"Tested image: {img_file}\nPredicted: {predicted_emotion}, Actual: {ground_truth}", sender="System")

    def start_live_camera(self):
        """
        Starts the live camera feed.
        The camera frames are processed to detect faces, crop them,
        predict emotion, and overlay the prediction on the frame.
        """
        # If the camera is already running, do nothing
        if self.cap is not None:
            return

        self.cap = cv2.VideoCapture(0)  # Open the default camera
        # Clear the left frame
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        # Create a new label for displaying the live camera feed
        self.display_label = tk.Label(self.left_frame, bg="black")
        self.display_label.pack(fill=tk.BOTH, expand=True)

        # Start updating the camera frames
        self._update_camera_frame()

    def _update_camera_frame(self):
        """
        Continuously captures frames from the camera, processes each frame to detect faces,
        draws bounding boxes and landmarks, crops faces for emotion prediction,
        and updates the display.
        """
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._add_chat_message("Failed to read from camera.", sender="System")
            return

        # Use MTCNN to detect faces and landmarks in the frame
        boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

        if boxes is not None:
            for box, ld in zip(boxes, landmarks):
                # Convert the bounding box coordinates to integers
                x1, y1, x2, y2 = box.astype(int)
                # Draw a green rectangle around the detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw small circles at each detected landmark point
                for (x, y) in ld.astype(int):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Crop the face region from the frame
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    # If the crop is empty, log the coordinates and skip this face
                    print("Bounding box:", x1, y1, x2, y2)
                    print("Cropped face shape:", face_crop.shape)
                    continue

                # Preprocess the cropped face and predict the emotion
                input_tensor = preprocess_image(face_crop)
                pred_idx = predict_emotion(self.model, input_tensor)
                predicted_emotion = EMOTION_MAP[pred_idx]
                self.image_prediction = predicted_emotion

                # Overlay the predicted emotion text near the bounding box on the frame
                cv2.putText(
                    frame,
                    predicted_emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        # Convert the processed frame for display in Tkinter
        disp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp_img = Image.fromarray(disp_img)
        disp_img = disp_img.resize((400, 400), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(disp_img)
        self.display_label.config(image=tk_img)
        self.display_label.image = tk_img

        # Schedule the next frame update after 30 milliseconds
        self.display_label.after(30, self._update_camera_frame)

    def stop_live_camera(self):
        """Stops the camera feed by releasing the VideoCapture object."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_user_input(self, event=None):
        """
        Processes user text input:
        - Displays the user's message in the chat box.
        - Uses the text classifier to predict an emotion from the input text.
        - Displays both the text-based prediction and the most recent image-based prediction.
        """
        user_text = self.user_entry.get().strip()
        if not user_text:
            return  # Ignore empty input

        # Display the user's input in the chat box
        self._add_chat_message(user_text, sender="User")
        # Get the prediction from the text classifier and include the image-based prediction
        text_prediction = f"text: {self.text_classifier(user_text)[0]['label']} Image: {self.image_prediction}"
        self._add_chat_message(text_prediction, sender="System")

        # Clear the input field after processing
        self.user_entry.delete(0, tk.END)

    def _add_chat_message(self, text, sender="User"):
        """
        Adds a message to the chat box.
        Uses different text colors depending on whether the message is from the user or system.
        """
        self.chat_box.config(state=tk.NORMAL)

        tag_color = "purple" if sender == "User" else "red"

        if not tag_color in self.chat_box.tag_names():
            self.chat_box.tag_config(tag_color, foreground=tag_color)

        # Insert the sender's name and the message text into the chat box
        self.chat_box.insert(tk.END, f"{sender}: ", tag_color)
        self.chat_box.insert(tk.END, text + "\n\n")

        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)


def main():
    # Create the main Tkinter window
    root = tk.Tk()
    gui = EmotionGUI(root)

    # Ensure that the camera is stopped when closing the window
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(gui, root))
    root.mainloop()


def on_closing(gui, root):
    """Cleanup function to stop the camera and destroy the Tkinter window."""
    gui.stop_live_camera()
    root.destroy()


if __name__ == "__main__":
    main()
