# Emotion Recognition Project

This project focuses on emotion recognition using a Convolutional Neural Network (CNN) model with prioritized convolution.

## Overview

Emotion recognition is a process of identifying human emotions from facial expressions. This project leverages a CNN model to accurately classify emotions from images.

## Features

- Utilizes a CNN model for emotion recognition
- Implements prioritized convolution for improved performance
- High accuracy in emotion classification

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ahmet-Toplu/emotion_recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd emotion_recognition
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Its designed to run easily, so just by running the ER_FaceNet.ipynb file it should create a model with approximate accuracy of 75.4% and save it as facenet_ec_{accuracy}.pth. After making the model you can run the emotion_gui.py to be able to test it using live camera and using the test images using a UI. The UI is made to work with other models as well the only thing needs to be changed is the classifier (facenet_classifier.py, this needed to be able to load the model for emotion_gui.py) and it will always choose the model with the best accuracy.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any inquiries, please contact alexxhhofman@gmail.com.
