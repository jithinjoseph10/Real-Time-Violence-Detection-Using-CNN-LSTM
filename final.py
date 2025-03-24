# final.py

import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.applications import VGG16
from keras import backend as K
from keras.models import Model

def start(mode, filepath):
    if mode == 0:
        real_time_detection()
    elif mode == 1:
        detect(mode, filepath)
    else:
        print("Invalid mode. Please specify 0 for real-time detection or 1 for video upload.")

def real_time_detection():
    print("Performing real-time detection...")
    # Implement real-time detection logic here

def detect(mode, filepath):
    print(f"Performing violence detection on video: {filepath}")
    model = load_model('model/vlstm_new.h5')
    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    cap = cv2.VideoCapture(filepath)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        processed_frame = preprocess_frame(frame)
        features = image_model_transfer.predict(processed_frame)
        prediction = model.predict(features[np.newaxis, ...])
        process_prediction(prediction)

    cap.release()
    cv2.destroyAllWindows()

def preprocess_frame(frame):
    processed_frame = frame  # Placeholder, replace with actual preprocessing steps
    return processed_frame

def process_prediction(prediction):
    if prediction > 0.5:
        print("Violence detected!")
        # Implement actions to take if violence is detected
    else:
        print("No violence detected.")

if __name__ == "__main__":
    print("This module contains functions for violence detection.")
