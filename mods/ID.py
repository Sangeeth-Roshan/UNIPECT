# Import required libraries
import tensorflow as tf
from keras.models import load_model

import cv2
import numpy as np

import os

# os
current_dir = os.path.dirname(os.path.abspath(__file__))


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Define a custom DepthwiseConv2D class to handle 'groups' argument
from keras.layers import DepthwiseConv2D
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        if groups is not None:
            print(f"Ignoring unsupported argument 'groups={groups}'")
        super().__init__(*args, **kwargs)

# Custom object scope for model loading
def load_model_with_custom_depthwise(model_path):
    with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
        return load_model(model_path, compile=False)


def ID(image):
    model_path = os.path.join(current_dir, "../models/id/keras_model.h5")
    try:
        model = load_model_with_custom_depthwise(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    
    for i in range(5):
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imshow("Webcam Image", image_resized)
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1
        
        
        file_path = os.path.join(current_dir, "../models/id/labels.txt")
        with open(file_path, "r") as file:
            class_names = file.readlines()
        
        
        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        fc = str(np.round(confidence_score * 100))[:-2]
        print("Class:", class_name.strip(), "| Confidence Score:", fc, "%")
        
        return fc

    