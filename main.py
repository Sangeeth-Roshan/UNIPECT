# Import required libraries
import tensorflow as tf
from keras.models import load_model

import statistics as s

import cv2
import numpy as np

# Import the modules
from mods.ID import ID

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

# Load the model
try:
    model = load_model_with_custom_depthwise("models/main/keras_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load labels
class_names = open("models/main/labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)

loop_num = 0

for i in range(5):
    # Grab the webcam's image
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
        # camera.release()
        # cv2.destroyAllWindows()
        

    # Resize and preprocess image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image_resized)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    # Predict with the model
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    fc = str(np.round(confidence_score * 100))[:-2]
    print("Class:", class_name.strip(), "| Confidence Score:", fc, "%")
    
    L = []
    L.append(prediction[0][0])
    mean = s.mean(L) * 100
    if mean > 50:
        Positive = True
    else:
        pass
    
    loop_num += 1
    if loop_num == 5:
        break
    
    

    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        # camera.release()
        # cv2.destroyAllWindows()
        break
camera.release()
cv2.destroyAllWindows()




'''ID CARD CHECK BEGINS'''
if Positive == True:
    
    # Load the model
    '''MODEL & LABELS'''
    try:
        model = load_model_with_custom_depthwise("models/id/keras_model.h5")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    # Load labels
    class_names = open("models/id/labels.txt", "r").readlines()
    camera = cv2.VideoCapture(0)
    
    loop1_num = 0
    L1 = []
    # while Positive == True:

    camera = cv2.VideoCapture(0)
    for i in range(5):
        ret, image = camera.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        
        
        # resize
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imshow("Webcam Image", image_resized)
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1
        
        # Predict with the model
        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        fc = str(np.round(confidence_score * 100))[:-2]
        print("Class:", class_name.strip(), "| Confidence Score:", fc, "%")
        L1.append(prediction[0][0])
        print(prediction[0][0])
        if cv2.waitKey(1) == 27:
        # camera.release()
        # cv2.destroyAllWindows()
            break
    
    if s.mean(L1) > 50:
        full_uniform = True
    else:
        pass
    
camera.release()
cv2.destroyAllWindows()

    
    

    
    

