import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading
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

class UniformInspectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uniform Inspection System")
        
        self.root.configure(bg="#2b3a42")

        self.title_label = Label(root, text="UNIPECT", font=("Helvetica", 24, "bold"), fg="#ffffff", bg="#2b3a42")
        self.title_label.pack(pady=10, anchor='nw', padx=10)

        self.label = Label(root, text="Welcome to the Uniform Inspection System", font=("Helvetica", 14), fg="#ffffff", bg="#2b3a42")
        self.label.pack(pady=10)

        self.start_button = Button(root, text="Start Inspection", font=("Helvetica", 12, "bold"), bg="#218c74", fg="#ffffff", command=self.start_inspection, relief="solid", borderwidth=2, padx=10, pady=5)
        self.start_button.pack(pady=10)

        self.check_again_button = Button(root, text="Check Again", font=("Helvetica", 12, "bold"), bg="#f39c12", fg="#ffffff", command=self.check_again, state=tk.DISABLED, relief="solid", borderwidth=2, padx=10, pady=5)
        self.check_again_button.pack(pady=10)

        self.quit_button = Button(root, text="Quit", font=("Helvetica", 12, "bold"), bg="#b33939", fg="#ffffff", command=root.quit, relief="solid", borderwidth=2, padx=10, pady=5)
        self.quit_button.pack(pady=10)

        self.video_label = Label(root, bg="#2b3a42")
        self.video_label.pack(pady=10)

        self.result_label = Label(root, font=("Helvetica", 33), fg="#ffffff", bg="#2b3a42")
        self.result_label.pack(pady=10)

        self.id_card_label = Label(root, font=("Helvetica", 33), fg="#ffffff", bg="#2b3a42")
        self.id_card_label.pack(pady=10)

        self.camera = None
        self.model = load_model_with_custom_depthwise("models/main/keras_model.h5")
        self.class_names = open("models/main/labels.txt", "r").readlines()

    def start_inspection(self):
        self.camera = cv2.VideoCapture(0)
        self.inspection_thread = threading.Thread(target=self.uniform_inspection)
        self.inspection_thread.start()
        self.update_frame()
        self.start_button.config(state=tk.DISABLED)
        self.check_again_button.config(state=tk.DISABLED)
        self.change_bg_color("#2b3a42")

    def check_again(self):
        self.result_label.config(text="")
        self.id_card_label.config(text="")
        self.start_button.config(state=tk.NORMAL)
        self.check_again_button.config(state=tk.DISABLED)

        # Close and reset the camera
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        self.camera = None

    def update_frame(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.root.after(10, self.update_frame)  # Update the frame every 10 ms

    def change_bg_color(self, color, text_fg):
        self.root.configure(bg=color)
        self.title_label.configure(bg=color, fg=text_fg)
        self.label.configure(bg=color, fg=text_fg)
        self.result_label.configure(bg=color, fg=text_fg)
        self.id_card_label.configure(bg=color, fg=text_fg)

    def uniform_inspection(self):
        L = []
        for i in range(5):
            ret, image = self.camera.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1

            prediction = self.model.predict(image_array, verbose=0)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]
            fc = str(np.round(confidence_score * 100))[:-2]
            print("Class:", class_name.strip(), "| Confidence Score:", fc, "%")
            
            L.append(prediction[0][0])

            if cv2.waitKey(1) == 27:
                break

        mean = s.mean(L) * 100
        if mean > 50:
            self.result_label.config(text="You are in uniform :)", fg="#ffffff")  # White color for positive result
            self.id_card_check()
        else:
            self.result_label.config(text="You are not in uniform!", fg="#ffffff")  # White color for negative result
            self.change_bg_color("#ff3d3d", "#ffffff")  # Red background for negative result
            self.check_again_button.config(state=tk.NORMAL)

        if self.camera:
            self.camera.release()
            self.camera = None

    def id_card_check(self):
        try:
            model = load_model_with_custom_depthwise("models/id/keras_model.h5")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        class_names = open("models/id/labels.txt", "r").readlines()
        L1 = []
        
        for i in range(5):
            ret, image = self.camera.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1

            prediction = model.predict(image_array, verbose=0)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            fc = str(np.round(confidence_score * 100))[:-2]
            print("Class:", class_name.strip(), "| Confidence Score:", fc, "%")
            L1.append(prediction[0][0])
            
            if cv2.waitKey(1) == 27:
                break

        if s.mean(L1) * 100 > 50:
            self.id_card_label.config(text="You are wearing an ID card", fg="#000000")  # Black color for positive result
            self.change_bg_color("#00ff42", "#000000")  # Green background for all positive results
        else:
            self.id_card_label.config(text="But you aren't wearing an ID card!", fg="#ffffff")  # White color for negative result
            self.change_bg_color("#ff3d3d", "#ffffff")  # Red background for any negative results

        if self.camera:
            self.camera.release()
            self.camera = None

        self.check_again_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = UniformInspectionApp(root)
    root.mainloop()
