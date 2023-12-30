
                                                                                                                                                                
### 2nd best
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# from ultralytics import YOLO
# from alexnetModel import predict
# from augment import extensive_augmentation
# import os
# import cv2
# import numpy as np

# class YOLODetectorApp:
#     def __init__(self, root):
#         self.root = root
#         self.image_path = ""
#         self.yolo = YOLO("./best.pt")

#         self.num_lines_label = tk.Label(root, text="Number of Cattle detected: ")
#         self.num_lines_label.pack()

#         self.root.title("YOLO Detector")
#         self.root.geometry("800x600")  # Set an initial size for the window

#         self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
#         self.upload_button.pack(pady=10)

#         self.detect_button = tk.Button(root, text="Detect", command=self.detect_objects)
#         self.detect_button.pack(pady=10)

#         self.original_image_label = tk.Label(root, text="Original Image")
#         self.original_image_label.pack()

#         # Create a Canvas for the original image with a Scrollbar
#         self.original_image_canvas = tk.Canvas(root)
#         self.original_image_canvas.pack(side="left", fill="both", expand=True)

#         self.original_image_scrollbar = tk.Scrollbar(root, command=self.original_image_canvas.yview)
#         self.original_image_scrollbar.pack(side="left", fill="y")

#         self.original_image_canvas.configure(yscrollcommand=self.original_image_scrollbar.set)

#         self.original_image_frame = tk.Frame(self.original_image_canvas)
#         self.original_image_canvas.create_window((0, 0), window=self.original_image_frame, anchor="nw")

#         self.original_image_display = tk.Label(self.original_image_frame)
#         self.original_image_display.pack()

#         self.result_image_label = tk.Label(root, text="Result Image")
#         self.result_image_label.pack()

#         # Create a Canvas for the result image with a Scrollbar
#         self.result_image_canvas = tk.Canvas(root)
#         self.result_image_canvas.pack(side="left", fill="both", expand=True)

#         self.result_image_scrollbar = tk.Scrollbar(root, command=self.result_image_canvas.yview)
#         self.result_image_scrollbar.pack(side="left", fill="y")

#         self.result_image_canvas.configure(yscrollcommand=self.result_image_scrollbar.set)

#         self.result_image_frame = tk.Frame(self.result_image_canvas)
#         self.result_image_canvas.create_window((0, 0), window=self.result_image_frame, anchor="nw")

#         self.result_image_display = tk.Label(self.result_image_frame)
#         self.result_image_display.pack()

#         self.image_path = ""
#         self.yolo = YOLO("./best.pt")

#     def upload_image(self):
#         self.image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
#         if self.image_path:
#             image = Image.open(self.image_path)
#             image = ImageTk.PhotoImage(image)
#             self.original_image_display.config(image=image)
#             self.original_image_display.image = image

#     def get_numeric_part(self, item):
#         numeric_part = item.replace('predict', '')
#         return int(numeric_part) if numeric_part.isdigit() else 0  # Use 0 if not a valid numeric part

#     def detect_objects(self):
#         if self.image_path:
#             # Load the original image
#             original_image = cv2.imread(self.image_path)

#             # Apply extensive augmentation
#             augmented_image = extensive_augmentation(original_image)

#             # Make predictions using YOLO on the augmented image
#             results = self.yolo(augmented_image, save=True, save_txt=True)

#             # Display the original image
#             original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#             original_image_pil = ImageTk.PhotoImage(original_image_pil)
#             self.original_image_display.config(image=original_image_pil)
#             self.original_image_display.image = original_image_pil

#             # Find the latest prediction number using the custom function
#             prediction_numbers = [self.get_numeric_part(folder_name) for folder_name in os.listdir("./runs/detect/") if folder_name.startswith("predict")]

#             if prediction_numbers:
#                 latest_prediction_number = max(prediction_numbers)
#             else:
#                 # Handle the case when there are no numbered folders, consider "predict" to be the latest
#                 latest_prediction_number = 0

#             latest_result_folder = os.path.join("./runs/detect/", f"predict{latest_prediction_number}")

#             # Find the detected image with a ".jpg" extension in the "predict" folder
#             detected_image_path = None
#             for file_name in os.listdir(latest_result_folder):
#                 if file_name.endswith(".jpg"):
#                     detected_image_path = os.path.join(latest_result_folder, file_name)
#                     break

#             if detected_image_path:
#                 # Load the detected image
#                 result_image = Image.open(detected_image_path)
                
#                 ### alexnet model line, prints the class
#                 print(predict(cv2.imread(detected_image_path)))
                
                
#                 result_image = ImageTk.PhotoImage(result_image)

#                 # Display the result image
#                 self.result_image_display.config(image=result_image)
#                 self.result_image_display.image = result_image

#                 # Find the label folder inside the latest "predict" folder
#                 label_folder_path = os.path.join(latest_result_folder, "labels")

#                 # Get the list of txt files in the label folder
#                 label_files = [file for file in os.listdir(label_folder_path) if file.endswith(".txt")]

#                 # Read the content of the first txt file (assuming there's only one txt file)
#                 num_lines = 0
#                 if label_files:
#                     label_file_path = os.path.join(label_folder_path, label_files[0])
#                     with open(label_file_path, 'r') as file:
#                         num_lines = sum(1 for line in file)

#                 # Display the number of lines in the Tkinter window
#                 self.num_lines_label.config(text=f"Number of Cattle detected: {num_lines}")
#             else:
#                 print("No detected image found.")
                
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = YOLODetectorApp(root)
#     root.mainloop()


##### 3RD BEST
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# from ultralytics import YOLO
# from alexnetModel import predict
# from augment import extensive_augmentation
# import os
# import cv2

# class YOLODetectorApp:
#     def __init__(self, root):
#         self.root = root
#         self.image_path = ""
#         self.yolo = YOLO("./best.pt")

#         self.num_lines_label = tk.Label(root, text="YOLO model count: Number of Cattle detected: ")
#         self.num_lines_label.pack()

#         # Additional label for AlexNet prediction
#         self.alexnet_prediction_label = tk.Label(root, text="AlexNet model output: ")
#         self.alexnet_prediction_label.pack()

#         self.root.title("YOLO Detector")
#         self.root.geometry("800x600")  # Set an initial size for the window

#         self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
#         self.upload_button.pack(pady=10)

#         self.detect_button = tk.Button(root, text="Detect", command=self.detect_objects)
#         self.detect_button.pack(pady=10)

#         self.original_image_label = tk.Label(root, text="Original Image")
#         self.original_image_label.pack()

#         # Create a Canvas for the original image with a Scrollbar
#         self.original_image_canvas = tk.Canvas(root)
#         self.original_image_canvas.pack(side="left", fill="both", expand=True)

#         self.original_image_scrollbar = tk.Scrollbar(root, command=self.original_image_canvas.yview)
#         self.original_image_scrollbar.pack(side="left", fill="y")

#         self.original_image_canvas.configure(yscrollcommand=self.original_image_scrollbar.set)

#         self.original_image_frame = tk.Frame(self.original_image_canvas)
#         self.original_image_canvas.create_window((0, 0), window=self.original_image_frame, anchor="nw")

#         self.original_image_display = tk.Label(self.original_image_frame)
#         self.original_image_display.pack()

#         self.result_image_label = tk.Label(root, text="Result Image")
#         self.result_image_label.pack()

#         # Create a Canvas for the result image with a Scrollbar
#         self.result_image_canvas = tk.Canvas(root)
#         self.result_image_canvas.pack(side="left", fill="both", expand=True)

#         self.result_image_scrollbar = tk.Scrollbar(root, command=self.result_image_canvas.yview)
#         self.result_image_scrollbar.pack(side="left", fill="y")

#         self.result_image_canvas.configure(yscrollcommand=self.result_image_scrollbar.set)

#         self.result_image_frame = tk.Frame(self.result_image_canvas)
#         self.result_image_canvas.create_window((0, 0), window=self.result_image_frame, anchor="nw")

#         self.result_image_display = tk.Label(self.result_image_frame)
#         self.result_image_display.pack()

#         self.image_path = ""
#         self.yolo = YOLO("./best.pt")

#     def upload_image(self):
#         self.image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
#         if self.image_path:
#             image = Image.open(self.image_path)
#             image = ImageTk.PhotoImage(image)
#             self.original_image_display.config(image=image)
#             self.original_image_display.image = image

#     def get_numeric_part(self, item):
#         numeric_part = item.replace('predict', '')
#         return int(numeric_part) if numeric_part.isdigit() else 0  # Use 0 if not a valid numeric part

#     def detect_objects(self):
#         if self.image_path:
#             # Load the original image
#             original_image = cv2.imread(self.image_path)

#             # Apply extensive augmentation
#             augmented_image = extensive_augmentation(original_image)

#             # Make predictions using YOLO on the augmented image
#             results = self.yolo(augmented_image, save=True, save_txt=True)

#             # Display the original image
#             original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#             original_image_pil = ImageTk.PhotoImage(original_image_pil)
#             self.original_image_display.config(image=original_image_pil)
#             self.original_image_display.image = original_image_pil

#             # Find the latest prediction number using the custom function
#             prediction_numbers = [self.get_numeric_part(folder_name) for folder_name in os.listdir("./runs/detect/") if folder_name.startswith("predict")]

#             if prediction_numbers:
#                 latest_prediction_number = max(prediction_numbers)
#             else:
#                 # Handle the case when there are no numbered folders, consider "predict" to be the latest
#                 latest_prediction_number = 0

#             latest_result_folder = os.path.join("./runs/detect/", f"predict{latest_prediction_number}")

#             # Find the detected image with a ".jpg" extension in the "predict" folder
#             detected_image_path = None
#             for file_name in os.listdir(latest_result_folder):
#                 if file_name.endswith(".jpg"):
#                     detected_image_path = os.path.join(latest_result_folder, file_name)
#                     break

#             if detected_image_path:
#                 # Load the detected image
#                 result_image = Image.open(detected_image_path)
                
#                 ### alexnet model line, prints the class
#                 alexnet_prediction = predict(cv2.imread(detected_image_path))
#                 # print(alexnet_prediction)
                
#                 # Set the AlexNet prediction text in the label
#                 self.alexnet_prediction_label.config(text=f"AlexNet model output: {alexnet_prediction}")
                
#                 result_image = ImageTk.PhotoImage(result_image)

#                 # Display the result image
#                 self.result_image_display.config(image=result_image)
#                 self.result_image_display.image = result_image

#                 # Find the label folder inside the latest "predict" folder
#                 label_folder_path = os.path.join(latest_result_folder, "labels")

#                 # Get the list of txt files in the label folder
#                 label_files = [file for file in os.listdir(label_folder_path) if file.endswith(".txt")]

#                 # Read the content of the first txt file (assuming there's only one txt file)
#                 num_lines = 0
#                 if label_files:
#                     label_file_path = os.path.join(label_folder_path, label_files[0])
#                     with open(label_file_path, 'r') as file:
#                         num_lines = sum(1 for line in file)

#                 # Display the number of lines in the Tkinter window
#                 self.num_lines_label.config(text=f"YOLO model counting: Number of Cattle detected: {num_lines}")
#             else:
#                 print("No detected image found.")
                
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = YOLODetectorApp(root)
#     root.mainloop()


import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from segmentation import segment_image
from alexnetModel import predict
from augment import extensive_augmentation
import os
import cv2

class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.image_path = ""
        self.yolo = YOLO("./best.pt")

        self.num_lines_label = tk.Label(root, text="YOLO model count: Number of Cattle detected: ")
        self.num_lines_label.pack()

        # Additional label for AlexNet prediction
        self.alexnet_prediction_label = tk.Label(root, text="AlexNet model output: ")
        self.alexnet_prediction_label.pack()

        self.root.title("YOLO Detector")
        self.root.geometry("800x600")  # Set an initial size for the window

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        self.original_image_label = tk.Label(root, text="Original Image")
        self.original_image_label.pack()

        # Create a Canvas for the original image with a Scrollbar
        self.original_image_canvas = tk.Canvas(root)
        self.original_image_canvas.pack(side="left", fill="both", expand=True)

        self.original_image_scrollbar = tk.Scrollbar(root, command=self.original_image_canvas.yview)
        self.original_image_scrollbar.pack(side="left", fill="y")

        self.original_image_canvas.configure(yscrollcommand=self.original_image_scrollbar.set)

        self.original_image_frame = tk.Frame(self.original_image_canvas)
        self.original_image_canvas.create_window((0, 0), window=self.original_image_frame, anchor="nw")

        self.original_image_display = tk.Label(self.original_image_frame)
        self.original_image_display.pack()

        self.result_image_label = tk.Label(root, text="Result Image")
        self.result_image_label.pack()

        # Create a Canvas for the result image with a Scrollbar
        self.result_image_canvas = tk.Canvas(root)
        self.result_image_canvas.pack(side="left", fill="both", expand=True)

        self.result_image_scrollbar = tk.Scrollbar(root, command=self.result_image_canvas.yview)
        self.result_image_scrollbar.pack(side="left", fill="y")

        self.result_image_canvas.configure(yscrollcommand=self.result_image_scrollbar.set)

        self.result_image_frame = tk.Frame(self.result_image_canvas)
        self.result_image_canvas.create_window((0, 0), window=self.result_image_frame, anchor="nw")

        self.result_image_display = tk.Label(self.result_image_frame)
        self.result_image_display.pack()

        self.image_path = ""
        self.yolo = YOLO("./best.pt")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = ImageTk.PhotoImage(image)
            self.original_image_display.config(image=image)
            self.original_image_display.image = image

    def get_numeric_part(self, item):
        numeric_part = item.replace('predict', '')
        return int(numeric_part) if numeric_part.isdigit() else 0  # Use 0 if not a valid numeric part

    def detect_objects(self):
        if self.image_path:
            # Load the original image
            original_image = cv2.imread(self.image_path)

            # Apply extensive augmentation
            augmented_image = extensive_augmentation(original_image)

            # Make predictions using YOLO on the augmented image
            results = self.yolo(augmented_image, save=True, save_txt=True)

            # Display the original image
            original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            original_image_pil = ImageTk.PhotoImage(original_image_pil)
            self.original_image_display.config(image=original_image_pil)
            self.original_image_display.image = original_image_pil

            # Find the latest prediction number using the custom function
            prediction_numbers = [self.get_numeric_part(folder_name) for folder_name in os.listdir("./runs/detect/") if folder_name.startswith("predict")]

            if prediction_numbers:
                latest_prediction_number = max(prediction_numbers)
            else:
                # Handle the case when there are no numbered folders, consider "predict" to be the latest
                latest_prediction_number = 0

            latest_result_folder = os.path.join("./runs/detect/", f"predict{latest_prediction_number}")

            # Find the detected image with a ".jpg" extension in the "predict" folder
            detected_image_path = None
            for file_name in os.listdir(latest_result_folder):
                if file_name.endswith(".jpg"):
                    detected_image_path = os.path.join(latest_result_folder, file_name)
                    break

            if detected_image_path:
                # Load the detected image
                # result_image = Image.open(detected_image_path)
                
                ### alexnet model line, prints the class
                alexnet_prediction = predict(cv2.imread(detected_image_path))
                # print(alexnet_prediction)
                
                # Set the AlexNet prediction text in the label
                self.alexnet_prediction_label.config(text=f"AlexNet model output: {alexnet_prediction}")

                result_image = segment_image(detected_image_path)
                
                # result_image = ImageTk.PhotoImage(result_image)
                result_image_pil = Image.fromarray(np.array(result_image))
                result_image_tk = ImageTk.PhotoImage(result_image_pil)

                # Display the result image
                self.result_image_display.config(image=result_image_tk)
                self.result_image_display.image = result_image_tk

                # Find the label folder inside the latest "predict" folder
                label_folder_path = os.path.join(latest_result_folder, "labels")

                # Get the list of txt files in the label folder
                label_files = [file for file in os.listdir(label_folder_path) if file.endswith(".txt")]

                # Read the content of the first txt file (assuming there's only one txt file)
                num_lines = 0
                if label_files:
                    label_file_path = os.path.join(label_folder_path, label_files[0])
                    with open(label_file_path, 'r') as file:
                        num_lines = sum(1 for line in file)

                # Display the number of lines in the Tkinter window
                self.num_lines_label.config(text=f"YOLO model counting: Number of Cattle detected: {num_lines}")
            else:
                print("No detected image found.")
                
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()
