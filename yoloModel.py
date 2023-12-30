
# import os
# from ultralytics import YOLO
# import cv2

# model = YOLO('./best.pt')  # set image size to 640
# # print(model)


# im2 = cv2.imread("./img.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels


# def process_folder(folder_path):
#     max_item = ''
#     res_img_folder = os.listdir(folder_path)

#     # Define a key function to extract the numeric part from each filename
#     def get_numeric_part(item):
#         numeric_part = item.replace('predict', '').replace('.txt', '')
#         return int(numeric_part) if numeric_part.isdigit() else 0  # Use 0 if not a valid numeric part

#     if res_img_folder:
#         # Find the item with the highest count using the custom key function
#         max_item = max(res_img_folder, key=get_numeric_part)

#         res_labels_folder_path = os.path.join(folder_path, max_item, 'labels')
#         res_image_path = os.path.join(folder_path, max_item, 'image0.jpg')
#         response_img = cv2.imread(res_image_path)

#         # Get a list of all files with the .txt extension in the labels folder
#         txt_files = [file for file in os.listdir(res_labels_folder_path) if file.endswith(".txt")]

#         if txt_files:
#             # Take the first .txt file found
#             txt_file_name = txt_files[0]
#             txt_file_path = os.path.join(res_labels_folder_path, txt_file_name)

#             # Open the file and read its contents
#             with open(txt_file_path, 'r') as file:
#                 # Read the lines and print the count
#                 lines = file.readlines()
#                 line_count = len(lines)
#                 return line_count, response_img
#         else:
#             print("Error: No .txt files found in the 'labels' folder.")
#     else:
#         print("Error: No items found in the folder.")
#     return None, None

# # Example usage
# line_count, image = process_folder("./runs/detect")

# if line_count is not None and image is not None:
#     print(f"Number of cattles: {line_count}")
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# import os
# from ultralytics import YOLO
# import cv2
# import numpy as np

# def process_image_and_folder(input_image, model_path='./best.pt'):
#     model = YOLO(model_path)  # set image size to 640

#     # Save the input image for YOLO processing
#     input_image_path = "./input_image.jpg"
#     cv2.imwrite(input_image_path, input_image)

#     # Perform YOLO detection on the input image
#     results = model.predict(source=input_image_path, save=True, save_txt=True)

#     folder_path = "./runs/detect"  # Update with the actual path where YOLO saves results

#     max_item = ''
#     res_img_folder = os.listdir(folder_path)

#     # Define a key function to extract the numeric part from each filename
#     def get_numeric_part(item):
#         numeric_part = item.replace('predict', '').replace('.txt', '')
#         return int(numeric_part) if numeric_part.isdigit() else 0  # Use 0 if not a valid numeric part

#     if res_img_folder:
#         # Find the item with the highest count using the custom key function
#         max_item = max(res_img_folder, key=get_numeric_part)

#         res_labels_folder_path = os.path.join(folder_path, max_item, 'labels')
#         res_image_path = os.path.join(folder_path, max_item, 'image0.jpg')
#         response_img = cv2.imread(res_image_path)

#         # Get a list of all files with the .txt extension in the labels folder
#         txt_files = [file for file in os.listdir(res_labels_folder_path) if file.endswith(".txt")]

#         if txt_files:
#             # Take the first .txt file found
#             txt_file_name = txt_files[0]
#             txt_file_path = os.path.join(res_labels_folder_path, txt_file_name)

#             # Open the file and read its contents
#             with open(txt_file_path, 'r') as file:
#                 # Read the lines and print the count
#                 lines = file.readlines()
#                 line_count = len(lines)
#                 return line_count, response_img
#         else:
#             print("Error: No .txt files found in the 'labels' folder.")
#     else:
#         print("Error: No items found in the folder.")
#     return None, None

# # Example usage
# input_image = cv2.imread("./img2.jpg")  # Replace with the actual path to your input image
# line_count, image = process_image_and_folder(input_image)

# if line_count is not None and image is not None:
#     print(f"Number of cattles: {line_count}")
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


import os
from ultralytics import YOLO
import cv2
import numpy as np

def process_image_and_folder(input_image, model_path='./best.pt'):
    model = YOLO(model_path)  # set image size to 640

    # Save the input image for YOLO processing
    input_image_path = "./input_image.jpg"
    cv2.imwrite(input_image_path, input_image)

    # Perform YOLO detection on the input image
    results = model.predict(source=input_image_path, save=True, save_txt=True)

    folder_path = "./runs/detect"  # Update with the actual path where YOLO saves results

    max_item = ''
    res_img_folder = os.listdir(folder_path)

    # Define a key function to extract the numeric part from each filename
    def get_numeric_part(item):
        numeric_part = item.replace('predict', '').replace('.txt', '')
        return int(numeric_part) if numeric_part.isdigit() else 0  # Use 0 if not a valid numeric part

    if res_img_folder:
        # Find the item with the highest count using the custom key function
        max_item = max(res_img_folder, key=get_numeric_part)

        res_labels_folder_path = os.path.join(folder_path, max_item, 'labels')

        # Get the image file in the folder (with any name and .jpg extension)
        image_file = [file for file in os.listdir(os.path.join(folder_path, max_item)) if file.lower().endswith('.jpg')]

        if image_files:
            # Take the first .jpg file found
            image_file_name = image_files[0]
            res_image_path = os.path.join(folder_path, max_item, image_file_name)
            response_img = cv2.imread(res_image_path)

            # Get a list of all files with the .txt extension in the labels folder
            txt_files = [file for file in os.listdir(res_labels_folder_path) if file.endswith(".txt")]

            if txt_files:
                # Take the first .txt file found
                txt_file_name = txt_files[0]
                txt_file_path = os.path.join(res_labels_folder_path, txt_file_name)

                # Open the file and read its contents
                with open(txt_file_path, 'r') as file:
                    # Read the lines and print the count
                    lines = file.readlines()
                    line_count = len(lines)
                    return line_count, response_img
            else:
                print("Error: No .txt files found in the 'labels' folder.")
        else:
            print("Error: No image files found in the folder.")
    else:
        print("Error: No items found in the folder.")
    return None, None

# Example usage
# input_image = cv2.imread("./img2.jpg")  # Replace with the actual path to your input image
# line_count, image = process_image_and_folder(input_image)

# if line_count is not None and image is not None:
#     print(f"Number of cattles: {line_count}")
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
