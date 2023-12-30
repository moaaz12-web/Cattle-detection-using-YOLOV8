# import pixellib
# from pixellib.instance import instance_segmentation
# import cv2

# segmentation_model = instance_segmentation()
# segmentation_model.load_model('mask_rcnn_coco.h5')

# # Apply instance segmentation
# res = segmentation_model.segmentFrame('./img.jpg', show_bboxes=True)
# image = res[1]

# cv2.imshow('Instance Segmentation', image)
import cv2
import numpy as np
from PIL import Image

def segment_image(image_path):
    # Read the image
    # image = cv2.imread(image_path)
    image  = Image.open(image_path)

    # Convert the PIL Image to a NumPy array
    image = np.array(image)


    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale for better results
    grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    segmented_image = image_rgb.copy()
    cv2.drawContours(segmented_image, contours, -1, (255, 0, 0), 2)  # Blue color for contours

    return segmented_image

    # # Display the original and segmented images
    # cv2.imshow("Original Image", image_rgb)
    # cv2.imshow("Segmented Image", segmented_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage
# segment_image(r"runs\detect\predict\image0.jpg")
