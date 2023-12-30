import cv2
import numpy as np

def extensive_augmentation(input_image):
    # Brightness
    brightness_factor = np.random.uniform(0.2, 1.4)
    input_image = cv2.convertScaleAbs(input_image, alpha=brightness_factor, beta=0)

    # Sharpness (using kernel for convolution)
    sharpening_kernel = np.array([[-1, -1, 0],
                                  [-1,  6, -1],
                                  [0, -1, 0]])
    input_image = cv2.filter2D(input_image, -1, sharpening_kernel)

    return input_image

# img = cv2.imread(r".\download.jpg")

# cv2.imshow("Original Image", img)

# # Call your augmentation function
# augmented_img = extensive_augmentation(img)

# cv2.imshow("Augmented Image", augmented_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
