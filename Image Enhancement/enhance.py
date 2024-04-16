import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image


img_bgr = cv.imread("New_Zealand_Coast.jpg", cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# Display 18x18 pixel image.
Image(filename="New_Zealand_Coast.jpg")

# mat = np.ones(img_rgb.shape, dtype=np.uint8)*50

# lighter = cv.add(img_rgb, mat)
# darker = cv.subtract(img_rgb, mat)

# plt.figure(figsize=[18,5])
# plt.subplot(131);plt.imshow(img_rgb);plt.title('Original Image')
# plt.subplot(132);plt.imshow(lighter);plt.title('Lighter Image')
# plt.subplot(133);plt.imshow(darker);plt.title('Darker Image')

matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker   = np.uint8(cv.multiply(np.float64(img_rgb), matrix1))
# img_rgb_brighter = np.uint8(cv.multiply(np.float64(img_rgb), matrix2))
img_rgb_higher = np.uint8(np.clip(cv.multiply(np.float64(img_rgb), matrix2), 0, 255))

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Lower Contrast")
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original")
plt.subplot(133); plt.imshow(img_rgb_higher);plt.title("Higher Contrast")
plt.show()
