import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

# img_read = cv.imread("building-windows.jpg", cv.IMREAD_GRAYSCALE)
# retval , img_thresh = cv.threshold(img_read, 100, 255, cv.THRESH_BINARY)


# # Show the images
# plt.figure(figsize=[18, 5])

# plt.subplot(121);plt.imshow(img_read, cmap="gray");  plt.title("Original")
# plt.subplot(122);plt.imshow(img_thresh, cmap="gray");plt.title("Thresholded")

# print(img_thresh.shape)
# plt.show()

# Read the original image
img_read = cv.imread("Piano_Sheet_Music.png", cv.IMREAD_GRAYSCALE)

# Perform global thresholding
retval, img_thresh_gbl_1 = cv.threshold(img_read, 50, 255, cv.THRESH_BINARY)

# Perform global thresholding
retval, img_thresh_gbl_2 = cv.threshold(img_read, 130, 255, cv.THRESH_BINARY)

# Perform adaptive thresholding
img_thresh_adp = cv.adaptiveThreshold(img_read, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 7)

# Show the images
plt.figure(figsize=[18,15])
plt.subplot(221); plt.imshow(img_read,        cmap="gray");  plt.title("Original");
plt.subplot(222); plt.imshow(img_thresh_gbl_1,cmap="gray");  plt.title("Thresholded (global: 50)");
plt.subplot(223); plt.imshow(img_thresh_gbl_2,cmap="gray");  plt.title("Thresholded (global: 130)");
plt.subplot(224); plt.imshow(img_thresh_adp,  cmap="gray");  plt.title("Thresholded (adaptive)");

plt.show()