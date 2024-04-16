import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image




img_rec = cv.imread("rectangle.jpg", cv.IMREAD_GRAYSCALE)

img_cir = cv.imread("circle.jpg", cv.IMREAD_GRAYSCALE)

# Bitwise AND operation
result1 = cv.bitwise_and(img_rec, img_cir, mask=None)
result2 = cv.bitwise_or(img_rec, img_cir, mask=None)
result3 = cv.bitwise_xor(img_rec, img_cir, mask=None)
result4 = cv.bitwise_not(img_rec, mask=None)

# Display the results
fig,ax = plt.subplots(3,3, figsize=(10,10))
ax[0,0].imshow(img_rec, cmap='gray')
ax[0,0].set_title('Rectangle')
ax[0,1].imshow(img_cir, cmap='gray')
ax[0,1].set_title('Circle')
ax[0,2].imshow(result1, cmap='gray')
ax[0,2].set_title('AND')
ax[1,0].imshow(result2, cmap='gray')
ax[1,0].set_title('OR')
ax[1,1].imshow(result3, cmap='gray')
ax[1,1].set_title('XOR')
ax[1,2].imshow(result4, cmap='gray')
ax[1,2].set_title('NOT')
ax[2,0].axis('off')
ax[2,1].axis('off')
ax[2,2].axis('off')
plt.show()