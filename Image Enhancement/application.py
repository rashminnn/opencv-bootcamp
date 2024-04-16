import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

img_bgr = cv.imread("coca-cola-logo.png")
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_background_bgr =cv.imread("checkerboard_color.png")
img_background_rgb =cv.cvtColor(img_background_bgr,cv.COLOR_BGR2RGB)

logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]

aratio = logo_w/img_background_rgb.shape[1]
dim = (logo_w ,int(img_background_rgb.shape[0]*aratio ))

img_background_rgb = cv.resize(img_background_rgb, dim, interpolation=cv.INTER_AREA)

grey = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
ret , img_mask = cv.threshold(grey, 127, 255, cv.THRESH_BINARY)

inv_mask =cv.bitwise_not(img_mask)

img_fg = cv.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)

img_foreground = cv.bitwise_and(img_rgb, img_rgb, mask=inv_mask)

result = cv.add(img_fg, img_foreground)
plt.imshow(result)
plt.show()