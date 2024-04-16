import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np

img_NZ_bgr = cv.imread("New_Zealand_Boat.jpg", cv.IMREAD_COLOR)
assert img_NZ_bgr is not None
img_NZ_rgb = img_NZ_bgr[:, :, ::-1]

cropped = img_NZ_rgb[150:450 ,300 : 600 ]
# resize1
# resize = cv.resize(cropped,None,fx=2,fy=2)

# resize2
# width = 400
# height = 200
# dim = (width, height)
# resize = cv.resize(cropped, dim, interpolation = cv.INTER_AREA)

# resize maintaining aspect ratio
width = 400
aratio = width/cropped.shape[1]
height = int(cropped.shape[0]*aratio)

dim= (width, height)

resize = cv.resize(cropped, dim, interpolation = cv.INTER_AREA)


# flipping 
flip_horizontal = cv.flip(img_NZ_rgb, 1)
flip_vertical = cv.flip(img_NZ_rgb, 0)
flip_both = cv.flip(img_NZ_rgb, -1)

plt.figure(figsize=(20,20))
plt.subplot(141);plt.imshow(img_NZ_rgb);plt.title('Original Image')
plt.subplot(142);plt.imshow(flip_horizontal);plt.title('Flipped Horizontally')
plt.subplot(143);plt.imshow(flip_vertical);plt.title('Flipped Vertically')
plt.subplot(144);plt.imshow(flip_both);plt.title('Flipped Horizontally & Vertically')
plt.show()





# plt.imshow(resize, cmap='gray')
# plt.show()

# plt.imshow(img_NZ_rgb)
# plt.show()