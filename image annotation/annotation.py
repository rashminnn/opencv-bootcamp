import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np

# Read in an image
image = cv.imread("Apollo_11_Launch.jpg", cv.IMREAD_COLOR)

imageLine = image.copy()

# The line starts from (200,100) and ends at (400,100)
# The color of the line is YELLOW (Recall that OpenCV uses BGR format)
# Thickness of line is 5px
# Linetype is cv2.LINE_AA

# cv.line(imageLine, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv.LINE_AA)

# cv.circle(image , (600,300) ,100 , (0,255,0) , thickness=5 , lineType=cv.LINE_AA)

# plt.imshow(image[:,:,::-1])

cv.rectangle(image, (500, 100), (700, 600), (255, 0, 255), thickness=5, lineType=cv.LINE_8)

# Display the image

imageText = image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontScale = 2.3
fontFace = cv.FONT_HERSHEY_PLAIN
fontColor = (0, 255, 0)
fontThickness = 2

cv.putText(imageText, text, (200, 700), fontFace, fontScale, fontColor, fontThickness, cv.LINE_AA)

# Display the image
plt.imshow(imageText[:, :, ::-1])


# Display the image
# plt.imshow(imageLine[:,:,::-1])

# Display the original image
# plt.imshow(image[:, :, ::-1])

plt.show()