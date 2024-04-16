import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

imagefiles = glob.glob(f"boat{os.sep}*")
imagefiles.sort()

images = []
for filename in imagefiles:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

# Stitch images
stitcher = cv2.Stitcher_create()
status, stitched_image = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    # Convert stitched image to RGB format
    stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
    
    # Find non-zero pixels (foreground) to determine the bounding box
    gray_stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_stitched_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Crop the stitched image
    cropped_image = stitched_image_rgb[y:y+h, x:x+w]
    
    # Display the cropped image
    plt.figure(figsize=[10, 5])
    plt.imshow(cropped_image)
    plt.axis("off")
    plt.title("Cropped Panorama")
    plt.show()
else:
    print("Stitching failed:", status)
