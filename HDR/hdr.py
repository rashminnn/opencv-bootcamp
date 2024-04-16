import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def readImagesAndTimes():
    # List of file names
    filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

    # List of exposure times
    times = np.array([1 / 30.0, 0.25, 2.5, 15.0], dtype=np.float32)

    # Read images
    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images, times

# Read images and exposure times
images, times = readImagesAndTimes()

# Align Images
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# Find Camera Response Function (CRF)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

# Plot CRF
x = np.arange(256, dtype=np.uint8)
y = np.squeeze(responseDebevec)

ax = plt.figure(figsize=(30, 10))
plt.title("Debevec Inverse Camera Response Function", fontsize=24)
plt.xlabel("Measured Pixel Value", fontsize=22)
plt.ylabel("Calibrated Intensity", fontsize=22)
plt.xlim([0, 260])
plt.grid()
plt.plot(x, y[:, 0], "r", x, y[:, 1], "g", x, y[:, 2], "b")

# Merge images into an HDR linear image
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

# Tonemap using Drago's method to obtain 24-bit color image
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago

plt.figure(figsize=(20, 10));plt.imshow(np.clip(ldrDrago, 0, 1));plt.axis("off")

cv2.imwrite("ldr-Drago.jpg", 255*ldrDrago[:, :, ::-1])
print("saved ldr-Drago.jpg")

# Tonemap using Reinhard's method to obtain 24-bit color image
print("Tonemaping using Reinhard's method ... ")
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)

plt.figure(figsize=(20, 10));plt.imshow(np.clip(ldrReinhard, 0, 1));plt.axis("off")

cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
print("saved ldr-Reinhard.jpg")

# Tonemap using Mantiuk's method to obtain 24-bit color image
print("Tonemaping using Mantiuk's method ... ")
tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = 3 * ldrMantiuk

plt.figure(figsize=(20, 10));plt.imshow(np.clip(ldrMantiuk, 0, 1));plt.axis("off")

cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
print("saved ldr-Mantiuk.jpg")

# Tonemap using Durand's method to obtain 24-bit color image
print("Tonemapping using Durand's method ... ")
tonemapDurand = cv2.xphoto.createTonemapDurand(1.5, 4, 1.0, 1, 1)
ldrDurand = tonemapDurand.process(hdrDebevec)

plt.figure(figsize=(20, 10))
plt.imshow(np.clip(ldrDurand, 0, 1))
plt.axis("off")
cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)
print("Saved ldr-Durand.jpg")

plt.show()


# Drago's method: This method compresses the dynamic range of HDR images while preserving local contrast. It tends to produce natural-looking results with enhanced details.

# Durand's method: Durand's tone mapping method adapts to local contrast variations in the image. It combines bilateral filtering with luminance and contrast adjustments to create visually pleasing results.

# Mantiuk's method: Mantiuk's tone mapping method enhances image details and global contrast while compressing the dynamic range. It is suitable for creating visually striking images with enhanced details.

# Reinhard's method: Reinhard's tone mapping method is based on global luminance and contrast adjustments. It aims to preserve the overall appearance of the scene while compressing the dynamic range for display.