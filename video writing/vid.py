import cv2 as cv
import matplotlib.pyplot as plt

source = "race_car.mp4"
cap = cv.VideoCapture(source)

if (cap.isOpened == False):
    print("Error opening video stream or file")

ret, frame = cap.read()

width = int(cap.get(3))
height = int (cap.get(4))
frame_rate = cap.get(cv.CAP_PROP_FPS)
avi = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (width, height))
mp4 = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, (width, height))

while (cap.isOpened()):
    ret,frame = cap.read()

    if ret == True:
        avi.write(frame)
        mp4.write(frame)

    else:
        break

cap.release()
avi.release()
mp4.release()

cv.destroyAllWindows()
