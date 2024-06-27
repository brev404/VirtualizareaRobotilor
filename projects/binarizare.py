import cv2 as cv
import numpy as np
 
# Load the image
img = cv.imread("D:/files/projects/Campus/gui_app/projects/data/poza.png")
# img = cv.resize(img, (640, 480))
if img is None:
    print("Error: File not found")
    exit(0)
 
cv.imshow('Input Image', img)
 
# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
# Convert image to binary
_, bw = cv.threshold(gray, 0.5, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
 
kernel1 = np.ones((13,13))
# kernel2 = np.ones((11,11))
# dilated2 = cv.dilate(bw, kernel2)
# eroded2 = cv.erode(dilated2,kernel1)
eroded2 = cv.dilate(bw, kernel1)
cv.imshow('contururi', eroded2)
 
# Define the region of interest (ROI) using a binary mask
roi_mask = np.zeros_like(bw)
# roi_mask[94:403, 301:640] = 1
roi_mask[100:480, 318:640] = 1
# roi_mask[:480, :640] = 1
cv.imshow('ROI Mask', roi_mask * 255)
 
# Apply the ROI mask to the binary image
bw_roi = cv.bitwise_and(eroded2, eroded2, mask=roi_mask)
 
# Find contours in the thresholded image with the ROI applied
contours, _ = cv.findContours(bw_roi, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
 
centre_unghiuri = {}
arii = []
for i, c in enumerate(contours):
 
    # Calculate the area of each contour
    area = cv.contourArea(c)
 
    # Ignore contours that are too small or too large
    if area < 3000 or 50000 < area:
        continue
    else:
        arii.append(area)
 
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)
   
    # Retrieve the key parameters of the rotated bounding box
    center = (int(rect[0][0]), int(rect[0][1]))
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = int(rect[2])
   
    if width < height:
        angle = 90 - angle
    else:
       
        angle = 180 - angle
 
    # print('arie: ', area, 'unghi: ', angle, 'centru: ', center)
    centre_unghiuri[angle] = center
 
    label = str(angle)
    cv.putText(img, label, (center[0] - 50, center[1]),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
    cv.drawContours(img, [box], 0, (0, 0, 255), 2)
 
cv.imshow('Output Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
 
# Save the output image to the current directory
cv.imwrite("D:/files/projects/Campus/gui_app/projects/data/output.png", img)
print(centre_unghiuri)