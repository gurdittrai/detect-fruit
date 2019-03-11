# Name: Andrew Maklingham, Gurditt Rai
# Date: Sunday, March 10, 2019
# Program Description
# It detects all the fruits in the provided image

import sys
import ntpath

#image
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Colour Dectection
#Simple colour detection will be handled within this function.
def colourDetect(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #detect colour space
    lower_red = np.array([105,10,10])
    upper_red = np.array([255,255,255])

    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# img path and name
fp = sys.argv[1]
name = ntpath.basename(fp)

# open img
image = cv2.imread(fp, 1)

# gray image
img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# apply blur to remove details
img = cv2.GaussianBlur(img,(9,9),cv2.BORDER_DEFAULT)

# get circles
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,120,param1=50,param2=30,minRadius=50, maxRadius=150)
# circles_round = np.uint16(np.around(circles))

count = 0
for i in circles[0,:]:
    # circle properties
    center = (i[0],i[1])
    radius = i[2]

    # draw the outer circle
    # cv2.circle(image,center,radius,(0,255,0),2)

    # crop properties
    x0 = int(i[0]-radius)
    x1 = int(i[0]+radius)
    y0 = int(i[1]-radius)
    y1 = int(i[1]+radius)
    cropimg = image[y0:y1,x0:x1]

    # draw cropped image
    # cv2.imshow('cropped circles', cropimg)

    # get red area
    redimg = colourDetect(cropimg)

    #threshold to make comparsion easier
    ret,thresh1 = cv2.threshold(redimg,105,255,cv2.THRESH_BINARY)

    # count and compare
    redcount = 0
    emptypixel = np.matrix('0 0 0')
    for i in xrange(redimg.shape[0]):
        for j in xrange(redimg.shape[1]):
            # get the pixel value
            value = np.array_equal(np.matrix(thresh1[i][j]), emptypixel)
            # remove background pixel from count
            if (value == False):
                redcount += 1

    #threshold for red
    total = redimg.shape[0] * redimg.shape[1]
    percent = (redcount*100) / total

    # count
    print percent
    if (percent > 35):
        count += 1
        cv2.circle(image,center,radius,(0,255,0),2)


title = str(count) + ' Fruits Detected'
cv2.imshow(title, image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# wait to exit
cv2.waitKey(0)