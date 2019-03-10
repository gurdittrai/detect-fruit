import sys
import ntpath

#image
import numpy as np
import cv2
from matplotlib import pyplot as plt

from itertools import product

def cmpimgs(img1,img2):
    plt.subplot(121),plt.imshow(img1,cmap = 'gray')
    plt.title('Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2,cmap = 'gray')
    plt.title('Image 2'), plt.xticks([]), plt.yticks([])

    plt.show()

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

# red color detect
# red = colourDetect(image)
# img_red = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)
# cv2.imshow('red', img_red)

# gray image
img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# apply blur to remove details
img = cv2.GaussianBlur(img,(9,9),cv2.BORDER_DEFAULT)

# get circles
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,120,param1=50,param2=30, minRadius=100, maxRadius=250)
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
    cv2.imshow('detected circles', cropimg)
    cv2.waitKey(0)

    # get red area
    redimg = colourDetect(cropimg)
    redcount = 0
    emptypixel = np.matrix('0 0 0')
    for i in xrange(redimg.shape[0]):
        for j in xrange(redimg.shape[1]):
            # get the pixel value
            value = np.array_equal(np.matrix(redimg[i][j]), emptypixel)
            # print value
            if (value):
                redcount += 1

    #threshold
    total = redimg.shape[0] * redimg.shape[1]
    percent = (redcount*100) / total

    # count
    print percent
    if (percent > 35):
        count += 1

print str(count) + ' circles'
cv2.imshow('detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# wait to exit
cv2.waitKey(0)