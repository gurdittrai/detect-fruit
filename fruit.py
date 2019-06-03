<<<<<<< HEAD
import sys
import ntpath

#image
import numpy as np
import cv2
from matplotlib import pyplot as plt

from itertools import product

def example(r):
    #gray values
    gvalues = np.unique(r)

    #d=1 and q=horizontal or vertical
    h = np.zeros_like(r)
    v = np.zeros_like(r)

    for p in product(gvalues,gvalues):
        #horizontal
        for i in xrange(1,r.shape[0]+1):
            #avoid edges
            j = i-1 if ((i-1)>=0) else 0
            h[p[0],p[1]] += pairs(p,r[j:i])

        #vertical
        for i in xrange(1,r.shape[1]+1):
            #avoid edges
            j = i-1 if ((i-1)>=0) else 0
            v[p[0],p[1]] += pairs(p,r[:,i-1])
    print(h)
    print(v)

# "On-tree fruit recognition using texture properties and fruit"
# appendix 7
def pairs(p,arr):
    #count number of pairs
    count = 0
    #add useless columns for pair check
    arr = np.append(-1,arr)
    arr = np.append(arr,-1)
    for i in xrange(1,arr.shape[0]-1):
        #match adj pairs
        if arr[i] == p[0]:
            if arr[i-1] == p[1]:
                count += 1
            if arr[i+1] == p[1]:
                count += 1
    return count


#example
R = np.asarray([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])
example(R)
=======
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


fruits = ["banana", "apple", "pear", "strawberry", "tomato", "Bell pepper"]


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
cv2.imwrite("output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# wait to exit
cv2.waitKey(0)
>>>>>>> e10c53d73d3ce9078a5c1e8ea10ffa823cbadd4a
