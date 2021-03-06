import io
import sys
import os

#Import the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

import cv2
import numpy as np
from PIL import Image, ImageDraw

if len(sys.argv) != 2:
    print("Please enter one file path")
    exit(0)

#crendentail_path = "D:\\Code\\side\\googleapi-vision\\googleapi-service-file.json"
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = crendentail_path

path = sys.argv[1]

#list of fruits that will be found
fruits = ["banana", "apple", "pear", "strawberry", "tomato", "bell pepper", "granny smith", "red apple"]

fruitType = ""

#Instantiates a client
client = vision.ImageAnnotatorClient()

#The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(__file__),
    path)

#Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

#Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

#**************#
#Here are the labels for what type of fruit is found in the image
#**************#
for label in labels:
    #print(label.description, label.score)
    if label.description.lower() in fruits:
        fruitType = label.description.lower()
        if label.description.lower() != "apple":
            break
    if fruitType == "apple" and (label.description.lower() == "green" or label.description.lower() == "red"):
        if label.description.lower() == "green":
            fruitType = "granny smith"
        else:
            fruitType = "red apple"
        break;


class coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def showimg(name, img):
    #size
    h = np.size(img, 0)
    w = np.size(img, 1)

    #fit to window
    aratio = w/float(h)
    h = 300
    w = int(h * aratio)


    #window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, img)

def draw_boxes(image, vertices):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    draw.polygon([
        vertices[0].x, vertices[0].y,
        vertices[1].x, vertices[1].y,
        vertices[2].x, vertices[2].y,
        vertices[3].x, vertices[3].y], None, 'blue')

    #image.show()
    return image

#get contours
def getcontours(rawimg, img, techname):
    #copies
    mask = np.zeros_like(rawimg, dtype=np.uint8)

    #get contour CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1
    #_,
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lCon = None
    lArea = 0
    for contour in contours:
        #contour parameters
        epsilion = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilion, True)

        #get area
        area = cv2.contourArea(contour)

        #get largest
        if area > lArea:
            lArea = area
            lCon = contour

    #largest contour area
    epsilion = 0.01 * cv2.arcLength(lCon, True)
    approx = cv2.approxPolyDP(lCon, epsilion, True)
    mask = cv2.drawContours(mask, [approx], 0, (255,255,255), 3) #, cv2.FILLED)

    #close shape
    cv2.fillPoly(mask, pts = [approx], color=(255,255,255))

    #mask
    mask = cv2.bitwise_not(mask)

    #showimg("mask",mask)

    return mask

def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    im = Image.open(path)
    width, height = im.size
    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        if object_.name.lower() == "fruit" or object_.name.lower() in fruits:
            #object id confidence
            print('\n{} (confidence: {})'.format(object_.name, object_.score))
            print(fruitType)
            vertices = []

            for vertex in object_.bounding_poly.normalized_vertices:
                vertices.append(coord((vertex.x * width), (vertex.y * height)))
            # draw_boxes(im, vertices)
            im_crop = crop_img(im, vertices)
            #im_crop.show()
            edge_detect(im_crop)
            break

def crop_img(im, vertices):
    area = (vertices[0].x-10, vertices[0].y-10, vertices[2].x+50, vertices[2].y+10)
    im_crop = im.crop(area)
    #im_crop.show()
    return im_crop

def LABConversion(img):
    #dimensions
    height, width = img.shape[:2]

    #covert image to Lab
    labImg=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    testimg = labImg.copy()
    L, a, b = cv2.split(labImg)

    #colours
    green = 0.0
    yellow = 0.0
    brown = 0.0
    red = 0.0

    #yellow
    low = np.array([1,112,176])
    upp = np.array([254,145,255])

    #yellow layer
    yellowimg = np.zeros_like(img)
    yellowimg = cv2.inRange(labImg, low, upp)

    #get count of pixels
    yellow = yellow + cv2.countNonZero(yellowimg)
    print "yellow", yellow

    #remove pixels from image
    cv2.normalize(labImg, labImg, dtype=cv2.CV_8UC1 ,mask=yellowimg)

    #green
    low = np.array([1,0,0])
    upp = np.array([254,120,255])

    #green layer
    greenimg = np.zeros_like(img)
    greenimg = cv2.inRange(labImg, low, upp)

    #get count of pixels
    green = green + cv2.countNonZero(greenimg)
    print "green", green

    #remove pixels from image
    cv2.normalize(labImg, labImg, dtype=cv2.CV_8UC1 ,mask=greenimg)

    #showimg("no green", labImg)

    #brown
    low = np.array([60,40,30])
    upp = np.array([185,155,195])


    #brown layer
    brownimg = np.zeros_like(img)
    brownimg = cv2.inRange(labImg, low, upp)

    #get count of pixels
    brown = brown + cv2.countNonZero(brownimg)
    print "brown", brown

    #remove pixels from image
    cv2.normalize(labImg, labImg, dtype=cv2.CV_8UC1 ,mask=brownimg)

    #showimg("no brown", labImg)

    #red
    low = np.array([13,35,20])
    upp = np.array([180,210,255])

    #red layer
    redimg = np.zeros_like(img)
    redimg = cv2.inRange(labImg, low, upp)

    #get count of pixels
    red = red + cv2.countNonZero(redimg)
    print "red", red

    #remove pixels from image
    cv2.normalize(labImg, labImg, dtype=cv2.CV_8UC1 ,mask=redimg)
    #showimg("no red", labImg)

    #total
    total = green + yellow + brown + red
    #no colours detected
    if (total == 0):
        return -1

    #get colour percentages
    greenPer = (green/total)*100
    yellowPer = (yellow/total)*100
    brownPer = (brown/total)*100
    redPer = (red/total)*100

    #show percentages
    print "yellow: ", yellowPer
    print "green: ", greenPer
    print "brown: ", brownPer
    print "red: ", redPer

    #calculate ripeness
    #*****************
    #Each fruit catergory will have a different ripeness evaluation.
    #It will depend on the fruitType
    #*****************
    if fruitType == "banana":
        if brownPer >= 20:
            if brownPer >= 50:
                return 4
            else:
                return 3
        elif greenPer >= 20:
            if greenPer >= 50:
                return 0
            else:
                return 1
        elif yellowPer > 50:
            return 2
        else:
            return 3
    elif fruitType == "granny smith":
        if brownPer >= 20:
            if brownPer >= 35:
                return 4
            else:
                return 3
        elif greenPer >= 75:
            if greenPer >= 50:
                return 2
        elif yellowPer > 30:
            return 3
        else:
            return 4
    elif fruitType == "red apple" or fruitType == "apple":
        if brownPer >= 20:
            return 4
        elif greenPer >= 20:
            if greenPer >= 50:
                return 0
            else:
                return 1
        elif redPer > 70:
            return 2
        else:
            return 3
    elif fruitType == "strawberry":
        if brownPer >= 20:
            if brownPer >= 50:
                return 4
            else:
                return 3
        elif greenPer >= 20:
            if greenPer >= 50:
                return 0
            else:
                return 1
        elif yellowPer > 50:
            return 2
        else:
            return 3
    elif fruitType == "tomato":
        if brownPer >= 20:
            if brownPer >= 50:
                return 4
            else:
                return 3
        elif greenPer >= 20:
            if greenPer >= 50:
                return 0
            else:
                return 1
        elif yellowPer > 50:
            return 2
        else:
            return 3
    elif fruitType == "pear":
        if brownPer >= 30:
            if brownPer >= 30:
                return 4
            else:
                return 3
        elif greenPer >= 20:
            if greenPer >= 50:
                return 0
            else:
                return 1
        elif yellowPer > 35:
            return 3
        else:
            return 3
    elif fruitType == "bell pepper":
        if brownPer >= 20:
            if brownPer >= 50:
                return 4
            else:
                return 3
        elif greenPer >= 20:
            if greenPer >= 50:
                return 0
            else:
                return 1
        elif yellowPer > 50:
            return 2
        else:
            return 3
    #defult case
    return -1


def rmvBackground_minor(img):
    #use this if loading in bgr
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img
    #image hsv
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    #detect colour space
    lower_white = np.array([0,100,120])
    upper_white = np.array([255,255,255])

    #mask
    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    #mask on rgb img
    noback = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return noback

def rmvBackground(img):
    #image rgb
    img_rgb = img
    #image hsv
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    #if fruitType == "apple":


    #colour values for green
    G_lower = np.array([28,46,45])
    G_upper = np.array([70,255,255])

    #yellow values
    Y_lower = np.array([18,85,0])
    Y_upper = np.array([28,255,255])

    #brown values
    B_lower = np.array([2,20,20])
    B_upper = np.array([75,255,150])

    #red values
    R_lower = cv2.inRange(img_hsv, (0,50,20), (5,225,225))
    R_upper = cv2.inRange(img_hsv, (175,50,20), (185,225,225))

    #green mask
    green_mask = cv2.inRange(img_hsv, G_lower, G_upper)
    green_mask_inv = cv2.bitwise_not(green_mask)

    #yellow mask
    yellow_mask = cv2.inRange(img_hsv, Y_lower, Y_upper)
    yellow_mask_inv = cv2.bitwise_not(yellow_mask)

    #brown mask
    brown_mask = cv2.inRange(img_hsv, B_lower, B_upper)
    brown_mask_inv = cv2.bitwise_not(brown_mask)

    #red mask
    red_mask = cv2.bitwise_or(R_lower, R_upper)
    red_mask_inv = cv2.bitwise_not(red_mask)

    #Combine masks into one picture to get to total banana
    fruit = cv2.bitwise_and(img_hsv, img_hsv, mask=green_mask+yellow_mask+brown_mask+red_mask)
    background = cv2.bitwise_and(img_hsv, img_hsv, mask=brown_mask_inv+yellow_mask_inv+green_mask_inv+red_mask_inv)
    #convert the picure the the background removed back to rgb for colour space LAB analysis
    fruit = cv2.cvtColor(fruit, cv2.COLOR_HSV2BGR)
    #showimg('background removed', fruit)
    return fruit

def edge_detect(image):
    ###########################
    # prep for edge detection #
    ###########################

    #covert to numpy array
    im = np.array(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #convert grayscale
    grayimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #remove background
    noback = rmvBackground_minor(im)

    #improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgCLAHE = clahe.apply(grayimg)

    #grayscale
    nobackgray = cv2.cvtColor(noback, cv2.COLOR_RGB2GRAY)
    nobackgray = clahe.apply(nobackgray)

    #blur to remove small details
    blurimg = cv2.GaussianBlur(nobackgray, (5, 5), 0)

    ##################
    # edge detection #
    ##################

    #using laplacian
    ddepth = 3
    kernel_size = 3
    imglap = cv2.Laplacian(blurimg, ddepth=cv2.CV_8U,ksize = 3)

    #perform morphology to close gaps
    se = np.ones((7,7), dtype='uint8')
    image_close_lap = cv2.morphologyEx(imglap, cv2.MORPH_TOPHAT, se) # (MORPH_CLOSE MORPH_TOPHAT)

    #getting contours
    mask = getcontours(im, image_close_lap, "Laplacian")

    #remove background
    img = rmvBackground(im)

    cv2.normalize(im, img, dtype=cv2.CV_8UC1 ,mask=mask)

    #apply mask
    #showimg('img',img)
    ###################
    # judging ripness #
    ###################

    #possible ripe values
    ans = [fruitType + " is very unripe", fruitType + " is unripe", fruitType + " is ripe", fruitType + " is overripe", fruitType + " is very overripe"]
    ripe_value = LABConversion(img)
    if (ripe_value == -1):
        print "no colour detected"
    else:
        print ans[ripe_value]
    #cv2.waitKey(0)

#locate the fruit in the image
localize_objects(path)
