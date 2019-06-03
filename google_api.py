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

crendentail_path = "D:\\Code\\side\\googleapi-vision\\googleapi-service-file.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = crendentail_path
 
path = sys.argv[1]

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

##print('Lables')
##for label in labels:
    #print(label.description, label.score)

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

    image.show()
    return image

#get contours
def getcontours(rawimg, img, techname):
    #copies
    mask = np.zeros_like(rawimg, dtype=np.uint8)

    #get contour CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    fruits = ["banana", "apple", "pear", "strawberry", "tomato", "bell pepper"]

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        if object_.name.lower() == "fruit":
            #object id confidence
            print('\n{} (confidence: {})'.format(object_.name, object_.score))
            # print('Normalized bounding polygon vertices: ')
            vertices = []

            for vertex in object_.bounding_poly.normalized_vertices:
                vertices.append(coord(vertex.x * width,vertex.y * height))
                # print(' - ({}, {})'.format(vertex.x, vertex.y))
            # draw_boxes(im, vertices)
            im_crop = crop_img(im, vertices)
            im_crop.show()
            edge_detect(im_crop)

def crop_img(im, vertices):
    area = (vertices[0].x, vertices[0].y, vertices[2].x, vertices[2].y)
    im_crop = im.crop(area)
    im_crop.show()
    return im_crop

def edge_detect(image):
    ###########################
    # prep for edge detection #
    ###########################

    #covert to numpy array
    im = np.array(image)

    #convert grayscale
    grayimg = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    #improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgCLAHE = clahe.apply(grayimg)

    #blur to remove small details
    blurimg = cv2.GaussianBlur(imgCLAHE, (5, 5), 0)

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

    #apply mask
    cv2.normalize(im, im, dtype=cv2.CV_8UC1 ,mask=mask)
    showimg('im',im)
    cv2.waitKey(0)

localize_objects(path)




#show original and resulting images
# showimg("rawimg", rawimg)
# showimg("banana_area", banana_area)


