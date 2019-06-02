import io
import os

#Import the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw

#Instantiates a client
client = vision.ImageAnnotatorClient()

#The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(__file__),
    './mytestimgs/GreenBanana04.jpg')

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

def draw_boxes(image, x1, x2, y1 ,y2):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    draw.polygon([
        x1, y1,
        x2, y1,
        x2, y2,
        x1, y2], None, 'blue')

    image.show()
    return image

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
    im = Image.open('./mytestimgs/GreenBanana04.jpg')
    width, height = im.size
    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        count = 0

        for vertex in object_.bounding_poly.normalized_vertices:
            if count == 0:
                x1 = vertex.x
                y1 = vertex.y
            elif count == 2:
                x2 = vertex.x
                y2 = vertex.y
            count = count + 1
            print(' - ({}, {})'.format(vertex.x, vertex.y))
        draw_boxes(im, (x1 * width), (x2 * width), (y1 * height), (y2 * height))

localize_objects('./mytestimgs/GreenBanana04.jpg')
