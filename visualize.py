import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def visualize_box(box, image, thick = 5):

    '''
    Visualize the bounding boxes with thick green sides
    Code for Q5 and Q6

    box: the bounding box coordinates, list of np arrays
    image: the 3D np array representing the original image (480 x 640 x 3)
    thick: how thick (how many pixels) the frame is

    This function returns a 3D np array representing the new image with bounding boxes
    '''
    height = image.shape[0]
    width = image.shape[1]
    I_new = np.copy(image)
    t = max(box[0]-thick,0)
    b = min(box[2]+thick,height)
    l = max(box[1]-thick,0)
    r = min(box[3]+thick,width)
    for h in range(t,box[0]):
        I_new[h, l:r, 0] = 0
        I_new[h, l:r, 1] = 255
        I_new[h, l:r, 2] = 0
    for h in range(box[0],box[2]):
        I_new[h, l:box[1], 0] = 0
        I_new[h, l:box[1], 1] = 255
        I_new[h, l:box[1], 2] = 0
        I_new[h, box[3]:r, 0] = 0
        I_new[h, box[3]:r, 1] = 255
        I_new[h, box[3]:r, 2] = 0
    for h in range(box[2],b):
        I_new[h, l:r, 0] = 0
        I_new[h, l:r, 1] = 255
        I_new[h, l:r, 2] = 0
    return I_new

def visualize(I, address = '../data/images/test/yea.jpg'):
    image = Image.fromarray(I)
    image.save(address)
    return