import cv2
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
import xml.etree.ElementTree as ET  
import os

def get_new_coords(x, y, size_x, size_y, start_x, start_y, end_x, end_y):
    cropped_x = x - start_x
    cropped_y = y - start_y

    Rx = size_x / (end_x - start_x)
    Ry = size_y / (end_y - start_y)

    x = cropped_x * Rx
    y = cropped_y * Ry

    return int(x), int(y)

def read_gt_from_xml(read_name):
    annotation = ET.parse('original_dataset/Annotations/{}.xml'.format(read_name))
    bnd_boxes = annotation.findall('.//bndbox')
    
    gt_boxes = []

    for bnd_box in bnd_boxes:
        gt_box = []
        for coord in bnd_box:
            gt_box.append(int(coord.text))
        gt_boxes.append(gt_box)

    return gt_boxes

def write_gt_to_xml(read_name, write_name, gt_boxes):
    annotation = ET.parse('original_dataset/Annotations/{}.xml'.format(read_name))
    bnd_boxes = annotation.findall('.//bndbox')

    for gt, bnd_box in zip(gt_boxes, bnd_boxes):
        for gt_coord, coord in zip(gt, bnd_box):
            coord.text = str(gt_coord)

    folders = annotation.findall('.//folder')

    for folder in folders:
        folder.text = 'VOC2007'

    filenames = annotation.findall('.//filename')

    for filename in filenames:
        filename.text = '{}.jpg'.format(write_name)

    names = annotation.findall('.//name')

    for name in names:
        name.text = 'smoke'
        
    annotation.write('Annotations/{}.xml'.format(write_name))

def find_boundaries(gt_boxes):
    old_xmin, old_ymin, old_xmax, old_ymax = None, None, None, None
    for i, gt in enumerate(gt_boxes):
        if i == 0:
            old_xmin, old_ymin, old_xmax, old_ymax = gt[0], gt[1], gt[2], gt[3]
        else:
            if gt[0] < old_xmin:
                old_xmin = gt[0]
            if gt[1] < old_ymin:
                old_ymin = gt[1]
            if gt[2] > old_xmax:
                old_xmax = gt[2]
            if gt[3] > old_ymax:
                old_ymax = gt[3]
    return old_xmin, old_ymin, old_xmax, old_ymax

def rand_gaussian_blur(image):
    sigma = random.randint(0, 2)
    image = gaussian_filter(image, sigma)
    return image

def rand_saturation(image):
    saturation = random.uniform(1,1.2)
    saturated_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated_image[...,1] = saturated_image[...,1] * saturation
    image = cv2.cvtColor(saturated_image, cv2.COLOR_HSV2BGR)
    return image

def rand_brightness(image):
    brightness = random.uniform(0.9,1.2)
    bright_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright_image[...,2] = bright_image[...,2] * brightness
    image = cv2.cvtColor(bright_image, cv2.COLOR_HSV2BGR)
    return image

def to_HSV(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image

def crop_and_resize(image, gt_boxes):
    size_x, size_y = image.shape[1], image.shape[0]

    old_xmin, old_ymin, old_xmax, old_ymax = find_boundaries(gt_boxes)

    start_x = random.randint(0, old_xmin)
    start_y = random.randint(0, old_ymin)
    end_x = random.randint(old_xmax, size_x)
    end_y = random.randint(old_ymax, size_y)

    cropped_image = image[start_y:end_y, start_x:end_x]

    image = cv2.resize(cropped_image, dsize = (size_x, size_y))

    return image, size_x, size_y, start_x, start_y, end_x, end_y

def augment_and_write_image(ID, new_ID):
    #gt_boxes boxes from original image
    gt_boxes = read_gt_from_xml(ID)

    #Original image
    image = cv2.imread('original_dataset/JPEGImages/{}.jpeg'.format(ID))

    #Augmentations
    image = rand_gaussian_blur(image)
    image = rand_saturation(image)
    #image = rand_brightness(image)
    image, size_x, size_y, start_x, start_y, end_x, end_y = crop_and_resize(image, gt_boxes)

    shown_image = image.copy()

    #Calculating new gt_boxes
    new_gt_boxes = []
    for gt in gt_boxes:
        old_xmin, old_ymin, old_xmax, old_ymax = gt[0], gt[1], gt[2], gt[3]
        new_xmin, new_ymin = get_new_coords(old_xmin, old_ymin, size_x, size_y, start_x, start_y, end_x, end_y)
        new_xmax, new_ymax =  get_new_coords(old_xmax, old_ymax, size_x, size_y, start_x, start_y, end_x, end_y)
        new_gt_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
        
        cv2.rectangle(shown_image, (new_xmin, new_ymin), (new_xmax, new_ymax), (0,255,0), 2)

    cv2.imwrite('JPEGImages/{}.jpg'.format(new_ID), image)
    write_gt_to_xml(ID, new_ID, new_gt_boxes)

if __name__ == '__main__':
    new_ID = 0
    for filename in os.listdir('original_dataset/JPEGImages'):
        filename = filename.strip('.jpeg')
        exists = os.path.isfile('original_dataset/Annotations/{}.xml'.format(filename))

        print('Augmenting {}'.format(filename))
        if exists:
            for i in range(50):
                new_ID = str(int(new_ID) + 1)
                augment_and_write_image(filename, new_ID)
    
    print('Finished augmenting')

    # print('Faulty images')
    # for jpg_filename in os.listdir('original_dataset/JPEGImages'):
    #     jpg_filename = jpg_filename.strip('.jpeg')

    #     exists = os.path.isfile('original_dataset/Annotations/{}.xml'.format(jpg_filename))
    #     if not exists:
    #         print(jpg_filename)

    # print('Faulty annotations')
    # for xml_filename in os.listdir('original_dataset/Annotations/'):
    #     xml_filename = filename.strip('.xml')

    #     exists = os.path.isfile('original_dataset/JPEGImages/{}.xml'.format(xml_filename))
    #     if not exists:
    #         print(filename)


