import cv2
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
import xml.etree.ElementTree as ET  

def get_new_coords(x,y):
    cropped_x = x - start_x
    cropped_y = y - start_y

    Rx = new_x / (end_x - start_x)
    Ry = new_y / (end_y - start_y)

    x = cropped_x * Rx
    y = cropped_y * Ry

    return x,y

pic = '000001'

#Reading bounding box from .xml
annotation = ET.parse('annotations/{}.xml'.format(pic))
root = annotation.getroot()

old_xmin = int(root.find(".//xmin").text)
old_ymin = int(root.find(".//ymin").text)
old_xmax = int(root.find(".//xmax").text)
old_ymax = int(root.find(".//ymax").text)

#Original image
img = cv2.imread('example_images/{}.jpg'.format(pic))

#Gaussian blur
gauss_image = gaussian_filter(img, sigma = 5)
cv2.rectangle(gauss_image, (old_xmin, old_ymin), (old_xmax, old_ymax), (0,255,0), 2)

#Crop and resize
new_x = img.shape[1]
new_y = img.shape[0]

start_x = random.randint(1, old_xmin)
start_y = random.randint(1, old_ymin)
end_x = random.randint(old_xmax, new_x)
end_y = random.randint(old_ymax, new_y)

cropped_image = img[start_y:end_y, start_x:end_x]

resized_image = cv2.resize(cropped_image, dsize = (new_x, new_y))

new_xmin, new_ymin = get_new_coords(old_xmin, old_ymin)
new_xmax, new_ymax =  get_new_coords(old_xmax, old_ymax)

cv2.rectangle(resized_image, (round(new_xmin), round(new_ymin)), (round(new_xmax), round(new_ymax)), (0,255,0), 2)

#HSV color space
HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.rectangle(HSV_image, (old_xmin, old_ymin), (old_xmax, old_ymax), (0,255,0), 2)

#Saturation
saturated_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
saturated_image[...,1] = saturated_image[...,1]*2
saturated_image = cv2.cvtColor(saturated_image, cv2.COLOR_HSV2BGR)

cv2.rectangle(saturated_image, (old_xmin, old_ymin), (old_xmax, old_ymax), (0,255,0), 2)

#Brightness
bright_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bright_image[...,2] = bright_image[...,2]*0.3
bright_image = cv2.cvtColor(bright_image, cv2.COLOR_HSV2BGR)

cv2.rectangle(bright_image, (old_xmin, old_ymin), (old_xmax, old_ymax), (0,255,0), 2)

cv2.rectangle(img, (old_xmin, old_ymin), (old_xmax, old_ymax), (0,255,0), 2)

#Resize all images for display
img = cv2.resize(img, (0,0), fx=0.3, fy=0.3) 
gauss_image = cv2.resize(gauss_image, (0,0), fx=0.3, fy=0.3) 
resized_image = cv2.resize(resized_image, (0,0), fx=0.3, fy=0.3) 
HSV_image = cv2.resize(HSV_image, (0,0), fx=0.3, fy=0.3) 
saturated_image = cv2.resize(saturated_image, (0,0), fx=0.3, fy=0.3) 
bright_image = cv2.resize(bright_image, (0,0), fx=0.3, fy=0.3) 

#Display
cv2.imshow('img', img)
cv2.imshow('gauss_img', gauss_image)
cv2.imshow('resized_image', resized_image)
cv2.imshow('HSV_image', HSV_image)
cv2.imshow('saturation_image', saturated_image)
cv2.imshow('brightness_image', bright_image)

cv2.waitKey(-1)