import cv2
import numpy as np



panorama_image = cv2.imread('images/360.jpg')

height, width, channels = panorama_image.shape

print(panorama_image.shape)

size = 150

x_curr = size
y_curr = size

horizontally_moved = 0
horizontal = False

last_vertical = 'down'
vertical_up = False
vertical_down = True

pred_box = None

id = 0

while(1):
    id += 1
    if pred_box == None:
        if horizontal:
            x_curr += 5
            horizontally_moved += 5
        elif vertical_up:
            y_curr -= 5
        elif vertical_down:
            y_curr += 5

        if y_curr >= height - size and vertical_down:
            horizontal = True
            vertical_down = False
        elif y_curr <= size and vertical_up:
            horizontal = True
            vertical_up = False
        elif horizontally_moved >= 2 * size and horizontal:
            horizontally_moved = 0
            horizontal = False
            if last_vertical == 'up':
                last_vertical = 'down'
                vertical_down = True
            elif last_vertical == 'down':
                last_vertical = 'up'
                vertical_up = True

    camera_image = panorama_image[y_curr-size:y_curr+size, x_curr-size:x_curr+size]
    cv2.imshow('panorama', camera_image)
    cv2.imwrite('panorama/{}.jpg'.format(id), camera_image)
    cv2.waitKey(10)
