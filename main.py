#!/usr/bin/env python

import math
import sys
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread(sys.argv[1], 0)
# w, h = template.shape[::-1]

# img = img[1412:1412+308, 572:572+241]

# ret, img = cv2.threshold(img,210,250,cv2.THRESH_TRUNC)

cells_w = 14
cells_h = 18

# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

# plt.imshow(img, cmap='gray'),plt.show()
# plt.imshow(opening, cmap='gray'),plt.show()

img = cv2.resize(img, (math.floor(img.shape[1] * 2), math.floor(img.shape[0] * 2)), cv2.INTER_CUBIC);

def apply_threshold(img, level):
    # ret, img = cv2.threshold(img,level,255,cv2.THRESH_BINARY)

    # ret, img = cv2.threshold(img,210,255,cv2.THRESH_TRUNC)

    # img = cv2.blur(img,(3,3))

    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    kernel = np.ones((3,3),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

    # img = cv2.resize(img, (math.floor(img.shape[1] * 0.5), math.floor(img.shape[0] * 0.5)), cv2.INTER_CUBIC);
    return img

# img = apply_threshold(img, 155)
plt.imshow(img, cmap='gray'),plt.show()

# def nothing(x):
#     pass

# cv2.namedWindow('image')

# # create trackbars for color change
# cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('G','image',0,255,nothing)
# cv2.setTrackbarPos('R','image', 178)

# while(1):
#     # get current positions of four trackbars
#     a = cv2.getTrackbarPos('R','image')
#     b = cv2.getTrackbarPos('G','image')

#     img2 = apply_threshold(img)
#     # print(a, b)

#     cv2.imshow('image',img2)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()
# sys.exit()

cell_w = img.shape[1] / cells_w
cell_h = img.shape[0] / cells_h


print(img.shape, cell_w, cell_h)

output = [[{'type': 'unknown'} for x in range(cells_w)] for y in range(cells_h)]

size = math.floor(cell_w * 0.70)

for cell_y in range(0, cells_h):
    for cell_x in range(0, cells_w):
        cell = img[
            math.floor(cell_y * cell_h):math.floor(cell_y * cell_h + cell_h),
            math.floor(cell_x * cell_w):math.floor(cell_x * cell_w + cell_w)
        ]
        cell = cell[1:-1, 1:-1]

        if np.sum(cell < (255//2))/(cell.shape[0]*cell.shape[1]) > 0.5:
            output[cell_y][cell_x] = {'type': 'blank', 'x': cell_x, 'y': cell_y};
        else:
            height = cell.shape[0]
            width = cell.shape[1]
            pad_h = (height - size) // 2
            pad_w = (width - size) // 2
            cell = cell[pad_h:size+pad_h, pad_w:size+pad_w]

            # print(cell.shape)
            output[cell_y][cell_x] = {'type': 'text', 'data': cell, 'x': cell_x, 'y': cell_y};

            ret,thresh = cv2.threshold(cell,180,255,0)
            thresh = cv2.bitwise_not(thresh)
            # plt.imshow(thresh, cmap='gray'),plt.show()
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(cell,(x,y),(x+w,y+h),(0,255,0),2)
            output[cell_y][cell_x]['cropped'] = cell[y-1:y+h+2,x-1:x+w+2]
            # cv2.drawContours(cell, contours, -1, (0,255,0), 3)
            # plt.imshow(cell, cmap='gray'),plt.show()
            # plt.imshow(output[cell_y][cell_x]['cropped'], cmap='gray'),plt.show()

# sys.exit()

lines = [];

current = [];
# Find all the columns of text
for cell_x in range(0, cells_w):
    for cell_y in range(0, cells_h):
        if output[cell_y][cell_x]['type'] == 'text':
            # print(output[cell_y][cell_x]['data'].shape)
            current.append(output[cell_y][cell_x])
        else:
            if len(current) > 1:
                lines.append(current)
            current = []
    if len(current) > 1:
        lines.append(current)
    current = []

# Find all the lines of text
for cell_y in range(0, cells_h):
    for cell_x in range(0, cells_w):
        if output[cell_y][cell_x]['type'] == 'text':
            current.append(output[cell_y][cell_x])
        else:
            if len(current) > 1:
                lines.append(current)
            current = []
    if len(current) > 1:
        lines.append(current)
    current = []


# print(lines)


for line in lines:
    # print(line)
    sequence = tuple(cell['data'] for cell in line)
    # print(sequence)
    line_img = np.concatenate(sequence, axis=1)

    sequence = tuple(cell['cropped'] for cell in line)
    sequence_w = 5 + sum(cell.shape[1] + 5 for cell in sequence)
    sequence_h = 10 + max(cell.shape[0] for cell in sequence)

    # create a new array with a size large enough to contain all the images
    final_image = np.ones((sequence_h, sequence_w), dtype=np.uint8) * 255

    current_x = 5 # keep track of where your current image was last placed in the x coordinate
    for cell in sequence:
        # add an image to the final array and increment the x coordinate

        # center vertically our image
        center_pad = (sequence_h-cell.shape[0]) // 2

        final_image[center_pad:center_pad+cell.shape[0],current_x:current_x+cell.shape[1]] = cell
        current_x += cell.shape[1] + 5

    # plt.imshow(final_image, cmap='gray'),plt.show()
    # plt.imshow(line_img, cmap='gray'),plt.show()

    config = ("-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -l fra --oem 0 --psm 8")
    text = pytesseract.image_to_string(line_img, config=config)
    text2 = pytesseract.image_to_string(final_image, config=config)
    ambiguous = text != text2
    if len(text) == len(line):
        for i, cell in enumerate(line):
            if 'letter' in cell and cell['letter'] != text[i]:
                print('ambiguous cell', cell['letter'], text[i], text)
                ambiguous = True
                del cell['letter']
            else:
                cell['letter'] = text[i]
                cell['part_of'] = text
                cell['part_of_img'] = line_img
        print(text, text2)
    else:
        ambiguous = True
        print('ambiguous line', text, text2)

    if ambiguous:
        plt.imshow(line_img, cmap='gray')
        plt.title(text)
        plt.show()
        plt.imshow(final_image, cmap='gray')
        plt.title(text2)
        plt.show()


plt.imshow(img, cmap='gray')

for line in output:
    for cell in line:
        if cell['type'] == 'blank':
            print('-', end='')
        elif 'letter' in cell:
            print(cell['letter'], end='')
            plt.annotate(cell['letter'], xy=(cell['x'] * cell_w, cell['y'] * cell_h))
        else:
            print('?', end='')
    print()
