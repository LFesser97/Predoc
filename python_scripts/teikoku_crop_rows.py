# -*- coding: utf-8 -*-
"""
teikoku_crop_rows.py

Created on Thu Sep  8 08:47:30 2022

@author: Lukas
"""

# import statements
import os
from PIL import Image
import json
import cv2
import numpy as np


# path dependencies
current_path = os.getcwd()

image_path = os.path.join(current_path, 'scans')
json_path = os.path.join(current_path, 'row_coordinates')
output_path = os.path.join(current_path, 'cropped_rows')


# rename this
def find_rect(image_source, rect):
    """
    
    [docstring goes here]

    """
    img = cv2.imread(image_source)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # img_crop will the cropped rectangle, img_rot is the rotated image
    img_crop, img_rot = crop_rect(img, rect)
    return img_crop


# function to crop a given rectangle
def crop_rect(img, rect):
    """
    
    [docstring goes here]

    """
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot
    

# main function
def main(image_path, json_path, output_path):
    """
    for every image and for every json in an input folder,
    crops all rows in an input json from the corresponding image

    Parameters
    ----------
    image_path : string
        file path of the input image directory
        
    json_path : string
        file path of the input json directory
        
    output_path : string
        file path of the output directory
        
    Returns
    -------
    None (cropped images are saved in the output folder)

    """
    # iterate over all images in the input folder
    for image_name in os.listdir(image_path):
        name = image_name.split(".")[0]
        json_name = name + '_0.json'
        
        # load row coordinates from json
        with open(os.path.join(json_path, json_name)) as f:
            row_list = json.load(f)
                        
            # iterate over all rows in input json
            row_count = 0
            for rect in row_list:
                row_image = find_rect(os.path.join(image_path, image_name), rect)
                
                # save cropped image
                output_name = os.path.join(output_path, name + '_row_' + str(row_count) + '.png')
                cv2.imwrite(output_name, row_image)
                
                row_count += 1
    

if __name__ == "__main__":
    main(image_path, json_path, output_path)