# -*- coding: utf-8 -*-
"""
unskew_scans_via_polygons.py

Created on Wed Sep 14 14:58:32 2022

@author: Lukas
"""

# import packages
import os
import json
import cv2
import numpy as np


# set directories
current_path = os.getcwd()

image_path = os.path.join(current_path, 'scans')
json_path = os.path.join(current_path, 'jsons')
output_path = os.path.join(current_path, 'unskewed_scans')


# Auxilliary functions
def fetch_corners(layout):
    """
    retrieves and rescales the corners of a polygon from the input json
    
    Parameters
    ----------
    layout : json
        input json with the polygon

    Returns
    -------
    corners : list
        list with the 4 corners stored as list [int(x), int(y)]

    """
    img_width = layout['original_width']
    img_height = layout['original_height']
    
    # need this for polygons
    points = layout['value']['points']
    
    # coordinates coming from Label Studio are given as percentages
    def rescale_coordinates(corner, width, height):
        x = corner[0]
        y = corner[1]
        rescaled_x = x * width * 0.01
        rescaled_y = y * height * 0.01
        return [int(rescaled_x), int(rescaled_y)]
        
    corners = [rescale_coordinates(coordinate, img_width, img_height) for coordinate in points]
    return corners


def sort_corners(corners):
    """
    make sure corners are in the right order, i.e.
    top left, top right, bottom right, bottom left

    Parameters
    ----------
    corners : list
        list of the 4 corners as elements [x, y]

    Returns
    -------
    sorted_corners : list
        sorted list of the 4 corners

    """
    corners.sort()
    corners[1:2].sort(key = lambda elem : elem[1])
    corners[3:4].sort(key = lambda elem : elem[1])
    sorted_corners = [[corners[0]], [corners[2]], [corners[3]], [corners[1]]]
    return sorted_corners
    
    
def rotate_and_crop(corners, img):
    """
    crops the text block from the input image img by detecting the smallest 
    rotated rectangle containing it; rotates it, and return the new (sub-)image

    Parameters
    ----------
    corners : list
        list of the 4 corners of the text block
        
    img : array
        the input image as read in by cv2

    Returns
    -------
    warped : array
        the cropped and rotated output image

    """
    # find the smallest rotated rectangle containing the text block
    cnt = np.array(corners)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    width = int(rect[1][0])
    height = int(rect[1][1])
    
    # find the transformation matrix by comparing source and destination corners
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # transform the text block
    warped = cv2.warpPerspective(img, M, (width, height))
    
    # correct rotation if necessary
    if warped.shape[0] < warped.shape[1]:
        warped = np.rot90(warped, 3)

    return warped


# main function
def main(image_path, json_path, output_path):
    """
    crops and unskews all text blocks contained in the json_path folder;
    outputs the resulting images in the output_path folder

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
    None (unskewed cropped images are saved in the output folder)

    """
    # iterate over all images in the input folder
    for image_name in os.listdir(image_path):
        name = image_name.split(".")[0]
        json_name = name + '.json'
        
        # load polygons from json
        with open(os.path.join(json_path, json_name)) as f:
            input_json = json.load(f)
            polygons = input_json[0]['annotations'][0]['result']
            img = cv2.imread(os.path.join(image_path, image_name), 0)
            
            # process the two pages in a scan separately
            i = 0
            for polygon in polygons:
                sorted_corners = sort_corners(fetch_corners(polygon))
                
                # crop and rotate the subimage
                cropped_image = rotate_and_crop(sorted_corners, img)
            
                # save cropped and unskewed image
                output_name = os.path.join(output_path, name + "_" + str(i) + '.png')
                cv2.imwrite(output_name, cropped_image)
                i += 1

                
if __name__ == "__main__":
    main(image_path, json_path, output_path)