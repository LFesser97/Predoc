# -*- coding: utf-8 -*-
"""
unskew_scans.py

Created on Mon Sep 12 14:37:19 2022

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
    retrieves and rescales the corners of a text box from the input json
    
    Parameters
    ----------
    layout : json
        input json with the coordinates of the 4 corner points

    Returns
    -------
    corners : list
        list with the 4 corners stored as list [int(x), int(y)]

    """
    coordinates = layout[0]['annotations'][0]['result']
    img_width = coordinates[0]['original_width']
    img_height = coordinates[0]['original_height']
    
    # coordinates coming from Label Studio are given as percentages
    def rescale_coordinates(corner, width, height):
        x = corner['value']['x']
        y = corner['value']['y']
        rescaled_x = x * width
        rescaled_y = y * height
        return [int(rescaled_x), int(rescaled_y)]
        
    corners = [rescale_coordinates(coordinate, img_width, img_height) for coordinate in coordinates]
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
    

    Parameters
    ----------
    corners : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    warped : TYPE
        DESCRIPTION.

    """
    cnt = np.array(corners)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # crop_rect
    def crop_rect(img, rect):
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

    img_crop, img_rot = crop_rect(img, rect)
    return img_crop
    

# main function
def main(image_path, json_path, output_path):
    """
    [function purpose goes here]

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
        for i in range(2):
            json_name = name + "_" + str(i) + '.json'
        
            # load corner points from json
            with open(os.path.join(json_path, json_name)) as f:
                layout = json.load(f)
                img = cv2.imread(os.path.join(image_path, image_name), 0)
                sorted_corners = sort_corners(fetch_corners(layout))
                
                # crop and rotate the subimage
                cropped_image = rotate_and_crop(sorted_corners , img)
            
                # save cropped and unskewed image
                output_name = os.path.join(output_path, name + "_" + str(i) + '.png')
                cv2.imwrite(output_name, cropped_image)

                
#if __name__ == "__main__":
#    main(image_path, json_path, output_path)

main(image_path, json_path, output_path)
    
    
# def rotate_and_crop(corners, img):
#     """
    

#     Parameters
#     ----------
#     corners : TYPE
#         DESCRIPTION.
#     img : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     warped : TYPE
#         DESCRIPTION.

#     """
#     cnt = np.array(corners)
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
#     width = int(rect[1][0])
#     height = int(rect[1][1])
#     src_pts = box.astype("float32")
#     dst_pts = np.array([[0, height-1],
#                         [0, 0],
#                         [width-1, 0],
#                         [width-1, height-1]], dtype="float32")
#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#     warped = cv2.warpPerspective(img, M, (width, height))
#     return warped