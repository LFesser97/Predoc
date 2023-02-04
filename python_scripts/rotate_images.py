"""
rotate_images.py

Created on Wed Feb 1 2023

@author: Lukas

This script rotates all images in a given input folder by k times 90 degrees, where k is an input integer. 
"""

# Import packages

import os
import numpy as np
import cv2


# Define auxilliary functions

def rotate_image(image, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_images(input_folder, output_folder, k):
    """
    Rotates all images in a given input folder by k times 90 degrees, where k is an input integer. 
    """
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over all images in input folder
    for image_name in os.listdir(input_folder):
        # Read image
        image = cv2.imread(os.path.join(input_folder, image_name))
        # Rotate image
        rotated_image = rotate_image(image, k * 90)
        # Save image
        cv2.imwrite(os.path.join(output_folder, image_name), rotated_image)


# Define main function

def main(input_folder, output_folder, k):
    """
    Rotates all images in a given input folder by k times 90 degrees, where k is an input integer. 
    """
    rotate_images(input_folder, output_folder, k)


input_folder = '' # Path to input folder
output_folder = '' # Path to output folder

k = 1 # Number of times to rotate image by 90 degrees

main(input_folder, output_folder, k)