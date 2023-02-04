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
    Rotates an image by a given angle.
    Resizes the image to avoid cropping.

    Parameters
    ----------
    image : numpy.ndarray
        Image to rotate.

    angle : int
        Angle in degrees to rotate image by.

    Returns
    -------
    rotated_image : numpy.ndarray
        Rotated image.
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    
    # Calculate new image dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2
    
    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated_image



def rotate_images(input_folder, output_folder, k):
    """
    Rotates all images in a given input folder by k times 90 degrees, where k is an input integer. 

    Parameters
    ----------
    input_folder : str
        Path to input folder.

    output_folder : str
        Path to output folder.
    
    k : int
        Number of times to rotate image by 90 degrees (to the left)

    Returns
    -------
    None.
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

k = -1 # Number of times to rotate image by 90 degrees

main(input_folder, output_folder, k)