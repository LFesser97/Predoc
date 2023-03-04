"""
check_deskewing_output.py

Created on Mon Feb 27 2023

@author: Lukas

This files contains all functions for the checking the output of the deskewing pipeline.
"""

# import packages

import cv2
import os
import json


# function to load in an image and return its size

def get_image_size(image_path):
    """
    Function to get the size of an image.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    height : int
        Height of the image in pixels.

    width : int
        Width of the image in pixels.
    """
    # load image using cv2
    img = cv2.imread(image_path)

    # get image size
    height = img.shape[0]
    width = img.shape[1]
    return height, width


# main function

def main(image_path, output_path):
    """
    Main function to check the output of the deskewing pipeline.

    Parameters
    ----------
    image_path : str
        Path to the image.

    output_path : str
        Path to the output.

    Returns
    -------
    None.
        candidate_list is saved as a json file in the output directory.
    """

    # iterate over all images in image_path
    for image in os.listdir(image_path):

        print(image)

        # initialize list for candidate images
        candidate_list = []

        # get image size
        height, width = get_image_size(os.path.join(image_path, image))

        # if height > 2 * width, add image to candidate list
        if height > 2 * width:
            candidate_list.append(image)

    # save candidate list as a json file in the output directory
    json_string = json.dumps(candidate_list, indent = 4)
    json_file = open(os.path.join(output_path, "candidate_list.json"), "w")
    json_file.write(json_string)
    json_file.close()


image_path = ''
output_path = ''

if __name__ == "__main__":
    main(image_path, output_path)