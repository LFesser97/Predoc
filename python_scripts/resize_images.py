"""
resize_images.py

Created on Thu Oct 13 12:38:11 2022

@author: Lukas

This script resizes all images in a folder by a given factor.
"""

# import packages

import os
import cv2
import argparse


# set path directories

path = os.getcwd()

input_path = os.path.join(path, "input_images")
output_path = os.path.join(path, "output_images")


# auxiliary function

def resize_image(image, fact):
    """
    Resizes an image by a given factor.
    
    Parameters
    ----------
    
    image : numpy.ndarray
        The image to be resized.
    
    fact : float
        The factor by which the image is resized.
    
    Returns
    -------
    
    res_img : numpy.ndarray
        the resized image
    
    """
    img = cv2.imread(image)
    res_img = cv2.resize(img, (0, 0), fx=1/fact, fy=1/fact)
    return res_img


# main function

def main(input_path, output_path, fact):
    """
    Main function.

    Parameters
    ----------

    input_path : str

    output_path : str

    fact : float

    Returns
    -------
    None.
        Saves the resized images in the output path.
    
    """    
    # get all images in the folder
    for image in os.listdir(input_path):
        res_img = resize_image(os.path.join(input_path, str(image)), fact)
        cv2.imwrite(os.path.join(output_path, 'res_' + str(image)), res_img)
    

if __name__ == "__main__":
    main(input_path, output_path, 5)

# if __name__ == "__main__":
#     # parse arguments from shell script
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--input_path", type=str, help="Path to the input folder.")
#     parser.add_argument("--output_path", type=str, help="Path to the output folder.")
#     parser.add_argument("--fact", type=float, help="Factor by which the images are resized.")
#     args = parser.parse_args()

#     print(args.input_path)

#     main(args.input_path, args.output_path, args.fact)