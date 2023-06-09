# -*- coding: utf-8 -*-
"""
model_output_conversion.py

Created on Tue Oct 4 13:25:46 2022

@author: Lukas

converts all the json files in a given input folder from the 
Label Studio output format to the format required by Detectron2
"""

# import statements
import json
import os

# path dependencies
current_path = os.getcwd()

input_path = os.path.join(current_path, 'label_studio_output')
output_path = current_path

# auxilliary functions
def convert_format(input_path, output_path):
    """
    converts an input json file into the format required by Detectron2

    Parameters
    ----------
    input_path : string
        path to the input json file

    output_path : string
        path to the output json file

    Returns
    -------
    None (writes a json file to the output path)

    """
    # load input json
    with open(input_path) as f:
        annotations = json.load(f)

        # construct output in the desired format
        output = {}


# main function
def main(input_path, output_path):
    """
    converts all the json files in a given input folder

    Parameters
    ----------
    input_path : string
        path to the input folder

    output_path : string
        path to the output folder

    Returns
    -------
    None.

    """
    for file in os.listdir(input_path):
        output_name = 'converted_' + file
        convert_format(os.path.join(input_path, file), os.path.join(output_path, output_name))


if __name__ == '__main__':
    main(input_path, output_path)