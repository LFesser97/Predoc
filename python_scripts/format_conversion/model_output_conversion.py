# -*- coding: utf-8 -*-
"""
model_output_conversion.py

Created on Wed Sep  7 13:25:46 2022

@author: Lukas

converts all the jsons in a given input folder from the model output format
to the one required by Label Studio for further annotations by hand
"""

# import statements
import cv2
import json
import os


# path dependencies
current_path = os.getcwd()

image_path = os.path.join(current_path, 'input_scans')
json_path = os.path.join(current_path, 'input_jsons')
output_path = os.path.join(current_path, 'output_jsons')


def take_size(image):
    """
    returns the total pixel size of an image

    Parameters
    ----------
    image : string
        file path of the input image

    Returns
    -------
    size : tuple
        triple (heigth, width, channels) in pixels
        
    """
    im = cv2.imread(image)
    (height, width, channels) = im.shape
    return (height, width, channels)


def resize_bbox(bbox, image_width, image_height):
    """
    resizes a bounding box on a given image to Label Studio format

    Parameters
    ----------
    bbox : list
        list of original coordinates (upper-left and lower-right)
        
    image_width : int
        width of the input image in pixel
        
    image_height : int
        height of the input image in pixel

    Returns
    -------
    resized_bbox : tuple
        (x, y, width, height), (x, y) is the upper-left bbox corner
        
    """
    x = (bbox[0] * 100) / image_width
    y = (bbox[1] * 100) / image_height
    w = ((bbox[2] - bbox[0]) * 100) / image_width
    h = ((bbox[3] - bbox[1]) * 100) / image_height
    return (x, y, w, h)


def convert_format(image, json_name, output_name):
    """
    converts an input json file into the format required by Label Studio

    Parameters
    ----------
    image : string
        file path of the input image
        
    json_name : string
        file path of the input json
        
    output_name : string
        file path of the output json

    Returns
    -------
    None (converted json files are saved in the specified output folder)

    """    
    # get image size
    image_sizes = take_size(os.path.join(image_path, image))
    image_height = image_sizes[0]
    image_width = image_sizes[1]
    
    # load input json
    with open(json_name) as f:
        predictions = json.load(f) 
        
        # construct output in the desired format
        result_json = {
            "data" : {
                "ocr" : "http://localhost:8081/"+image
                },
            "predictions" : [
                {
                    "model_version" : "column_model_v9",
                    "result" : [],
                    "score" : 0}]}
        
        # append reformated predictions
        bbox_id = 0
        
        for prediction in predictions:
            resized_bbox = resize_bbox(prediction['coordinates'], image_width, image_height)
            result_json['predictions'][0]['result'].append({
                "original_width" : image_width,
                "original_height" : image_height,
                "image_rotation" : 0,
                "value":{
                    "x": resized_bbox[0],
                    "y": resized_bbox[1],
                    "width": resized_bbox[2],
                    "height": resized_bbox[3],
                    "rotation" : 0
                    },
                "id": bbox_id,
                "from_name": "image",
                "to_name": "image",
                "type": "rectanglelabels" # might have to change this if we want to keep predicted classes
                })
            bbox_id += 1
        
        # save output
        with open(os.path.join(output_path, output_name),'w') as o:
            json.dump(result_json, o, indent=4)
        

def main(image_path, json_path, output_path):
    """
    converts all json formats in a given input folder
    
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
    None.

    """
    for image_name in os.listdir(image_path):
        name = image_name.split(".")[0]
        json_name = name + '.json'
        output_name = 'converted_' + json_name
        convert_format(image_name,
                       os.path.join(json_path, json_name),
                       os.path.join(output_path, output_name))
    
    
if __name__ == '__main__':
    main(image_path, json_path, output_path)    