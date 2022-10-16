# -*- coding: utf-8 -*-
"""
column_inference_via_cropping.py

Created on Mon Aug 29 16:18:51 2022

@author: Lukas
"""

# Can we get this from LayoutParser?!
import sys 
import os
import json
import layoutparser
import torch
import cv2
import os
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm
from layoutparser import Rectangle, TextBlock, Layout
from PIL import Image

# Directories for main function. For test purposes only.

file_path = os.getcwd()

image_path = os.path.join(file_path, '64_1.png') # Need to provide original image, not cropped one
output_path = os.path.join(file_path, '64_1_column_output.json') # This should probably refer to a json file, not a folder
model_path = '/mnt/data01/dell-japan/model_weights/ww-column-labeling/model-output/ww-column-model-v3'
segment_output_path = os.path.join(file_path, '64_1.json')


# main function to run the column model on all coordinate boxes provided by the segment model
def main(segment_output_path, image_path, model_path, output_path):
    """
    runs column model-based inference on the blocks provided by the segment model

    Parameters
    ----------
    segment_output_path : string
        file path to the segment model output (json)
        
    image_path : string
        file path to the input image
        
    model_path : string
        file path to the column model to be used for inference
        
    output_path : string
        file path to the output directory

    Returns
    -------
    output_predictions : json
        json containing the column model predictions, saved in the output directory

    """
    # import segment model output, filter to get row boxes
    segment_output_json = open(segment_output_path)
    segment_output = json.load(segment_output_json)
    row_boxes = [block for block in segment_output
                 if block["class"] in {"text_region", "family", "position"}]
    output_predictions = []
    
    # WRITE THIS AS AN AUXILLIARY FUNCTION
    # crop rows from original image, run inference, and rescale output
    for box in row_boxes:
        box_coordinates = box["coordinates"]
        cropped_image = crop_image(image_path, box_coordinates)
        cropped_predictions = block_inference(cropped_image, model_path)
        rescaled_predictions = rescale_boxes(box_coordinates, cropped_predictions)
        # output_predictions += rescaled_predictions # without box alignment function
        output_predictions += align_boxes(rescaled_predictions) # with box alignment function
        
    # save output
    with open(output_path, 'w') as f:
        json.dump(output_predictions, f, indent=4)


# LayoutModel class, copied from pr-ww-model-inference.py
# Import this in the final script?
class LayoutModel(object):

	def __init__(self, model_dir,
					   config_name = 'config.yaml',
					   model_name  = 'model_final.pth',
					   label_map = None,
					   extra_config= []):

		cfg = get_cfg()
		cfg.merge_from_file(os.path.join(model_dir, config_name))
		cfg.MODEL.WEIGHTS = os.path.join(model_dir, model_name)
		cfg.merge_from_list(extra_config)
		self.cfg = cfg
		cfg.MODEL.DEVICE='cpu' if not torch.cuda.is_available() else 'cuda:0'
		self.label_map = label_map
		self._create_model()

	def gather_output(self, outputs):

		instance_pred = outputs['instances'].to("cpu")

		cur_blocks = []
		scores = instance_pred.scores.tolist()
		boxes  = instance_pred.pred_boxes.tensor.tolist()
		labels = instance_pred.pred_classes.tolist()

		for score, box, label in zip(scores, boxes, labels):
			x_1, y_1, x_2, y_2 = box
			
			if self.label_map is not None:
				label = self.label_map[label]

			cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
                )
			cur_blocks.append(cur_block)

		return cur_blocks

	def _create_model(self):
		self.model = DefaultPredictor(self.cfg)

	def pred_block(self, image):

		outputs = self.model(image)
		blocks =  self.gather_output(outputs)
		
		return blocks


# crop_image
def crop_image(image_source, coordinates):
    """
    crops an input image according to the coordinates of a given block

    Parameters
    ----------
    image_source : string
        file path for the input image
    coordinates : list
        list of length 4 with the block coordinates as entries

    Returns
    -------
    block : 
        a list of the cropped images

    """
    im = Image.open(image_source)
    im_block = im.crop(tuple(coordinates)) # need coordinates in tuple, not list format
    return im_block
        

# inference function for cropped images
def block_inference(image_block, model_path):
    """
    applies the column model to a given (cropped) input image

    Parameters
    ----------
    image_block : image
        cropped (sub-) image, usually the output of crop_image
    model_path : string
        file path to the model directory

    Returns
    -------
    block_predictions: list of dicts
        the predictions made by the column model

    """
    layout = LayoutModel(model_path,
            label_map={0: 'xx'},
            extra_config=["MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.2,
                          "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])
    image_block = np.array(image_block) # need to get image into cv2 format
    h, w = image_block.shape[:2]
    segments = layout.pred_block(image_block)
    block_predictions = [{
			'coordinates': seg.coordinates,
			'class': seg.type,
			'confidence': seg.score
	        } for seg in segments] # list comprehension to gather predictions for image_block
    return block_predictions


# Comments on the coordinate representation:
# for coordinates [x1, y1, x2, y2], (0, 0) is the upper-left image corner,
# (x1, y1) is the upper-left box corner, (x2, y2) is the lower-right box corner

# rescale function for block predictions
def rescale_boxes(block_coordinates, block_predictions):
    """
    rescales the block-based coordinates of the predicted boxes
    to coordinates on the original input image

    Parameters
    ----------
    block_coordinates : list
        coordinates of the block in the original image
    block_predictions : list
        coordinates of the predicted boxes w.r.t. the cropped image/ block

    Returns
    -------
    rescaled_predictions : list
        coordintes of the predicted boxes w.r.t. the original image

    """
    # Add coordinates, b_ for block, p_ for prediction 
    def transform_coordinates(block_coordinates, block_prediction):
        [b_x1, b_y1, b_x2, b_y2] = block_coordinates
        [p_x1, p_y1, p_x2, p_y2] = block_prediction["coordinates"]
        block_prediction["coordinates"] = [b_x1 + p_x1, b_y1 + p_y1, b_x1 + p_x2, b_y1 + p_y2]
        return block_prediction
    
    # Transform coordinates for all predicted blocks
    rescaled_predictions = [transform_coordinates(block_coordinates, block) for block in block_predictions]
    return rescaled_predictions


def align_boxes(predictions):
    """
    aligns all column box predictions within the same block to have the same height/ depth

    Parameters
    ----------
    predictions : list (of dicts)
        boxes predicted by the column model for a given segment block

    Returns
    -------
    predictions : list (of dicts)
        re-aligned boxes based on the maximal heigh and maximal depth (y coordinates)

    """
    box_heights = [box["coordinates"][1] for box in predictions]
    max_height = min(box_heights)
    for box in predictions:
        box["coordinates"][1] = max_height
        
    box_depth = [box["coordinates"][3] for box in predictions]
    max_depth = max(box_depth)
    for box in predictions:
        box["coordinates"][3] = max_depth

    return predictions


if __name__ == "__main__":
    main(segment_output_path, image_path, model_path, output_path)



# # Test rescale_boxes
# test_layout = [9.975238800048828, 4.102687358856201, 781.914306640625, 910.8379516601562]

# test_prediction_json = open("Visualization/jsons/test_image.json")
# test_prediction = json.load(test_prediction_json)

# test_output = rescale_boxes(test_layout, test_prediction)
# with open("Visualization/jsons/rescaled_test_image.json", 'w') as f:
#      json.dump(test_output, f, indent=4)


# # Test block_inference
# file_path = os.getcwd()
# image_path = os.path.join(file_path, 'test_output.jpg')
# test_image = Image.open(image_path)
# output_path = file_path			
# column_model_path = '/mnt/data01/dell-japan/model_weights/ww-column-labeling/model-output/ww-column-model-v3'
# test_output = block_inference(test_image, column_model_path)
# with open(os.path.join(output_path, 'test_image.json'), 'w') as f:
#     json.dump(test_output, f, indent=4)


# # Test crop_image on toy example
# test_image = "C:/Users/Lukas/Documents/Harvard/Predoc/PR Inference/Visualization/images/62_0.png"
# test_layout = [9.975238800048828, 4.102687358856201, 781.914306640625, 910.8379516601562]
# test_output = crop_image(test_image, test_layout)
# test_output.save('C:/Users/Lukas/Documents/Harvard/Predoc/PR Inference/Visualization/test_output.jpg')