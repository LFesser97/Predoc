# -*- coding: utf-8 -*-
"""
teikoku_segment_inference.py

Created on Sun Sep  4 11:17:07 2022

@author: Lukas
"""

# Import packages

import os
import json
import layoutparser
from layoutparser import Rectangle, TextBlock, Layout
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
from tqdm import tqdm

# Set directories

images = ''
output = ''
weights = ''

# Define class "LayoutModel"

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
			cur_block = TextBlock(Rectangle(x_1, y_1, x_2, y_2), type=label, score=score)
			cur_blocks.append(cur_block)
		return cur_blocks

	def _create_model(self):
		self.model = DefaultPredictor(self.cfg)

	def pred_block(self, image):
		outputs = self.model(image)
		blocks =  self.gather_output(outputs)
		return blocks

# Define main function for teikoku

def main(images_path = images, output_path = output, model_path = weights):
    print("Running model inference on Teikoku...")
    layout = LayoutModel(model_path, label_map={    # Change label_map
        0:"Title",
        1:"Address",
        2:"Text",
        3:"Number",
        4:"Variable",
        5:"CompanyCat"},
        extra_config=["TEST.DETECTIONS_PER_IMAGE", 250,    # Change extra_config
			"MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6,
			"MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.35])
    images = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    count = 0
    for img in tqdm(sorted(images)[1000:]):    # Why are we starting at entry 1000? What if we don't?
        image = cv2.imread(img)
        h, w = image.shape[:2]
        if h <= 2000 or w <= 2000:
            count += 1
        segments = layout.pred_block(image)
        dict = []    # rename this
        for seg in segments:
            dict.append({'coordinates': seg.coordinates, 
                         'class': seg.type, 
                         'confidence': seg.score})
        with open(os.path.join(output_path, img.split("/")[-1].split(".")[0] + '.json'), 'w') as f:    # simplify this
            json.dump(dict, f, indent=4)

# Include '__main__' statement to execute script

if __name__ == '__main__':
    main()