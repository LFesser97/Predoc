# -*- coding: utf-8 -*-
"""
segment_inference.py

Created on Sun Aug 28 14:56:22 2022

@author: Lukas
"""

import sys 
import os
import json

file_path = os.getcwd()

file_path = os.path.split(file_path)[0]
file_path = os.path.split(file_path)[0]
file_path = os.path.split(file_path)[0]

images = os.path.join(file_path, "image_files")

# output = os.path.join(file_path, "model_outputs/block_outputs") # for block model inference

output = os.path.join(file_path, "model_outputs/column_outputs") # for column model inference

# weights = os.path.join(file_path, "model_weights/ww1939-bs2-it45k-apr21-e9") # for block model inference

weights = os.path.join(file_path, "model_weights/ww-column-labeling/model-output/ww-column-model-v3") # for column model inference


import layoutparser

# from layoutparser import BaseLayoutModel
from layoutparser import Rectangle, TextBlock, Layout

# from layoutparser.ocr import *
# from layoutparser.image import *
# from layoutparser.visualization import *
# from layoutparser.coordinate import *

# from lib.utils import *
# from lib.constants import *

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import cv2

from tqdm import tqdm
import os

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

def main(
	book="WhosWho",
# 	images_path='/media/krishna/large-hdd/ww1939-page-image',
# 	output_path='/media/krishna/large-hdd/ww1939-model-inference',
# 	model_path='/home/krishna/Dropbox (Melissa Dell)/Label-Studio-Labeling/whoswho-1939-labeling/model-output/ww1939-bs2-it30k-apr21-e9'

    images_path = images,
    output_path = output,
    model_path = weights
):
	
	os.makedirs(output_path, exist_ok=True)
	
	if book == "PR":
		print("Running model inference on Personnel Records...")
		layout = LayoutModel(model_path,
			label_map={0:"title",1:"text",2:"variable",3:"branch"},
			extra_config=["TEST.DETECTIONS_PER_IMAGE", 250, 
							"MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6,
							"MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.35])

		img_directories = [os.path.join(images_path, p) for p in os.listdir(images_path)]

		for dir in tqdm(sorted(img_directories)[1200:]):
			col_images = [os.path.join(dir, i) for i in os.listdir(dir)]
			subdir = dir.split("/")[-1]
			os.makedirs(os.path.join(output_path, subdir), exist_ok=True)
			for col in col_images:
				image = cv2.imread(col)
				segments = layout.pred_block(image)
				dict = []
				for seg in segments:
					dict.append({
						'coordinates': seg.coordinates,
						'class': seg.type,
						'confidence': seg.score
					})

				with open(os.path.join(output_path, subdir, col.split("/")[-1].split(".")[0] + '.json'), 'w') as f:
					json.dump(dict, f, indent=4)
			
	if book == "WhosWho":
		print("Running model inference on Whos Who...")
		layout = LayoutModel(model_path,
			label_map={
				0:"page_frame",
				1:"row",
				2:"title",
				3:"text_region",
				4:"subtitle",
				5:"position",
				6:"family",
				7:"bio",
				8:"other"
			},
			extra_config=["TEST.DETECTIONS_PER_IMAGE", 250, 
							"MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6,
							"MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.35])

		images = [os.path.join(images_path, p) for p in os.listdir(images_path)]
		count = 0
		for img in tqdm(sorted(images)[1000:]):
			
			image = cv2.imread(img)
			h, w = image.shape[:2]
			if h <= 2000 or w <= 2000:
				count += 1
			segments = layout.pred_block(image)
			dict = []
			for seg in segments:
				dict.append({
					'coordinates': seg.coordinates,
					'class': seg.type,
					'confidence': seg.score
				})

			with open(os.path.join(output_path, img.split("/")[-1].split(".")[0] + '.json'), 'w') as f:
				json.dump(dict, f, indent=4)
		print(f"total mess {count}")
if __name__ == '__main__':
	main()