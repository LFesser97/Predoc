# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:36:58 2022

@author: Lukas
"""

# %%
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import cv2
import json

import layoutparser as lp

# %%

# Set directories
current_folder = os.getcwd()
image_folder  = os.path.join(current_folder , 'images')
json_folder  = os.path.join(current_folder , 'jsons')
output_folder = os.path.join(current_folder , 'annotations')

# page_list=[]

# # Changed input directory
# for image in image_folder:
#     image=image.split('.')[0]
#     image=image.split('/')[-1]
#     page=image[:-2]
#     page_list.append(page)
   
# page_list=sorted(page_list)


# image_list=[]
# for page in page_list:
#     if page not in image_list:
#         image_list.append(page)

# print(image_list)

# %%

# DOES THIS ONLY WORK WITH COLUMN MODEL OUTPUTS???

image=Image.open(os.path.join(image_folder, 'test_output.jpg'))
with open(os.path.join(json_folder, 'aligned_test_block.json')) as f:
    image_json=json.load(f)
    img_boxes=[]
    block_type=[]
    block_score=[] # added this

    draw=ImageDraw.Draw(image)


    # color_map={"branch_block":"red","personnel_block":"blue","table_block":"crimson","title_block":"green","var_block":"pink"}
        
    color_map = {
            "text_region" : "red",
            "row" : "blue",
            "family" : "green",
            "subtitle" : "pink",
            "position" : "yellow",
            "title" : "orange",
            "xx" : "black"
            }
        
    for json_box in image_json:
        img_boxes.append(json_box["coordinates"])
        block_type.append(json_box["class"])
        block_score.append(json_box["confidence"]) # added this


    for box,type, score in zip(img_boxes,block_type, block_score):
        x1,y1,x2,y2=box
        #x2=x1+w
        #y2=y1+h
        draw.rectangle([x1,y1,x2,y2],outline=color_map[type],width=5)
        
        # # added this:
        # text_x = x1
        # text_y = y1
        # trunc_score = round(score, 4)
        # font = ImageFont.truetype("arial.ttf", size=30)
        # draw.text((text_x, text_y), str(type)+" ("+str(trunc_score)+")", 
        #           fill = 'black', align = 'right', font = font)

        # Update this
        
    image.show
    image.save(os.path.join(output_folder, 'aligned_test_output.png')) # Put your output path here


# %%