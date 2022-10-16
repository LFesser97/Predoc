# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:51:02 2022

@author: Lukas
"""

from pycocotools.coco import COCO
import layoutparser as lp
import random
import cv2

# New packages
import os


def load_coco_annotations(annotations, coco=None):
    """
    Args:
        annotations (List):
            a list of coco annotaions for the current image

        coco (`optional`, defaults to `False`):
            COCO annotation object instance. If set, this function will
            convert the loaded annotation category ids to category names
            set in COCO.categories
            
    """
    layout = lp.Layout()

    for ele in annotations:

        x, y, w, h = ele['bbox']

        layout.append(
            lp.TextBlock(
                block = lp.Rectangle(x, y, w+x, h+y),
                type  = ele['category_id'] if coco is None else coco.cats[ele['category_id']]['name'],
                id = ele['id']
            )
        )

    return layout


# update these with local directories
# COCO_ANNO_PATH = 'data/publaynet/annotations/val.json'
# COCO_IMG_PATH  = 'data/publaynet/val'

file_path = os.getcwd()

COCO_ANNO_PATH = os.path.join(file_path, "jsons", "62_0.json")

# COCO_IMG_PATH = os.path.join(file_path, "images", "62_0")
COCO_IMG_PATH = os.path.join(file_path, "images")

coco = COCO(COCO_ANNO_PATH)


# update these with Japan labels
color_map = {
    'text_region':   'red',
    'title':  'blue',
    'list':   'green',
    'table':  'purple',
    'figure': 'pink',
}


for image_id in random.sample(coco.imgs.keys(), 1):
    image_info = coco.imgs[image_id]
    annotations = coco.loadAnns(coco.getAnnIds([image_id]))

    image = cv2.imread(f'{COCO_IMG_PATH}/{image_info["file_name"]}')
    layout = load_coco_annotations(annotations, coco)

    viz = lp.draw_box(image, layout, color_map=color_map)
    display(viz) # show the results
    


    