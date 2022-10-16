# -*- coding: utf-8 -*-
"""
register_dataset

Created on Sun Sep 25 14:22:39 2022

@author: Lukas

Sets up a custom dataset for use by Detectron2's DatasetCatalog. Should be run
before (re-)training a model with train_net.py or plain_train_net.py
"""

# import packages
import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# move path directories later?
path = os.getcwd()

# The following three lines will only work if your dataset is in COCO format 
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "tk_train.json", path)
register_coco_instances("my_dataset_val", {}, "tk_val.json", path)