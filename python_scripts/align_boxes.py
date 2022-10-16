# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:51:38 2022

@author: Lukas
"""

def align_boxes(predictions):
    """
    

    Parameters
    ----------
    predictions : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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