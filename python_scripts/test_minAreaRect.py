# -*- coding: utf-8 -*-
"""
test_minAreaRect

Created on Tue Sep 13 08:59:42 2022

@author: Lukas
"""

import numpy as np
import cv2

cnt2 = np.array([
        [[64, 49]],
        [[122, 11]],
        [[391, 326]],
        [[308, 373]]
    ])

rect = cv2.minAreaRect(cnt2)
