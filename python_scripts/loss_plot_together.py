# -*- coding: utf-8 -*-
"""
loss_plot_together.py

Created on Tue Sep 20 09:24:29 2022

@author: Lukas
"""

# %%
import json
import matplotlib.pyplot as plt

experiment_folder = '/Users/yangxinmei/Downloads/outputssmallnms0912_val'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

# %%
print(experiment_metrics)
# %%

plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()

# %%