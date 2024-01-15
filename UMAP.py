#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:32:15 2023

@author: mainaoshige
"""

# Loading the library

# pandas https://pandas.pydata.org
# To install,
# pip install pandas
# or
# conda create -c conda-forge -n name_of_my_env python pandas
import pandas as pd

# numpy https://numpy.org/ja/
# To install,
# pip install numpy
# or
# conda install numpy
import numpy as np

# UMAP https://umap-learn.readthedocs.io/en/latest/
# To install,
# pip install umap-learn
# or
# conda install -c conda-forge umap-learn
import umap

# seaborn https://seaborn.pydata.org/installing.html
# To install,
# pip install seaborn
# or
# conda install seaborn -c conda-forge
import seaborn as sns

# Matplotlib https://matplotlib.org/3.5.3/index.html
# To install,
# pip install matplotlib
# or
# conda install matplotlib
import matplotlib.pyplot as plt


sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})


# data processing
df=pd.read_csv("umap_all.csv")

# List of all data
InputItems = ["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_Dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_Bydoctor"]
print(InputItems)

# # List of data divided by sex
# InputItems = ["Time_from_onset","Age","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_Dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_Bydoctor"]
# print(InputItems)

# Extracting severity data
severity = df["Severity"]
print(severity)

# Extracting input data
input_data = df[InputItems]
print(input_data)

data = df.drop("Severity", axis=1)


# # # Assigning severity
fit = umap.UMAP()
u = fit.fit_transform(data)



# Colormap reference - Matplotlib https://matplotlib.org/stable/gallery/color/colormap_reference.html
# Visualization
plt.figure(figsize = (30,12))
plt.subplot(122)
scatter=plt.scatter(u[:,0], u[:,1],  c =severity , 
            cmap = "coolwarm", alpha=0.35)
plt.legend(*scatter.legend_elements())
plt.title('UMAP all');



# Hiding axis labels
plt.xticks([])
plt.yticks([])


# Changing the parameters
def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(input_data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=severity, cmap="coolwarm")
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=severity, cmap="coolwarm")
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=severity, cmap="coolwarm", s=100)
    plt.legend()
    plt.title(title, fontsize=18)


for n in (2, 5, 10, 20, 50, 100, 200):
  draw_umap(n_neighbors=n, title='n_neighbors = {}'.format(n))
for d in (0.0, 0.1, 0.25, 0.5, 0.8, 0.99):
  draw_umap(min_dist=d, title='min_dist = {}'.format(d))


import numba
@numba.njit()
def red_channel_dist(a,b):
    return np.abs(a[0] - b[0])
@numba.njit()
def hue(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if cmax == r:
        return ((g - b) / delta) % 6
    elif cmax == g:
        return ((b - r) / delta) + 2
    else:
        return ((r - g) / delta) + 4

@numba.njit()
def lightness(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    return (cmax + cmin) / 2.0

@numba.njit()
def saturation(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    chroma = cmax - cmin
    light = lightness(r, g, b)
    if light == 1:
        return 0
    else:
        return chroma / (1 - abs(2*light - 1))
@numba.njit()
def hue_dist(a, b):
    diff = (hue(a[0], a[1], a[2]) - hue(b[0], b[1], b[2])) % 6
    if diff < 0:
        return diff + 6
    else:
        return diff

@numba.njit()
def sl_dist(a, b):
    a_sat = saturation(a[0], a[1], a[2])
    b_sat = saturation(b[0], b[1], b[2])
    a_light = lightness(a[0], a[1], a[2])
    b_light = lightness(b[0], b[1], b[2])
    return (a_sat - b_sat)**2 + (a_light - b_light)**2

@numba.njit()
def hsl_dist(a, b):
    a_sat = saturation(a[0], a[1], a[2])
    b_sat = saturation(b[0], b[1], b[2])
    a_light = lightness(a[0], a[1], a[2])
    b_light = lightness(b[0], b[1], b[2])
    a_hue = hue(a[0], a[1], a[2])
    b_hue = hue(b[0], b[1], b[2])
    return (a_sat - b_sat)**2 + (a_light - b_light)**2 + (((a_hue - b_hue) % 6) / 6.0)
for m in ("euclidean", red_channel_dist, sl_dist, hue_dist, hsl_dist):
    name = m if type(m) is str else m.__name__
    draw_umap(n_components=2, metric=m, title='metric = {}'.format(name))



