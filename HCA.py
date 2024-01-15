# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:55:06 2023

@author: maina
"""

# Loading the library

# pandas https://pandas.pydata.org
# pip install pandas
# or
# conda create -c conda-forge -n name_of_my_env python pandas
import pandas as pd

# numpy https://numpy.org/ja/
# pip install numpy
# or
# conda install numpy
import numpy as np

# seaborn https://seaborn.pydata.org/index.html
# pip install seaborn
# or
# conda install seaborn -c conda-forge
import seaborn as sns

# matplotlib https://matplotlib.org/stable/
# pip install matplotlib
# or
# conda install -c conda-forge matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] ='sans-serif'

rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio']

# Loading data

#covid = pd.read_csv("clusteranalysis20240105.csv")
covid = pd.read_csv("clusteranalysis20240105.csv") 

covid_data = covid.drop("Classification", axis=1) 

# Extract "classification" and assign to severity, remove the column at the same time

severity = covid_data.pop("Severity") 

#Create a list of severity

severity_name = severity.unique() 

# List of colors to use (4 colors)

color=['blue','green','orange','red']

# Match a color to each classification label

lut = dict(zip(severity_name, color)) 

# Create a list showing the color (R,G,B) to display for each severity

row_colors = severity.map(lut) 

#　Hierarchical Clustering Dendrogram

# https://seaborn.pydata.org/generated/seaborn.clustermap.html

g = sns.clustermap(covid_data,

                    row_colors=row_colors,
                   
                   

                    method='ward',
                   

                    metric='euclidean',
                   
                    cmap='Reds')

# Title

plt.title("all patients", fontsize=20)

# Severity Legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=lut[name]) for name in lut]
plt.legend(handles, lut, title='severity',
            bbox_to_anchor=(0, 0.3), bbox_transform=plt.gcf().transFigure, loc='upper right')

# Save the image

g.savefig('clustermap.png', dpi=150)
