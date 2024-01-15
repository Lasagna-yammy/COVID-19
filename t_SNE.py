#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:31:16 2023

@author: mainaoshige
"""

# Loading the library

# pandas https://pandas.pydata.org
# To install,
# pip install pandas
# or
# conda create -c conda-forge -n name_of_my_env python pandas
import pandas as pd

# Matplotlib https://matplotlib.org/3.5.3/index.html
# To install,
# pip install matplotlib
# or
# conda install matplotlib
import matplotlib.pyplot as plt

# scikit-learn https://scikit-learn.org/stable/index.html
# To install,
# pip install tsne
# or
# conda install -c maxibor tsne
from sklearn.manifold import TSNE

# data processing
df=pd.read_csv("umap_male.csv")

# # List of all data
# InputItems = ["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_Dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_Bydoctor"]
# print(InputItems)

# List of data divided by sex
InputItems = ["Time_from_onset","Age","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_Dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_Bydoctor"]
print(InputItems)

# Extracting severity data
severity = df["Severity"]
print(severity)

# Extracting input data
input_data = df[InputItems]
print(input_data)

data = df.drop("Severity", axis=1)

# t-SNE
tsne = TSNE(n_components=2, random_state=1)
tsne_reduced = tsne.fit_transform(input_data)


# Visualization
plt.figure(figsize = (30,12))
plt.subplot(122)
scatter=plt.scatter(tsne_reduced[:,0],tsne_reduced[:,1],  c =severity , 
            cmap = "coolwarm", alpha=0.35)
plt.legend(*scatter.legend_elements())
plt.title('t-SNE male')
plt.show()

# Changing the parameters

# Perplexity
for perplexity in (5, 10, 30, 50):
    plt.figure(figsize = (30,12))
    plt.subplot(122)
    mapper = TSNE(perplexity=perplexity)
    tsne_reduced = mapper.fit_transform(input_data)
    title='perplexity = {0}'.format(perplexity)
    plt.title(title)
    plt.scatter(tsne_reduced[:,0],tsne_reduced[:,1],  c =severity , 
                cmap = "coolwarm", edgecolor = "None", alpha=0.35)
    plt.colorbar()
    plt.show()

# early_exaggeration
for early_exaggeration in (6, 12, 24, 48):
    plt.figure(figsize = (30,12))
    plt.subplot(122)
    mapper = TSNE(early_exaggeration=early_exaggeration)
    tsne_reduced = mapper.fit_transform(input_data)
    title='early_exaggeration = {0}'.format(early_exaggeration)
    plt.title(title)
    plt.scatter(tsne_reduced[:,0],tsne_reduced[:,1],  c =severity , 
                cmap = "coolwarm", edgecolor = "None", alpha=0.35)
    plt.colorbar()
    plt.show()

# init
for init in ["random", "pca"]:
    plt.figure(figsize = (30,12))
    plt.subplot(122)
    mapper = TSNE(init=init)
    tsne_reduced = mapper.fit_transform(input_data)
    title='init = {0}'.format(init)
    plt.title(title)
    plt.scatter(tsne_reduced[:,0],tsne_reduced[:,1],  c =severity , 
                cmap = "coolwarm", edgecolor = "None", alpha=0.35)
    plt.colorbar()
    plt.show()


