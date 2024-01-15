#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:56:47 2023

@author: mainaoshige
"""

# Loading the library

# numpy https://numpy.org/ja/
# To install,
# pip install numpy
# or
# conda install numpy
import numpy as np

# pandas https://pandas.pydata.org
# To install,
# pip install pandas
# or
# conda create -c conda-forge -n name_of_my_env python pandas
import pandas as pd

# scikit-learn https://scikit-learn.org/stable/index.html
# To install,
# pip install -U scikit-learn
# or
# conda install scikit-learn-intelex
from sklearn.model_selection import train_test_split

# Matplotlib https://matplotlib.org/3.5.3/index.html
# To install,
# pip install matplotlib
# or
# conda install matplotlib
import matplotlib.pyplot as plt

# Load CSV file
data_frame = pd.read_csv('4deep_clinicalbloodAI.csv')

# Remove missing values
data_frame = data_frame.dropna()

# Set the name of the response variable
y = data_frame["Severity"]

# #Set the name of the explanatory variable Clinical data
# X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]]

# #Set the name of the explanatory variable　Clinical and blood data
# X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]]

# # Set the name of the explanatory variable　Clinical, blood and chest radiography data(By doctor)
# X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_Bydoctor"]]

# # Set the name of the explanatory variable　Clinical, blood and chest radiography data(By AI)
X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_ByAI"]]


# Split training data and varification data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# Create decisiontree
from sklearn import tree

# Set decisiontree
model = tree.DecisionTreeClassifier(max_depth=3)

# Learn model
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))
print(np.array(y_test))

# Visualization
plt.figure(figsize=(40, 25))
tree.plot_tree(model,
               feature_names=X_train.columns.to_list(),
               class_names=['0','1','2','3'],filled=True)

