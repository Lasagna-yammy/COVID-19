#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:57:54 2023

@author: mainaoshige
"""

# Loading the library

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
from sklearn.ensemble import RandomForestClassifier


# Load CSV file
data_frame = pd.read_csv('4deep_clinicalbloodAI.csv')

# Remove missing values
data_frame = data_frame.dropna()

# Specifying the target variable
y = data_frame["Severity"]


# #Set the name of the explanatory variable Clinical data
# X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]]

# # Set the name of the explanatory variable　Clinical and blood data
# X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]]

# # Set the name of the explanatory variable　Clinical, blood and chest radiography data(By doctor)#X = data_frame[["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","Ddimer","C_reactive_protein","Lactate_Dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_Bydoctor"]]
# X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_Bydoctor"]]

# # Set the name of the explanatory variable　Clinical, blood and chest radiography data(By AI)
X = data_frame[["Time_from_onset","Age","Sex","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Ureanitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph_ByAI"]]


# Split training data and varification data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))


