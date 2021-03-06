# -*- coding: utf-8 -*-
"""w2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ipxegtL6kLl80LkvLOqKZ-ln8puDQ3-M
"""

# !wget https://data-science-ml-case-study.s3.ap-south-1.amazonaws.com/training.csv
# !wget https://data-science-ml-case-study.s3.ap-south-1.amazonaws.com/evaluation.csv

# ! pip install essentia

# !unzip "data.zip"

import essentia
import essentia.standard
import essentia.streaming
from essentia.standard import *
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import argparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot






# Create feature Dataframe
def create_feature_json_v3(d1):
  with open(d1) as f:
    json_object = json.load(f)
  e= Energy()

  average_loudness =json_object['lowlevel']['average_loudness']
  dissonance_mean = json_object['lowlevel']['dissonance']['mean']
  energy_erbbands =  e.compute(json_object['lowlevel']['erbbands']['mean'])
  energy_melbands = e.compute(json_object['lowlevel']['melbands']['mean'])
  mfcc = json_object['lowlevel']['mfcc']['mean']
  beat_loudness_ratio=json_object['rhythm']['beats_loudness_band_ratio']['mean']
  chords_changes_rate = json_object['tonal']['chords_changes_rate']
  chord_histogram = json_object['tonal']['chords_histogram']
  bpm = json_object['rhythm']['bpm']
  chord_key = json_object['tonal']['chords_key']
  chord_scale= json_object['tonal']['chords_scale']
  mfcc = json_object['lowlevel']['mfcc']['mean']

  out = {'chord_key' : chord_key ,'chord_scale':chord_scale,'bpm':bpm,'average_loudness': average_loudness , 'dissonance_mean' : dissonance_mean , 'energy_erbbands' : energy_erbbands, 'energy_melbands' : energy_melbands ,'chords_changes_rate': chords_changes_rate}
  root ="mfcc"
  root_1="chord_hist"
  i =0
  for it in mfcc:
    key_name=root+str(i)
    out[key_name] = it
    i=i+1
  i= 0
  for it in chord_histogram:
    key_name=root_1 + str(i)
    out[key_name] = it
    i=i+1

  return out


ap= argparse.ArgumentParser()
ap.add_argument("-i" ,"--input" , required=True , help="Path to input train file")
args=vars(ap.parse_args())

train = pd.read_csv(args['input'])


# x_train = pd.DataFrame()
feature_list=[]
i = 0
for index, row in train.iterrows():
  # d1=train['AF_PATH'][0]
  temp_feature=create_feature_json_v3(row['AF_PATH'])
  temp_feature['TRACK_ID'] = row['TRACK_ID']
  temp_feature['y'] = row['MOOD_TAG']
  # temp_feature
  feature_list.append(temp_feature)
x_train_v2 = pd.DataFrame(feature_list)

dummy_chord_scale = pd.get_dummies(x_train_v2['chord_scale'])
dummy_chord_key = pd.get_dummies(x_train_v2['chord_key'])
x_train_v2=pd.concat((x_train_v2,dummy_chord_key), axis=1)
x_train_v2=pd.concat((x_train_v2 , dummy_chord_scale), axis = 1)
x_train_v2.drop(['chord_key' , 'chord_scale'] , axis= 1, inplace=True)
dummy_y=pd.get_dummies(x_train_v2['y'])
x_train_v2=pd.concat((x_train_v2,dummy_y) , axis = 1)
x_train_v2.drop(['y','SAD'], axis=1, inplace=True)
y=x_train_v2['HAPPY']
X=x_train_v2.drop(['HAPPY','TRACK_ID'] , axis=1, inplace=True)
X=x_train_v2

# Tuning Hyper parameter
print("Tuning Hyper parameter")
XGB = XGBClassifier(random_state=0)
parameter_grid = {'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [3, 4, 5]}

cross_validation = StratifiedKFold(n_splits=3,random_state=0,shuffle=True)

grid_search_XGB = GridSearchCV(XGB,param_grid=parameter_grid,cv=cross_validation,n_jobs=-1,verbose=0)

grid_search_XGB.fit(X, y)




print('Best score: {}'.format(grid_search_XGB.best_score_))
print('Best parameters: {}'.format(grid_search_XGB.best_params_))
print('Saving Best Model')
pickle.dump(grid_search_XGB, open("xgboost.pickle.dat", "wb"))
print("Saved model to: xgboost.pickle.dat")

