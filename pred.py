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



# This function tract features from json
def create_feature_test(d1):
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
ap.add_argument("-i" ,"--test_file" , required=True , help="Path to input train file")
args=vars(ap.parse_args())

test = pd.read_csv(args['test_file'])

feature_list=[]
i = 0
for index, row in test.iterrows():
  # d1=train['AF_PATH'][0]
  temp_feature=create_feature_test(row['AF_PATH'])
  temp_feature['TRACK_ID'] = row['TRACK_ID']
  # temp_feature['y'] = row['MOOD_TAG']
  # temp_feature
  feature_list.append(temp_feature)
x_test = pd.DataFrame(feature_list)



dummy_chord_scale = pd.get_dummies(x_test['chord_scale'])
dummy_chord_key = pd.get_dummies(x_test['chord_key'])
x_test=pd.concat((x_test,dummy_chord_key), axis=1)
x_test=pd.concat((x_test , dummy_chord_scale), axis = 1)
x_test.drop(['chord_key' , 'chord_scale'] , axis= 1, inplace=True)

x_test.to_csv('test_features.csv' , index=False)
track_id = x_test['TRACK_ID']
x_test.drop(['TRACK_ID'] , axis=1,inplace=True)
print("Feature Extraction Done")


print(x_test.head())
# print('hiii')

loaded_model = pickle.load(open("xgboost.pickle.dat", "rb"))
print("Loaded model from: xgboost.pickle.dat")

predictions = loaded_model.predict(x_test)
print(predictions)
test['MOOD_TAG']=predictions
test.loc[(test.MOOD_TAG == 0),'MOOD_TAG']='SAD'
test.loc[(test.MOOD_TAG == 1),'MOOD_TAG']='HAPPY'


test.to_csv('out_tuned_1.csv', index=False)

# accuracy = accurac_score(x_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))




