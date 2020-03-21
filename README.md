# music_classification
1. This Project contains 2 file train.py and pred.py and one directory data which contain extracted song features.
2. First Run
python train.py --input //path to your training csv//
After succesful execution xgboost.pickle.dat file will be created.
3. Now run
python pred.py --test_file //path to evalution file
After succesful execution out.csv will be created with predicted Mood tag Happy or Sad.
