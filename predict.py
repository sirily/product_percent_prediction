#!/usr/bin/python

import sys, getopt, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def score(inputfile):
    X = pd.read_csv(inputfile)
    if X.isnull().values.any():
        print('Invalid input')

    else:
        with open('finalized_model.pkl', 'rb') as fin:
            scaler, clf = pickle.load(fin)

        if X.shape[1] > 18:
            print('Invalid shape! Must be 18 columns, only sensor data')
        else:
            inp = scaler.transform(X)
            return clf.predict(inp)

    

def main(argv):
   inputfile = ''
   outputfile = ''
   predictions = []
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('predict.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
        if opt == '-h':
            print('predict.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            predictions.append(score(arg))
            print(predictions)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            pd.DataFrame(predictions).to_csv(outputfile) 

if __name__ == "__main__":
   main(sys.argv[1:])