import sys, getopt
from rdkit import Chem
import numpy as np
import math,joblib, sys, os.path, csv
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.impute import SimpleImputer

def print_help():
# Print options of the code
        print('Code help:')
        print('\nUsage: python file.py -options <input>')
        print('\n<input> = .csv file if no other option is given. The code calculates descriptors from input. ')
        print('<input> must be the whole path to a .csv file with 5 fields:')
        print(' ID, Compound name, SMILES, CAS, Activity')
        print('\nOptions:')
        print('  -v               verbose output')
        print('  -m <activity>    create model from given descriptors')
        print('                      <input> is a plain text file containing a matrix with the descriptors')
        print('                      <activity> contains the activity in plain text format\n')
        print('  -c <model>       apply <model> to predict <input>')
        print('                      <input> is a plain text file containing a matrix with the descriptors')
        print('                              of the molecules whose activity is going to be predicted')
        print('                      <model> .pkl file with the model\n')
        sys.exit()
#
# FUNCTION: prints prediction in .csv format
# d: <filename> taken from <csv> and where prediction is going to be stored
# as <filename>.txt
# pred: database with descriptor matrix for prediction 
# predict: vector with predictions
def print_csv(d,pred,predict):
	with open(d+"_prediction.txt","w+") as f: 
		write = csv.writer(f) 
		for i, row in pred.iterrows():
			r=pred.loc[i,'ID'],pred.loc[i,'SMILES'],pred.loc[i,'Compound name'],int(predict[i])
			write.writerow(r)

