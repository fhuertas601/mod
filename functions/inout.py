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
#
# FUNCTION: prints usage info in terminal
def print_help():
# Print options of the code
        print('Code help:')
        print('\nUsage: python mod.py -options <input>')
        print('\n<input> = .csv file if no other option is given. The code calculates descriptors from input. ')
        print('<input> must be the whole path to a .csv file with 5 fields:')
        print(' ID, Compound name, SMILES, CAS, Activity')
        print('\nOptions:\n')
        print('  -v               verbose output\n')
        print('  -p <n_procs>     request <n_procs> number of processors to be used for model evaluation\n')
        print('  -m <activity>    create model from given descriptors')
        print('                      <input> is a plain text file containing a matrix with the descriptors')
        print('                      <activity> is the file with the activity in plain text format\n')
        print('  -d <csv_file>    read <csv_file> with info on prediction dataset to produce')
        print('                              an easy-to-read file with the results of the prediction ')
        print('                      <input> is a plain text file containing a matrix with the descriptors')
        print('                              of the molecules whose activity is going to be predicted\n')
        print('  -c <model>       apply <model> to predict <input>')
        print('                      <input> is a plain text file containing a matrix with the descriptors')
        print('                              of the molecules whose activity is going to be predicted')
        print('                      <model> .pkl file with the model\n')
        print('  -n <desc_file>   read <desc_file> with descriptors, search for NaN values,')
        print('                              remove them and create a new clean descriptors file')
        sys.exit()
#
# FUNCTION: prints prediction in .csv format
# d: <filename> taken from <csv> and where prediction is going to be stored
# as <filename>.txt
# pred: database with descriptor matrix for prediction 
# predict: vector with predictions
# path: file 'f' is saved in path (where model for prediction is located)
def print_csv(d,pred,predict,path):
	with open(path+'/'+d+"_prediction.txt","w+") as f: 
		write = csv.writer(f) 
		for i, row in pred.iterrows():
			r=pred.loc[i,'ID'],pred.loc[i,'SMILES'],pred.loc[i,'Compound name'],int(predict[i])
			write.writerow(r)

# Prints descriptors (number) with NaN values
def detect_nan(arg,verb):
	print('-> Evaluating descriptors with NaN values. Choose -v for more info')
	x = np.loadtxt(arg)
	dnan_num=[]
	dnan_mol=[]
	for i in range(0,x.shape[1]):
		dnan=0
		for j in range(0,217):
			if 'nan' in str(x[j][i]) == 'nan':
# Counts number of nans per descriptor
				dnan=dnan+1
		if dnan != 0:
# Choose to select only cases for which number of NaN values is greater
# that <value> (10 here)
#		if dnan != 0 and dnan > 10:
			dnan_num.append(dnan)
			dnan_mol.append(i)
	
	if verb:
		print('\nDescriptor  NaN')
		for i in range(0,len(dnan_mol)):
			print(dnan_mol[i],'          ',dnan_num[i])
	print('\nTotal number of descriptors with non-zero NaN cases\n',len(dnan_mol),'\n')
	create_non_nan(dnan_mol,x,arg)
"""
# Makes no sense, all molecules have some descriptor (normally many) with
# NaN results
#
	mnan_num=[]
	mnan_mol=[]
	for i in range(0,217):
		mnan=0
		for j in range(0,x.shape[1]):
			if 'nan' in str(x[i][j]) == 'nan':
# Counts number of nans per molecule
				mnan=mnan+1
		if mnan != 0:
			mnan_num.append(mnan)
			mnan_mol.append(i)
	for i in range(0,len(mnan_mol)):
		print(mnan_mol[i],mnan_num[i])
"""

# Takes the columns (descriptors) that contain NaN values
# and creates a new one without NaN values
def create_non_nan(dnan_mol,x,arg):
	print('->  Creating new file without NaN values in:',arg+'_no_nan\n')
	temp=np.empty((x.shape[0],x.shape[1]-len(dnan_mol)))
	p=0
	for i in range(0,x.shape[1]):
		if i not in dnan_mol:
			temp[:,p]=x[:,i]
			p=p+1

	with open (arg+'_no_nan','w+') as f:
		np.savetxt(f,temp,fmt='%6.9f')
