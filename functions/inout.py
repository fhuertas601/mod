import sys, getopt
from rdkit import Chem
import numpy as np
import pandas as pd
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
        print('  -v                  verbose output\n')
        print('  -p <n_procs>        request <n_procs> number of processors to be used for model evaluation\n')
        print('  -m <activity>       create model from given descriptors')
        print('                        <input> is a plain text file containing a matrix with the descriptors')
        print('                        <activity> is the file with the activity in plain text format\n')
        print('  -w <model>          calculate the prediction dataset descriptors, estimate the applicability')
        print('                              domain and evaluate the model')
        print('                        <input> is the path to the .csv file with the target molecules')
        print('                        <model> is the filename of the model, that is looked for in the same path')
        print('                              as <input>')
        print('  -d <csv_file>       read <csv_file> with info on prediction dataset to produce')
        print('                              an easy-to-read file with the results of the prediction ')
        print('                        <input> is a plain text file containing a matrix with the descriptors')
        print('                              of the molecules whose activity is going to be predicted\n')
        print('  -c <model>          apply <model> to predict <input>')
        print('                        <input> is a plain text file containing a matrix with the descriptors')
        print('                              of the molecules whose activity is going to be predicted')
        print('                        <model> is a .pkl file with the model\n')
        print('  -n <desc_file>      read <desc_file> with descriptors, search for NaN values,')
        print('                              process them and create a new clean descriptors file\n')
        print('  -r <# descriptors>  number of descriptors to be calculated\n')
        sys.exit()
#
# FUNCTION: prints prediction in .csv format
# d: <filename> taken from <csv> and where prediction is going to be stored
# as <filename>.txt
# pred: database with descriptor matrix for prediction 
# predict: vector with predictions
# path: file 'f' is saved in path (where model for prediction is located)
def print_csv(pred,predict,da,path,inp):
	with open(path+'/'+inp+"_prediction.txt","w+") as f: 
		write = csv.writer(f) 
		write.writerow('ID '+'SMILES '+'Compound name '+'Prediction '+'App.  Domain')
		print('\n-> Result of the prediction:')
		print('ID','  SMILES ','  Compound name','  Prediction','  App.  Domain')
		for i, row in pred.iterrows():
			r=pred.loc[i,'ID'],pred.loc[i,'SMILES'],pred.loc[i,'Compound name'],int(predict[i]),str(da[i])
			print(pred.loc[i,'ID'],pred.loc[i,'SMILES'],pred.loc[i,'Compound name'],int(predict[i]),str(da[i]))
			write.writerow(r)

# Prints descriptors (number) with NaN values
def detect_nan(path,inp,verb):
	print('-> Evaluating descriptors with NaN values:')
	print('taken from:\n'+' '+path+'/'+inp)
	x = np.loadtxt(path+'/'+inp)
	names=[]
	print('\n->Descriptor names:')
# Descriptor names are taken from the file '<path>+*+_names.dat' in <path>,
# so no other file should be there. This file contains the names of the
# descriptors that have been calculated to generate the model (<dscs>)
	a= [s for s in os.listdir(path) if "_names.dat" in s]
	print('taken from:\n'+' '+path+'/'+a[0])
	with open(path+'/'+a[0],'r') as f:
		names=f.read().splitlines()
	dnan_num=[]
	dnan_mol=[]
# Indices are integers
	ind=[i for i in range(0,x.shape[0])]
# Data is saved from the descriptors file in <data>
	data=pd.DataFrame(data=np.float_(x[0:,0:]),index=ind,columns=names)
# Descriptors of the type AtomTypeEState are stored as dictionary (in
# <AtomTypeEState>) with 0.0 as value
	AtomTypeEState_names=names[933:1249]
	zero=[0.0]*len(AtomTypeEState_names)
	AtomTypeEState=dict(zip(AtomTypeEState_names, zero))
# Assign AtomTypeEState descriptors that are NaN with value 0.0
	data=data.fillna(value=AtomTypeEState)
	print('\nOf the total '+str(x.shape[1])+' descriptors:')
# Save NaN descriptors and the number that have been filled
	print('\n->Number of filled NaN values:',len(AtomTypeEState_names))
	print('saved to:\n'+' '+path+'/'+inp+'_nan_filled.dat')
	with open(path+'/'+inp+'_nan_filled.dat','w+') as f:
		for i in AtomTypeEState_names:
			f.write(i+'\t'+str(0.0)+'\n')
	data_bool=data.isna()
# Save in .csv format the boolean analysis of NaN/non-NaN values
	with open(path+'/'+inp+'_nan_analysis.dat','w+') as f:
		data_bool.to_csv(f, sep='\t')
# Save to '*_nan.dat' descriptors with NaN values that will be dropped
	with open(path+'/'+inp+'_nan_dropped.dat','w+') as nan:
# Create file with the indices of the valid descriptors and save them
	  	with open(path+'/'+inp+'_valid_no-nan.dat','w+') as nonan:
	  		for i in names:
# <r> contains the column for each descriptor (all values for the same
# descriptor)
	  			r=data_bool[str(i)].values.tolist()
	  			if sum(r):
	  				nan.write(i+' '+str(sum(r))+'\n')
	  			else:
	  				nonan.write(i+'\n')
# Drop columns with one or more NaN values
	m=data.dropna(axis=1)
	print('\n-> Number of dropped NaN values:',x.shape[1]-m.shape[1])
	print('saved to:\n',path+'/'+inp+'_nan_dropped.dat')
# Save in .csv format the descriptors file clean of NaN values
	print('\n-> Number of valid (no-NaN) descriptors:',m.shape[1])
	print('saved to:\n',path+'/'+inp+'_no-nan.dat')
	with open(path+'/'+inp+'_clean.txt','w+') as f:
		m=m.to_numpy()
		np.savetxt(f,m,fmt='%.6f')
	return m	
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
"""
# Commented out because dropna() is used
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
"""
