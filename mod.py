from __future__ import print_function, absolute_import, division, unicode_literals 
from mordred import Calculator, descriptors
from rdkit import Chem 
from rdkit.Chem import AllChem, Draw
from sklearn import datasets, svm
from datetime import datetime
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.impute import SimpleImputer

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras

#print(tf.__version__)
import joblib, sys, os.path, getopt, json
sys.path.append('functions')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import build_model as bm
import inout as io

# Default values
#
verb=False
# Create date string for file saving
date=datetime.now()
date_string = date.strftime("%d.%m.%Y_%H.%M")
# Read Mordred descriptors
mordred_desc = '/home/gallo/Documents/articles/codes/mordred_supp_info/mordred_descriptor_list.csv'
desc_list = pd.read_csv(mordred_desc) # Mordred descriptor list
# Definition of the calc variable, which evaluates the descriptors
calc = Calculator(descriptors, ignore_3D=True)
# Save the name and dimension
d_name=desc_list['name']
dscpts_dim=desc_list['dimension']
dscpts_name=[]
dscpts_idx=[]
#
# Number of descriptors to calculate
# If ndesc = list, calculate this list of descriptors
# If ndesc = integer, calculate descriptors from 0 to ndesc
# If ndesc = -1, calculate all descriptors 
# By default, all descriptors are calculated
ndescs=len(calc)-1
maxnum=len(d_name)
#
try:
	opts, args = getopt.getopt(sys.argv[1:],"vp:hm:c:nr:ew:",["create-model"])
except getopt.GetoptError as err:
	print('Error!',err)
	sys.exit(2)
for opt, arg in opts:
# The argument of -m is the filename with activity info
	if opt == '-h':
		io.print_help()
# Verbose output
	elif opt == '-v':
		verb=True
		print('Verbose output')
	elif opt in ("-r"):
# If input is int, the integer number of descriptors is calculated
		ndescs=int(arg)
		maxnum=ndescs
"""
# Calculate list of descriptors is not yet implemented
# If input is list, the list of descriptors is calculated
		elif isinstance(arg, list):
			ndescs=ndesc
			dscpts_name=[len(ndescs)]
			maxnum=len(ndescs)
			for i in len(ndescs):
				if dscpts_dim[i]=='2D':
					dscpts_name.append(d_name[i])
					print('is list',ndescs)
					sys.exit()
"""

# <dscpts_name> conatains the names of the descriptors that are being
# calculated. That is saved to a file '*_descnames' further down to be read
# when NaN values have to be removed
for i in range(1,maxnum):
	if dscpts_dim[i] !='3D':
		dscpts_name.append(d_name[i])
		dscpts_idx.append(i)
	elif len(dscpts_name) > maxnum:
		break
# dscs contains all the descriptors that are being calculated, with their
# respective index trough which they can be called to Calculator
dscs=np.stack((dscpts_idx, dscpts_name))
#
if args:
# Argument (no -option), is taken as the path for the descriptors file, is saved in dumb
	dumb=args[0]
# Path (without filename) is stored in 'path'
	path=dumb.rsplit("/",1)[0]
# Filename is stored in 'inp'
	inp=dumb.split("/")[len(dumb.split("/"))-1]
else:
	print('Please, provide a input filename')
	sys.exit(2)
#
d=''
nproc=2
for opt, arg in opts:
# Number of processors to be used
	if opt in ("-p"):
		nproc=int(arg)
# Create model from descriptors file
	elif opt in ("-m","--create-model"):
# The argument of -m 'arg' is the filename with activity info
		print('\n    TASK')
		print('Load descriptors, build model and evaluate it')
		if not os.path.exists(path+'/'+arg):
			print('\n',arg,'does not exist in',path)
			sys.exit()
# Descriptors (from <input>) and activity values (from <filename>) are loaded
		x,y=bm.load_desc(path,inp,arg)
# FUNCTION: build model from descriptors and activity file
# Explanation and function in build_model.py file
		bm.build_model(x,y,verb,path,nproc)
		sys.exit()
	elif opt in ("-n"):
# Asks to detect number of NaN values per descriptor, assuming all 2D
# descriptors have been calculated
		print('\n    TASK')
		print('Check and remove number of NaN values in the descriptors file\n')
		io.detect_nan(path,inp,verb)
		sys.exit()
	elif opt in ("-w"):
# .csv file with the target chemicals (ID, Compound name, SMILES, CAS)
		csv_file=pd.read_csv(path+'/'+inp)
# Calculate descriptors and remove/replace to make them agree with those
# used to build the model, store them in <clean> and save them in
# <desc_file>
		clean=fct.calculate_pred(calc,csv_file,path,inp,date_string,verb)
		desc_file=path+'/'+inp+'_prediction_descriptor_list.txt'
		with open(desc_file,'w+') as f:
			np.savetxt(f,clean,fmt='%.6f')
# Call model and calculate
		predict=bm.predict(arg,path,inp,drop,csv_file)
		with open(path+'/prediction.json', "w") as f:
			for i in predict:
				json.dump(int(i), f)
		sys.exit()
	elif opt in ("-c"):
# Load descriptors and evaluate the model
# Descriptors are assumed to be sanitized in comparison with the
# descriptors used to build the model, as contained in the file *_clean.txt
		print('\n    TASK')
		print('  Load descriptors of the prediction data and model and predict')
		bm.predict(arg,path,inp,arg,dscs)
		sys.exit()
	elif opt in ("-e"):
		print('\n    TASK')
		print('  Extract descriptors from the .desc files and save them as an array\n')
# Extract descriptors from .desc files and create np matrix with their
# values
		fct.col_to_row(path,ndescs,date_string)
		sys.exit()
#		 
print('\n    TASK')
print('  Load database and calculate descriptors')
# Read the data
# Database must have 5 fields (columns) and must start with the following line:
# ID,Compound name, SMILES, CAS, Activity
if ".csv" in inp:
	csv_file=pd.read_csv(path+'/'+inp)
else:
	print('Ups! If you really are using a .csv file, tell me!')
	sys.exit()
# Calculate descriptors
print('\n-> Calculating descriptors (3D disregarded)\n')
x,y=fct.calculate_desc(calc,csv_file,path,inp,date_string,verb)
desc_file=path+'/'+inp+'_descriptor_list.txt'
print('saved to:\n',desc_file)
# Save descriptores to 'f' as numpy array. Each file is a molecule with
# descriptors in rows
with open(desc_file,'w+') as f:
	np.savetxt(f,x,fmt='%.6f')
with open(path+'/activity.txt','w+') as f:
	np.savetxt(f,y,fmt='%1i')

# Deal with NaN values
print('-> Dealing with NaN values')
x=io.detect_nan(path,inp+'_descriptor_list.txt',verb)
# Build model
print('-> Building model')
bm.build_model(x,y,verb,path,nproc)

"""
# Generate a .png from a molecule
from rdkit.Chem import Draw
for i in range(0,2):
	# The file is given the name of the molecule
	name = dat.loc[i,'Compound name']+'.png'
	# The SMILES in Panda variable 'dat' is used to draw the molecule 
	print_mol = Chem.MolFromSmiles(dat.loc[i,'SMILES'])
	Draw.MolToFile(print_mol,name)
"""
