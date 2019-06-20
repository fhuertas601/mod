from __future__ import print_function, absolute_import, division, unicode_literals
from mordred import Calculator, descriptors
from rdkit import Chem 
from rdkit.Chem import AllChem, Draw
from sklearn import datasets, svm
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
import joblib, sys, os.path, getopt
sys.path.append('functions')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import build_model as bm
import inout as io

### User definitions
#
#molecs = 'data/anticocc/own/cocc-db_curation.csv'
model_name = 'newmodel'
try:
	opts, args = getopt.getopt(sys.argv[1:],"vp:hm:d:c:",["create-model"])
except getopt.GetoptError as err:
	print('Error!',err)
	sys.exit(2)
for opt, arg in opts:
# The argument of -m is the filename with activity info
	if opt == '-h':
		io.print_help()

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

a=verb=False
d=''
nproc=2
for opt, arg in opts:
# Verbose output
	if opt == '-v':
		verb=True
		print('Verbose output')
# Number of processors to be used
	elif opt in ("-p"):
		nproc=int(arg)
# Create model from descriptors file
	elif opt in ("-m","--create-model"):
# The argument of -m 'arg' is the filename with activity info
		print('\n    TASK')
		print('load descriptors, build model and evaluate it')
		if not os.path.exists(path+'/'+arg):
			print('\n',arg,'does not exist in',path)
			sys.exit()
# Descriptors (from <input>) and activity values (from <filename>) are loaded
		print('\n-> Loading the descriptors and the activity data')
		x,y=bm.load_desc(path,inp,arg)
		print('  Descriptor values loaded from ',path+'/'+inp,sep='')
		print('  Activity values loaded from ',path+'/'+arg,sep='')
# FUNCTION: build model from descriptors and activity file
# Explanation and function in build_model.py file
		bm.build_model(x,y,verb,path,nproc)
		sys.exit()
	elif opt in ("-d"):
# 
		csv_file=arg
	elif opt in ("-c"):
		print('\n    TASK')
		print('load descriptors of the prediction data and model and predict')
		bm.predict(arg,path,inp,csv_file)
		sys.exit()
		 
print('\n    TASK')
print('load database and calculate descriptors')
	
mordred_desc = '/home/gallo/Documents/articles/codes/mordred_supp_info/mordred_descriptor_list.csv'

# Read descriptors from file
a=True

# Definition of the calc variable, which evaluates the descriptors
calc = Calculator(descriptors, ignore_3D=True)
# Number of descriptors to calculate
ndesc = -1
# If ndesc = list, calculate this list of descriptors
# If ndesc = integer, calculate descriptors from 0 to ndesc
# If ndesc = -1, calculate all descriptors 
if ndesc==-1:
	ndescs=len(calc)
elif isinstance(ndesc, int):
	ndescs=ndesc
else:
	ndescs=ndesc

# Read the data
# Database must have 5 fields (columns) and must start with the following line:
# ID,Compound name, SMILES, CAS, Activity
if ".csv" in inp:
	dat = pd.read_csv(inp) # Database
else:
	print('Ups! If you really are using a .csv file, tell me!')
	sys.exit()
desc_list = pd.read_csv(mordred_desc) # Mordred descriptor list

### Code
#
"""
# Formated printing Pandas
for i, row in dat.iterrows():
	print(row)
"""
if verb:
	print('\nThese descriptors are being calculated:')
	names=desc_list.loc[:,'name']
	print('ID    Descriptor name')
	for i in range(ndescs):
		print('{:<5} {:13}'.format(i,names[i]))

	print('The size of the matrix with descriptors and activity is')
	print(d.shape)
# Descriptor list is stored in x as a float numpy array
# Activity list is stored in y as an integer numpy array
x=np.zeros([dat.shape[0],ndescs])
y=np.zeros(dat.shape[0])
x,y=f.calculate_desc(calc,dat,desc_list,ndescs,verb)
# d array has first column as activity, and next columns as descriptors
d=np.column_stack([y,x])

##	with open(model_name+"/descriptor_list.txt",'w') as file:
##		np.savetxt(file,x,fmt='%.6f')


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
# Save the model to folder model_file/model_file
joblib.dump(m, model_name+"/"+model_name+".pkl", compress=3)
