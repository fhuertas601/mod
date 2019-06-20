from __future__ import print_function 
from mordred import Calculator, descriptors
from rdkit import Chem 
from rdkit.Chem import AllChem, Draw
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

import sys
sys.path.append('../../functions')
import pandas as pd
import numpy as np
import functions as f

# Number of compounds (lines) that is present in the descriptors .csv file
nl=5
# Verbose option
#v=False
v=True

### Populate dcptrs with the descriptors filenamee
#
file_dscrpts='descriptors.csv'

### Read the data
dat = pd.read_csv(file_dscrpts) # Database with all .csv data
# Smiles and id for the molecules
molecid=['SMILES','MOLECULEID']
ids = np.array(dat.filter(molecid))
# Descriptors for the models
dscrpts=['C-003','C-008','C-026','H-047','PCR','PCD','H_G']
cols = np.array(dat.filter(dscrpts))
if v:
	print(cols)
	print(ids)

### Invoque the models of anticoccidial activity
#
for i in range(0,nl):
	mod4=f.mod4(cols[i][0],cols[i][1],cols[i][2],cols[i][3])
	mod8=f.mod8(cols[i][4],cols[i][5])
	mod21=f.mod21(cols[i][6])
	p=f.tds(mod4,mod8,mod21)

# Calculate joint probability of anticoccidial activity
	print()
	print(i+1,ids[i][0])
	print('-------')
	if v:
		print('mod4:',mod4)
		print('mod8:',mod8)
		print('mod21:',mod21)

	if p > 0.5:
		print('The compound is classified as anticoccidial with confidence:')
	else:
		print('The compound is classified as non-active against anticoccidia with confidence:')

	print('P = ',p)

