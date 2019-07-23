from rdkit import Chem 
from datetime import datetime
import numpy as np
import pandas as pd
import math, os, sys

def calculate_desc(calc,dat,path,inp,date_string,verb):
    desc = {}
    desc_data=str(path+'/model_data_'+date_string)
    os.mkdir(desc_data)
# Arrays are defined as floats from the beginning
    desc_array = np.zeros([dat.shape[0],len(calc)])
    activity = np.zeros([dat.shape[0]])
    col=['Activity','ID','Compound name','SMILES','CAS']
    for i in col:
    	if i not in dat.columns:
     		print(i,'column missing')
     		sys.exit()
    for i, row in dat.iterrows():
        if verb:
            print()
            print('ID','    Compound name')
            print('-------------------------------')
            print(dat.loc[i,'ID'],'   ',dat.loc[i,'Compound name'].replace(";",","))
            print(dat.loc[i,'SMILES']) 
            print('Activity =',dat.loc[i,'Activity']) 
        activity[i]=dat.loc[i,'Activity']
        comp_name=str(dat.loc[i,'Compound name'])
        if dat.loc[i,'ID'] < 10:
        	filename=str(desc_data+'/'+str(dat.loc[i,'ID']).zfill(2)+'_'+comp_name.replace("/","")+'.desc')
        else:
        	filename=str(desc_data+'/'+str(dat.loc[i,'ID'])+'_'+comp_name.replace("/","")+'.desc')
        with open(filename.replace(" ",""),'w+') as f:
# Print the name (stored in dscpts_name) and calculated descriptor
        	if dat.loc[i,'SMILES'] is not None:
        		desc_value = calc(Chem.MolFromSmiles(row['SMILES']))
        		for key, value in desc_value.items():
        			f.write(str(key)+' '+str(value)+'\n')
        with open(path+'/'+inp+'_names.dat','w+') as f:
        	for key, value in desc_value.items():
        		f.write(str(key)+'\n')
        desc_array[i]=list(desc_value.values())

    return desc_array, activity

def calculate_pred(calc,dat,path,inp,date_string,verb):
    desc = {}
    desc_data=str(path+'/prediction_data_'+date_string)
    os.mkdir(desc_data)
# Arrays are defined as floats from the beginning
    desc_array = np.zeros([dat.shape[0],len(calc)])
    for i, row in dat.iterrows():
        if verb:
            print()
            print('ID','    Compound name')
            print('-------------------------------')
            print(dat.loc[i,'ID'],'   ',dat.loc[i,'Compound name'].replace(";",","))
            print(dat.loc[i,'SMILES']) 
        comp_name=str(dat.loc[i,'Compound name'])
        if dat.loc[i,'ID'] < 10:
        	filename=str(desc_data+'/'+str(dat.loc[i,'ID']).zfill(2)+'_'+comp_name.replace("/","")+'.desc')
        else:
        	filename=str(desc_data+'/'+str(dat.loc[i,'ID'])+'_'+comp_name.replace("/","")+'.desc')
        with open(filename.replace(" ",""),'w+') as f:
# Print the name (stored in dscpts_name) and calculated descriptor
        	if dat.loc[i,'SMILES'] is not None:
        		desc_value = calc(Chem.MolFromSmiles(row['SMILES']))
        		for key, value in desc_value.items():
        			f.write(str(key)+' '+str(value)+'\n')
        desc_array[i]=list(desc_value.values())

# Indices are integers
    ind=[i for i in range(0,desc_array.shape[0])]
# Take dropped descriptors to drop again them in the calculation of the
# prediction set
    dropp_names= [s for s in os.listdir(path) if "_nan_dropped.dat" in s]
    with open(path+'/'+dropp_names[0],'r') as f:
    	dropp_names = [line.split(' ')[0] for line in f]
    names= [s for s in os.listdir(path) if "_names.dat" in s]
    with open(path+'/'+names[0],'r') as f:
    	names=f.read().splitlines()

# Store descriptors as df, with integers as indices and descriptor names as
# columns
    data=pd.DataFrame(data=np.float_(desc_array[0:,0:]),index=ind,columns=names)
# AtomTypeEState that are NaN, set to zero
    AtomTypeEState_names=names[933:1249]
    zero=[0.0]*len(AtomTypeEState_names)
    AtomTypeEState=dict(zip(AtomTypeEState_names, zero))
# Assign AtomTypeEState descriptors that are NaN with value 0.0
    data=data.fillna(value=AtomTypeEState)
# Drop columns that have been removed for model building
    clean=data.drop(columns=dropp_names)

    return clean 

# Function that evaluates the models for anticoccidials (Afinidad LXVIII,
# 554, Julio - Agosto 2011, Clasificación y cribado virtual de candidatos a
# fármaco anticoccidiales mediante el empleo de una estrategia
# probabilística de combinación de la información)

def mod4(c003,c008,c026,h047):
	mod4=2.036-0.785*c003-0.653*c008+.544*c026
	return mod4

def mod8(pcr,pcd):
	mod8=-15.543+13.617*pcr-0.091*pcd
	return mod8

def mod21(h_g):
	mod21=3.188-0.014*h_g
	return mod21

# Function that calculates the 'Regla del testimonio concurrente' (?: see
# functions above under 'anticocc'-> mod4, mod8, mod21).

def tds(p1,p2,p3):
	b_c = 1-((1-p1)*(1-p2)*(1-p3))
	
	return b_c

def GetPrincipleQuantumNumber(atNum):
  """ Get principal quantum number for atom number """
  if atNum <= 2:
    return 1
  elif atNum <= 10:
    return 2
  elif atNum <= 18:
    return 3
  elif atNum <= 36:
    return 4
  elif atNum <= 54:
    return 5
  elif atNum <= 86:
    return 6
  else:
    return 7

def chir(chiraltag):
	if chiraltag == 'CHI_UNSPECIFIED':
		exp=0.0
		ew=math.exp(exp)            
	elif chiraltag == 'CHI_TETRAHEDRAL_CCW':
		exp=-1.0
		ew=math.exp(exp)
	elif chiraltag == 'CHI_TETRAHEDRAL_CW':
		exp=1.0
		ew=math.exp(exp)

	return(exp,ew)

def col_to_row(path,ndescs,date_string):
# Get descriptors form the .desc files into an array, that can be used to
# generate the model/make the prediction. Files must end as .desc, and
# start with a number. They are already saved that way, so the code can be
# directly used with the files from the original calculation
	a=0
# <mat> contains the descriptors
	mat = np.zeros((len(os.listdir(path)),ndescs))
	for f in sorted(os.listdir(path)):
# Consider only files ending with .desc
		if f.split('.')[1] == 'desc':
			y=open(path+'/'+f,"r")
			lines=y.readlines()
			result=[]
# Convert value to float, but if that's not possible (because the column is
# a string and a ValueError is raised, print NaN value instead
			for x in lines:
				try:
					result.append(float(x.split(' ')[2].rstrip()))
				except:
					result.append(np.nan)
			y.close()
			mat[a]=result
			a=a+1
# Save descriptores stored in <mat> too 'f' as numpy array. Each file is a
# molecule with descriptors in rows
	print('-> Descriptors extracted from .desc files:')
	print('saved to: \n'+path+'/'+date_string+'_descriptors\n')
	with open(path+'/'+date_string+'_descriptors','w+') as f:
		np.savetxt(f,mat,fmt='%.6f')

