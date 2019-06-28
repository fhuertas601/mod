from rdkit import Chem
from datetime import datetime
import numpy as np
import math, os
def calculate_desc_pred(calc,dat,desc_list,ndescs,verb):
    desc = {}
# Arrays are defined as floats from the beginning
    desc_array = np.zeros([dat.shape[0],ndescs])
    for i, row in dat.iterrows():
        store= []
        if verb:
            print()
            print('ID','    Compound name')
            print('-------------------------------')
            print(dat.loc[i,'ID'],'   ',dat.loc[i,'Compound name'].replace(";",","))
            print(dat.loc[i,'SMILES']) 
        for j in range(0,ndescs):
            if dat.loc[i,'SMILES'] is not None:
            # Print the name (stored in desc_list) and calculated descriptor
            	desc_value = calc(Chem.MolFromSmiles(row['SMILES']))[j];
            	store=np.append(store,desc_value)
            	if verb: 
            		print(j,' ',desc_list.loc[j,'name'],'=',desc_value)
        desc_array[i]=store

    return desc_array

def calculate_desc(calc,dat,desc_list,ndescs,path,verb):
    desc = {}
# Create folder 'desc_data' with today's date and time to save the files
# with the info of the descriptors
    date=datetime.now()
    date_string = date.strftime("%d.%m.%Y_%H.%M")
    desc_data=str(path+'/desc_data_'+date_string)
    os.mkdir(desc_data)
# Arrays are defined as floats from the beginning
    desc_array = np.zeros([dat.shape[0],ndescs])
    activity=np.zeros(dat.shape[0],dtype=int)
    for i, row in dat.iterrows():
        store= []
# print(dat.loc[i,'ID'])
        if row['Activity'] == 'Sensitizer':
        	activity[i]=1
        elif row['Activity'] == 'Non-Sensitizer':
        	activity[i]=0

        if verb:
# For verbose output, a bunch of information is printed via standard output
# (terminal)
            print()
            print('ID','    Compound name')
            print('-------------------------------')
            print(dat.loc[i,'ID'],'   ',dat.loc[i,'Compound name'].replace(";",","))
            print(dat.loc[i,'SMILES']) 
            if row['Activity'] == 'Sensitizer':
            	print(row['Activity'],'1')
            elif row['Activity'] == 'Non-Sensitizer':
            	print(row['Activity'],'0')
            elif row['Activity'] == 0:
            	print('Inactive compound')
            elif row['Activity'] == 1:
            	print('Active compound')
            else:
            	print('ERROR: No correct activity specified')
        comp_name=str(dat.loc[i,'Compound name'])
        filename=str(desc_data+'/'+str(dat.loc[i,'ID'])+'_'+comp_name.replace("/","")+'.desc')
        chars="(); "
        for k in chars:
        	filename=filename.replace(k,"")
        with open(filename,'w+') as f:
        	for j in range(0,ndescs):
        		if dat.loc[i,'SMILES'] is not None:
            # Print the name (stored in desc_list) and calculated descriptor
        			desc_value = calc(Chem.MolFromSmiles(row['SMILES']))[j];
        			store=np.append(store,desc_value)
# Print the descriptors for each SMILES and whether it is taken as test or
# training molecule
        			if verb: 
        				print(j,' ',desc_list.loc[j,'name'],'=',desc_value)
        			f.write(str(j)+' '+str(desc_list.loc[j,'name'])+' '+str(desc_value)+'\n')
# desc_array stores all the descriptors. 
# Row: compound by ID (i.e.: row 0 contains all descriptors of compound
# with ID = 0)
# Column: descriptor by ID (i.e.: column 0 has #compounds entries of ABC
# descriptor)
#        desc_array([i])=store[:] print(desc_array.shape) print(desc_array.shape)
        desc_array[i]=store

    return desc_array, activity

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

