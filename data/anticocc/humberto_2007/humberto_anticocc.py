from __future__ import print_function
from mordred import Calculator, descriptors
from rdkit import Chem 
from rdkit.Chem import AllChem, Draw
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from mendeleev import element

import sys
sys.path.append('../../functions')
import pandas as pd
import numpy as np
import functions as f
import array as arr

tbl = Chem.GetPeriodicTable()

# Verbose output mode
v = True
# Exponent for the evaluation of \pi_k
k=9
with open('smiles.dat', 'r') as datfile:
	smi = datfile.read().splitlines()

# For looping over all the molecules (SMILES) in the input file 
for i in smi:
	mol=Chem.MolFromSmiles(i)
	nat=mol.GetNumAtoms()
	idx=[]
	for j in mol.GetAtoms():
		idx.append(j.GetAtomicNum())
		nats=len(mol.GetAtoms())
	if -v:
		print('-------')
# Print the adjacency matrix, number of atoms and indexes
		print(i)
		print(Chem.rdmolops.GetAdjacencyMatrix(mol))
		print('Number of atoms:',nats)
		print('Index','Element','Atomic number')
		for j in mol.GetAtoms():
			print(j.GetIdx(),tbl.GetElementSymbol(j.GetAtomicNum()),j.GetAtomicNum())
	khe_mtx=np.zeros([nat,nat])
	pauling_mtx=np.zeros([nat,nat])
	ind_mtx=Chem.rdmolops.GetAdjacencyMatrix(mol)
	n=0
	for j in mol.GetAtoms():
		at_aw=element(j.GetAtomicNum())
		neighs = element([i.GetAtomicNum() for i in j.GetNeighbors()])
		l=f.GetPrincipleQuantumNumber(j.GetAtomicNum())
		dv=tbl.GetNOuterElecs(j.GetAtomicNum())-j.GetTotalNumHs()
		d= j.GetTotalDegree()-j.GetTotalNumHs()
		ew = f.chir(str(j.GetChiralTag()))
		pauling_atom=at_aw.en_pauling
		pauling=[pauling_atom]
		khe_atom=((dv-d)/l**2)*ew[1]
		khe=[khe_atom]
# Save the chiral tag as string
		if v:
			print()
			print('---')
			print(tbl.GetElementSymbol(j.GetAtomicNum()))
			print('Idx',j.GetIdx(),'Atomic number',j.GetAtomicNum())
			print('H_atoms', j.GetTotalNumHs())
			print('Hybridization', j.GetHybridization())
			print('Chirality', j.GetChiralTag())
			print('e^', ew[0])
#			print('E/Z isomerism', j.GetStereo())
		#	print('Sigma electrons', j.GetTotalDegree())
		#	print('Valence electrons', tbl.GetNOuterElecs(j.GetAtomicNum()))
			print('Delta v', dv)
			print('Delta', d)
			print('L', l)
			print('KHE', khe)
			print('Pauling EN', pauling)
			print('Neighbours number =',len(j.GetNeighbors()))
		for i in j.GetNeighbors():
			l=f.GetPrincipleQuantumNumber(i.GetAtomicNum())
			dv=tbl.GetNOuterElecs(i.GetAtomicNum())-i.GetTotalNumHs()
			d= i.GetTotalDegree()-i.GetTotalNumHs()
			ew_i=f.chir(str(i.GetChiralTag()))
			khe_i= ((dv-d)/l**2)*ew_i[1]
			atom = element(i.GetAtomicNum())
			pauling_i = atom.en_pauling
# The electronegativity is multiplied by e^w (see equation (2) in
# H. González-Dı́az et al. / Bioorg.Med. Chem. 15 (2007) 962–968)
			khe.append(khe_i)
			pauling.append(pauling_i)
			if v:
				print('Atom', j.GetIdx(),'(',tbl.GetElementSymbol(j.GetAtomicNum()),') - ','Neighbour','(',tbl.GetElementSymbol(i.GetAtomicNum()),')', i.GetIdx())
				print('KHE =',khe_i)
				print('e^',ew_i[0])
				print('Pauling EN',pauling_i)

#		khe_neighs.append(khe)
		if v:
			print('All KHEs',khe)
			print('KHE sum',sum(khe))
			print('All Pauling NEs',pauling)
			print('Pauling sum',sum(pauling))
		for w in j.GetNeighbors():
			neigh_aw=element(w.GetAtomicNum())
			if sum(khe) == 0 and khe_i == 0:
				khe_mtx[n][n]=0.0
				khe_mtx[n][w.GetIdx()]=0.0
			else:
				khe_mtx[n][n]=khe_atom/sum(khe)
				khe_mtx[n][w.GetIdx()]=khe_i/sum(khe)
			if sum(pauling) == 0 and pauling_i == 0:
				pauling_mtx[n][n]=0.0
				pauling_mtx[n][w.GetIdx()]=0.0
			else:
				pauling_mtx[n][n]=pauling_atom/sum(pauling)
				pauling_mtx[n][w.GetIdx()]=pauling_i/sum(pauling)
		trace_pauling=np.zeros([nat])
		trace_khe=np.zeros([nat])
		for i in range(0,nats):
			trace_pauling[i]=pauling_mtx[i][i]**k
			trace_khe[i]=khe_mtx[i][i]**k
		n=n+1

	print()
	print(idx)
	print()
	print('Transition matrix - KHE')
	print(khe_mtx)
	print()
	print('Trace of the transition matrix - KHE')
	print(np.matrix.trace(khe_mtx))
	print()
	print('Trace^9 - KHE')
	print(np.matrix.trace(khe_mtx)**9)
	print()
	print('Transition matrix - Pauling')
	print(pauling_mtx)
	print()
	print('Trace of the transition matrix - Pauling')
	print(np.matrix.trace(pauling_mtx))
	print()
	print('Trace^9 - Pauling')
	print(sum(trace_pauling))
	print()
	print('Model')
	print(-0.532*nats+1.799*sum(trace_pauling)+16.335)
