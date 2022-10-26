import numpy as np
import itertools
import NNP
import os
import glob

params={
"eta": [0.035,0.06,0.1],
"rmin":0.5,
"rmax":6.5,
"r_s": [1.0],
"kappa":[2.5,8.5],
"labd": [1.0, -1.0],
"xi": [0.001, 0.002, 0.003]
}

"""
mol = [['O', '-0.005703', '0.385159', '-0.000000'],
        ['H', '-0.796078', '-0.194675', '-0.000000'], 
        ['H', '0.801781', '-0.190484', '0.000000']]

mol1 = [['O', '-0.005703', '0.385159', '-0.000000'],
        ['H', '-0.796078', '-0.194675', '-0.000000'], 
        ['H', '0.801781', '-0.190484', '0.000000'],
        ['O', '-0.005703', '0.900000', '-0.000000']]
"""


def acsf(mol):
    molall=[]
    for i in range(len(mol)):
        #G2_input
        G2_data=[]
        pairwise_permutations = np.array(list(itertools.permutations(mol,2)))
        #print(pairwise_permutations)
        for eta in params["eta"]:
            for r_s in params["r_s"]:
                    g2=NNP.G2(pairwise_permutations[i][0][1:], pairwise_permutations[i][1][1:], params["rmin"], params["rmax"],eta, r_s,)
                    G2_data.append(g2) 
        #print(len(G2_data))
        G4_data=[]
        three_atom_permutations = list(itertools.permutations(mol,3))
        #print(three_atom_permutations)
        for labd in params["labd"]:
            for xi in params["xi"]:
                for eta in params["eta"]:
                    g4_sum=0
                    for j in range(3):
                        g4_sum+=NNP.G4(three_atom_permutations[j][0][1:], three_atom_permutations[j][1][1:], three_atom_permutations[j][2][1:], eta, labd, xi, params["rmin"], params["rmax"]) # issue?
                    G4_data.append(g4_sum)
        G2_data=np.array(G2_data)
        G4_data=np.array(G4_data)
        molall.append(np.concatenate((G2_data,G4_data),axis=0))
    return np.asarray(molall)
#print(acsf(mol))
path = glob.glob("systems/group*/")
Tot=[]
Tot1=[]
for i in path:
    atom=np.load(i+"/atom.npy")
    print(atom)
    energy=np.load(i+"/energy.npy")
    for test in atom:
        mol1=[]
        test[:,1:]*0.5291772
        mol = test.tolist()
        for s in mol:
            s=[str(i) for i in s]
            mol1.append(s)
        symm=acsf(mol1)
        Tot.append(symm)
    Tot1.append(energy)
Tot1=np.concatenate((Tot1[0],Tot1[1],Tot1[2],Tot1[3]))
np.save("symm.npy", np.asarray(Tot))
np.save("energy.npy",np.array(Tot1))
#print(np.asarray(Tot)[0])