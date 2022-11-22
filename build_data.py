"""
Filename: build_data.py
build data for training
"""
import itertools
import glob
import numpy as np
import ACSF

params={
"etaG2": [1,2,3],
"rmin":0.5,
"rmax":6.5,
"r_s": [1.0],
"kappa":[2.5,8.5],
"labd": [1.0, -1.0],
"etaG4": [1],
"zeta": [1, 2]
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
    """
    define mol as molecule
    """
    symmmol=[]
    for atom_i in range(len(mol)):
        G2=[]
        for eta in params["etaG2"]:
            g2=[]
            for r_s in params["r_s"]:
                for atom_j in range(len(mol)):
                    if atom_i != atom_j:
                        g2.append(ACSF.G2(mol[atom_i][1:],mol[atom_j][1:],params["rmin"],params["rmax"],eta,r_s))
            G2.append(sum(g2)) 
        #print(len(G2_data))
        G4=[]
        for labd in params["labd"]:
            for zeta in params["zeta"]:
                for eta in params["etaG4"]:
                    g4=[]
                    mol_index =[s for s in range(len(mol))]
                    mol_index.remove(atom_i)
                    Tjk = list(itertools.combinations(mol_index,2))
                    for atom_j,atom_k in Tjk:
                        g4.append(ACSF.G4(mol[atom_i][1:],mol[atom_j][1:],mol[atom_k][1:],zeta,labd,eta,params["rmin"],params["rmax"]))
                    G4.append(sum(g4))
        symmmol.append(G2+G4)
    return symmmol

path = glob.glob("systems/group*/")
Tot=[]
Tot1=[]
for i in path:
    atom=np.load(i+"/atom.npy")
    energy=np.load(i+"/energy.npy")
    for test in atom:
        mol1=[]
        D=test[:,1:]*0.5291772
        mol2 = test.tolist()
        for s in mol2:
            s=[str(i) for i in s]
            mol1.append(s)
        symm=acsf(mol1)
        print(symm)
        Tot.append(symm)
    Tot1.append(energy)
Tot1=np.concatenate((Tot1[0],Tot1[1],Tot1[2],Tot1[3]))
np.save("symm.npy", np.asarray(Tot))
print(np.asarray(Tot).shape)
np.save("energy.npy",np.array(Tot1))
#print(np.asarray(Tot)[0])
