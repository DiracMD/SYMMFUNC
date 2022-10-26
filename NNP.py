import numpy as np 
import math as pi


# cut off function
"""
def cutoff_fc(dist, rmin, rmax):
    mask = dist >rmin
    fc = 0.5 * (np.cos(pi * (dist - rmin) / (rmax - rmin)) + 1)
    fc[dist >= rmax] = 0
    return fc
"""

def cutoff_fc(dist, rmin, rmax):
    if dist <= abs(rmax-rmin):
        fc = 0.5 * (np.cos(np.pi * dist / (rmax-rmin)) + 1)
    elif dist > abs(rmax-rmin):
        fc = 0
    return fc

# radial function
def G1(atom_i, atom_j, rmin, rmax):
    i=np.array([float(i) for i in atom_i])
    j=np.array([float(j) for j in atom_j])
    dist = np.linalg.norm(i-j)
    return cutoff_fc(dist, rmin, rmax)

    
def G2(atom_i, atom_j, rmin, rmax, eta, Rs):
    i=np.array([float(i) for i in atom_i])
    j=np.array([float(j) for j in atom_j])
    dist = np.linalg.norm(i-j)
    fc = cutoff_fc(dist,rmin,rmax)
    return np.exp(-eta * (dist - Rs)**2) * fc

def G3(atom_i, atom_j, rmin, rmax, kappa):
    i=np.array([float(i) for i in atom_i])
    j=np.array([float(j) for j in atom_j])
    dist = np.linalg.norm(i-j)
    dist = np.asarray(dist)
    fc = cutoff_fc(dist, rmin, rmax)
    return np.exp(-kappa * dist) * fc

# angular function
def G4(atom_i, atom_j, atom_k, xi, labd, eta, rmin, rmax):
    i=np.array([float(i) for i in atom_i])
    j=np.array([float(j) for j in atom_j])
    k=np.array([float(k) for k in atom_k])
    dist_ij = np.linalg.norm(i-j)
    dist_ik = np.linalg.norm(i-k)
    dist_jk = np.linalg.norm(j-k)
    fc_ij = cutoff_fc(dist_ij, rmin, rmax)
    fc_ik = cutoff_fc(dist_ik, rmin, rmax)
    fc_jk = cutoff_fc(dist_jk, rmin, rmax)
    angle_ijk = np.arccos(np.dot(i-j, i-k) / (dist_ij * dist_ik))
    return 2**(1-xi) * (1 + labd * np.cos(angle_ijk))**(xi)*np.exp(-eta * (dist_ij**2+dist_ik**2+dist_jk**2)) * fc_ij * fc_ik * fc_jk

def G5(atom_i, atom_j, atom_k, xi, labd, eta, rmin, rmax):
    i=np.array([float(i) for i in atom_i])
    j=np.array([float(j) for j in atom_j])
    k=np.array([float(k) for k in atom_k])
    dist_ij = np.linalg.norm(i-j)
    dist_ik = np.linalg.norm(i-k)
    dist_jk = np.linalg.norm(j-k)
    fc_ij = cutoff_fc(dist_ij, rmin, rmax)
    fc_ik = cutoff_fc(dist_ik, rmin, rmax)
    fc_jk = cutoff_fc(dist_jk, rmin, rmax)
    angle_ijk = np.arccos(np.dot(i-j, i-k) / (dist_ij * dist_ik))
    return (2-xi) * (1 + labd * np.cos(angle_ijk))**(xi) * np.exp(-eta * (dist_ij**2+dist_ik**2)) * fc_ij * fc_ik
