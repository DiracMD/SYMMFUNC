"""
filename: acsf.py
author:   Jxiao
version:  0.1
"""
import numpy as np 

def cutoff_fc(dist, rmin, rmax):
    if dist <= abs(rmax-rmin):
        fc = 0.5 * (np.cos(np.pi * dist / (rmax-rmin)) + 1)
    elif dist > abs(rmax-rmin):
        fc = 0
    return fc

def cutoff_fc2(dist, rmin, rmax):
    if dist <= abs(rmax-rmin):
        fc = pow(np.tanh(1-dist/(rmax-rmin)),3)
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
    costheta = 0.5/(dist_ij*dist_ik)*(dist_ij**2+dist_ik**2-dist_jk**2)
    gauss=np.exp(-eta * (dist_ij**2+dist_ik**2+dist_jk**2)) * fc_ij * fc_ik * fc_jk
    return 2 * pow(0.5*(1 + labd * costheta),xi)*gauss

def G5(atom_i, atom_j, atom_k, xi, labd, eta, rmin, rmax):
    i=np.array([float(i) for i in atom_i])
    j=np.array([float(j) for j in atom_j])
    k=np.array([float(k) for k in atom_k])
    dist_ij = np.linalg.norm(i-j)
    dist_ik = np.linalg.norm(i-k)
    dist_jk = np.linalg.norm(j-k)
    fc_ij = cutoff_fc(dist_ij, rmin, rmax)
    fc_ik = cutoff_fc(dist_ik, rmin, rmax)
    costheta = 0.4/(dist_ij*dist_ik)*(dist_ij**2+dist_ik**2-dist_jk**2)
    gauss=np.exp(-eta * (dist_ij**2+dist_ik**2+dist_jk**2)) * fc_ij * fc_ik
    return 2 * pow(0.5*(1 + labd * costheta),xi)*gauss
