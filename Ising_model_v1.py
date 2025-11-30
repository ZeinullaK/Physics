#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
from __future__ import division
from scipy.ndimage import convolve, generate_binary_structure
import torch
from torch.nn.functional import conv2d, conv3d
import scipy.constants as const
#%%
device='cuda'
#%%
N = 50
lattice = torch.zeros((N, N)).to(device)
random_lattice = torch.rand((N, N)).to(device)
lattice[random_lattice>=0.5] = 1
lattice[random_lattice<0.5] = -1

lattice_copy = lattice.clone()
plt.imshow(lattice_copy.to('cpu'), cmap='gray')

lattice = lattice.unsqueeze(0).unsqueeze(0)
#%%
def get_energy_tensor(lattice):
    # applies the nearest neighbours summation
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    kern = torch.tensor(kern.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    E_arr = -lattice * conv2d(lattice, kern, padding='same')
    return E_arr
#%%
def get_energy(lattice):
    return get_energy_tensor(lattice).sum()
#%%
def get_dE_tensor(lattice):
    return -2*get_energy_tensor(lattice)
# %%
def metropolis(lattice, steps, BJs):
    energies = []
    magnetization = []
    avg_magnetization = []
    lattice_batch = torch.clone(lattice)

    for _ in range(steps):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        dE = get_dE_tensor(lattice_batch)[:, :, i::2, j::2]
        change = (dE>=0)*(torch.rand(dE.shape).to(device) < torch.exp(-BJs*dE)) + (dE<0)

        lattice_batch[:, :, i::2, j::2][change] *= -1

        energies.append(get_energy(lattice_batch).item())
        magnetization.append(lattice_batch.sum(axis=(2,3)).item())
        avg_magnetization.append((lattice_batch.sum(axis=(2,3))/N).item())   

    return energies, magnetization, avg_magnetization, lattice_batch
# %%
energies, magnetization, avg_magnetization, lattice_end = metropolis(lattice, 4000, 0.5)
lattice_copy = lattice_end.clone()
lattice_copy = lattice_copy.squeeze()
plt.imshow(lattice_copy.to('cpu'), cmap='gray')
# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
ax = axes[0]
ax.plot(energies)
ax.set_xlabel('Monte Carlo Time Steps')
ax.set_ylabel(r'Energy')
ax = axes[1]
ax.plot(magnetization)
ax.set_xlabel('Monte Carlo Time Steps')
ax.set_ylabel(r'Total magnetization')
# %%
# 500 steps is enough for equilibrium
def temp_scan(lattice, steps, BJs):
    Es =[]
    Ms = []
    Cs = []
    k = const.k
    for _ in range(len(BJs)): 
        energies, magnetization, avg_magnetization, lattice_end = metropolis(lattice, steps, BJs[_])
        energies = torch.tensor(energies)
        magnetization = abs(torch.tensor((magnetization)))
        avg_E = torch.mean(energies[500:-1])
        avg_M = torch.mean(magnetization[500:-1])
        C = (torch.mean(energies[500:-1]**2).item() - avg_E**2) * (BJs[_]**2 * k)
        Es.append(avg_E)
        Ms.append(avg_M)
        Cs.append(C)
    
    return Es, Ms, Cs
#%%
N = 20
lattice = torch.zeros((N, N)).to(device)
random_lattice = torch.rand((N, N)).to(device)
lattice[random_lattice>=0.5] = 1
lattice[random_lattice<0.5] = -1

lattice_copy = lattice.clone()
plt.imshow(lattice_copy.to('cpu'), cmap='gray')

lattice = lattice.unsqueeze(0).unsqueeze(0)
#%%
BJs = torch.arange(0.1, 1.0, 0.05)
Es, Ms, Cs = temp_scan(lattice, 5000, BJs)
# %%
fig, axes = plt.subplots(1, 3, figsize = (20, 7))
ax = axes[0]
ax.plot(BJs, Es)
ax.set_xlabel(r'Temperature $\beta J$')
ax.set_ylabel('Average energy')
ax = axes[1]
ax.plot(BJs, Ms)
ax.set_xlabel(r'Temperature $\beta J$')
ax.set_ylabel('Average magnetization')
ax = axes[2]
ax.plot(BJs, Cs)
ax.set_xlabel(r'Temperature $\beta J$')
ax.set_ylabel('Heat capacity')
# %%
# 3D Ising model
N = 100
random_lattice = torch.rand((N,N,N)).to(device)
lattice = torch.zeros((N,N,N)).to(device)
lattice[random_lattice>=0.5] = 1
lattice[random_lattice<0.5] = -1

lattice = lattice.unsqueeze(0).unsqueeze(0)
#%%
def get_energy_tensor_3D(lattice):
    kern = generate_binary_structure(3, 1) 
    kern[1][1][1] = False
    kern = torch.tensor(kern.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    E_arr = -lattice * conv3d(lattice, kern, padding='same')
    return E_arr
#%%
def get_energy_3D(lattice):
    return get_energy_tensor_3D(lattice).sum(axis=(2, 3, 4)).squeeze(0, 1)
# %%
def get_dE_tensor_3D(lattice):
    return -2*get_energy_tensor_3D(lattice)
# %%
def metropolis_3D(lattice, steps, BJ):
    energies = []
    magnetization = []
    lattice_batch = torch.clone(lattice)
    for t in range(steps):
        i = np.random.randint(0,2)
        j = np.random.randint(0,2)
        k = np.random.randint(0,2)
        dE = get_dE_tensor_3D(lattice_batch)[:,:,i::2,j::2,k::2]
        change = (dE>=0)*(torch.rand(dE.shape).to(device) < torch.exp(-BJ*dE)) + (dE<0)
        lattice_batch[:,:,i::2,j::2,k::2][change] *=-1
        energies.append(get_energy_3D(lattice_batch))
        magnetization.append(lattice_batch.sum(axis=(1, 2, 3, 4)))
    return energies, magnetization, lattice_batch
#%%
energies, magnetization, lattice_end = metropolis_3D(lattice, 1000, 0.5)
energies = torch.tensor(energies)
magnetization = torch.tensor(magnetization)
#%%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
ax = axes[0]
ax.plot(energies)
ax.set_xlabel('Monte Carlo Time Steps')
ax.set_ylabel(r'Energy')
ax = axes[1]
ax.plot(magnetization)
ax.set_xlabel('Monte Carlo Time Steps')
ax.set_ylabel(r'Total magnetization')
# %%
from mpl_toolkits.mplot3d import Axes3D
# %%
lattice_copy = lattice_end.squeeze().to('cpu')
N = lattice_copy.shape[0]

X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')

up_spins = lattice_copy == 1
down_spins = lattice_copy == -1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[up_spins], Y[up_spins], Z[up_spins], c='red', marker='o', s=50, alpha=0.01, label='Spin +1')
ax.scatter(X[down_spins], Y[down_spins], Z[down_spins], c='blue', marker='o', s=50, alpha=0.01, label='Spin -1')

ax.set_xlabel('i')
ax.set_ylabel('j')
ax.set_zlabel('k')
ax.legend()
plt.show()
# %%