# %%
from copy import deepcopy
from qutip.states import fock
from qutip.solver import Result
from qutip.operators import create, destroy
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
%matplotlib qt
#%%

psi0=qt.fock(2,0)#initial qubit state, at ground
# %%
H1=qt.destroy(2)+qt.create(2)#qubit half pi pulse
psi1=(-1j*H1*np.pi/4).expm()*psi0#qubit state after first half pi pulse

# %%
phonon_dim=10
psi2=qt.tensor(psi1,qt.fock(10,0))#tensor the qubit and phonon state
H2=qt.tensor(qt.destroy(2),qt.create(phonon_dim))+qt.tensor(qt.create(2),qt.destroy(phonon_dim))#swap hamitonian
psi3=(-1j*H2*np.pi/2).expm()*psi2#evolution of the swap operation
#%%
psi4=psi3.ptrace(1)#for the later part, only leave the phonon
def H_displacement(alpha):
    return alpha*qt.create(phonon_dim)+np.conj(alpha)*qt.destroy(phonon_dim)
def parity(density_matrix):
    diag=density_matrix.diag()
    P=0
    for i,n in enumerate(diag):
        P=P+n*(-1)**i
    return P
#%%
zz=[]
for imag in np.linspace(-2,2,20):
    z=[]
    for real in np.linspace(-2,2,20):
        H_phonon=H_displacement(real+1j*imag)
        z.append(parity((-1j*H_phonon).expm()*psi4))
    zz.append(z)

# %%
zz=np.array(zz)
zz=np.real(zz)

def axis_for_mesh(axis):
    begin=axis[0]
    end=axis[-1]
    length=len(axis)
    step=axis[1]-axis[0]
    begin=begin-step/2
    end=end+step/2
    length=length+1
    return np.linspace(begin,end,length)
axis=np.linspace(-2,2,20)

xx,yy=np.meshgrid(axis_for_mesh(axis),axis_for_mesh(axis))

fig, ax1, = plt.subplots(1, 1, figsize=(6,6))
im = ax1.pcolormesh(yy,xx, zz, cmap='seismic',vmin=-1, vmax=1)
fig.colorbar(im)
fig.gca().set_aspect('equal', adjustable='box')
fig.show()

# %%
qt.plot_wigner(psi4,cmap='seismic')
# %%
