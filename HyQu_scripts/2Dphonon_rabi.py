# %%
import enum
from qutip.qobj import ptrace
from re import T
from numpy.core.function_base import linspace
from numpy.matrixlib.defmatrix import matrix
from qutip.visualization import plot_fock_distribution
from qutip.states import coherent
import hbar_compiler
import hbar_processor
import hbar_simulation_class
import hbar_fitting
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from qutip.qip.circuit import Measurement, QubitCircuit
import qutip as qt
from qutip import basis
from matplotlib.colors import ListedColormap
from copy import deepcopy
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_simulation_class)
reload(hbar_fitting)
%matplotlib qt
# %%
#qubit dimission, we only consider g and e states here
qubit_dim=2
#phonon dimission
phonon_dim=3
#how many phonon modes we consider here
phonon_num=2
#the frequency difference between qubit and phonon (qubit minus phonon)
qubit_freq=5970.04
phonon_freq=5974.11577
interaction_1_freq=5972.2
interaction_3_freq=5972.95
qubit_phonon_detuning=qubit_freq-phonon_freq

#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]*phonon_num
#T1 T2 is the average number of the two interaction point
t1=[(13.1+9.7)/2]+[81]*(phonon_num)
t2=[(9.8+10.1)/2]+[134]*(phonon_num)
#pi time list for different fock state
pi_time_list=[0.9616123677058709,
 0.679329038657111,
 0.5548147810734809,
 0.48027408123596266]
pi_time_list=[0.899+0.08,0.623+0.08,0.525+0.08]
#set up the processor and compiler,qb5d97 is the qubit we play around
#Omega=50 because we use width of pi pulse as 20ns
qb_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims, Omega=50/(2*np.pi),g=[0.26,0.099],\
    rest_place=qubit_phonon_detuning,FSR=1.1)

qb_compiler = hbar_compiler.HBAR_Compiler(qb_processor.num_qubits,\
    qb_processor.params, qb_processor.pulse_dict)

qb_simulation=hbar_simulation_class.Simulation(qb_processor,qb_compiler)
qb_simulation.swap_time_list=pi_time_list
# %%
def ptrace_list(density_matrix_list,i):
    population_list=[]
    for matrix in density_matrix_list:
        population_list.append(
            qt.expect(
                qt.num(
                    matrix.dims[0][i]
                    ),
                    matrix.ptrace(i)
                    )
        )
    return population_list

qb_simulation.t_list=np.linspace(0.01,20,150)
zz_data_qubit=[]
zz_data_phonon1=[]
zz_data_phonon2=[]
for detuning in np.linspace(-1,2,150):
    qb_simulation.generate_fock_state(0)
    qb_simulation.phonon_rabi_measurement(detuning=detuning,if_fit=0)
    zz_data_qubit.append(ptrace_list(qb_simulation.final_state_list,0))
    zz_data_phonon1.append(ptrace_list(qb_simulation.final_state_list,1))
    zz_data_phonon2.append(ptrace_list(qb_simulation.final_state_list,2))
# %%
zz_data=np.array(zz_data_phonon2).transpose()
#%%
zz_data=np.load('simulated_data//2D_phonon_rabi_2.npy')
x_axis=np.linspace(-1,2,200)
y_axis=np.linspace(0.01,20,150)
# %%
def axis_for_mesh(axis):
    begin=axis[0]
    end=axis[-1]
    length=len(axis)
    step=axis[1]-axis[0]
    begin=begin-step/2
    end=end+step/2
    length=length+1
    return np.linspace(begin,end,length)
x_axis=np.linspace(-1,2,150)
y_axis=qb_simulation.t_list
xx,yy=np.meshgrid(axis_for_mesh(x_axis),axis_for_mesh(y_axis))
fig, ax1, = plt.subplots(1, 1, figsize=(10,6))
cmap_teals = ListedColormap(np.loadtxt('cmap_teals.txt')/255)
im = ax1.pcolormesh(xx,yy, zz_data, cmap=cmap_teals,vmin=0, vmax=1)
fig.colorbar(im)
fig.legend()
fig.show()

# %%
np.save('simulated_data//2D_phonon_rabi_3_phonon2.npy',zz_data)























# %%
