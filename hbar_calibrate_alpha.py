# %%
from copy import deepcopy
import enum
from re import T
from numpy.core.function_base import linspace
import scipy
from scipy.ndimage.measurements import label
from scipy.sparse import data
from qutip.visualization import plot_fock_distribution
from qutip.states import coherent, fock
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
%matplotlib qt
#%%
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_simulation_class)
reload(hbar_fitting)

#qubit dimission, we only consider g and e states here
qubit_dim=2
#phonon dimission
phonon_dim=10
#how many phonon modes we consider here
phonon_num=1
#the frequency difference between qubit and phonon (qubit minus phonon)
qubit_freq=5970.04
phonon_freq=5974.115
interaction_1_freq=5972.2
interaction_3_freq=5972.95
qubit_phonon_detuning=qubit_freq-phonon_freq

#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]*phonon_num
#T1 list of the system 77
t1=[13]+[81]*(phonon_num)
#T2 list of the system 104
t2=[12.4]+[134]*(phonon_num)
#pi time list for different fock state
pi_time_list=[0.9616123677058709,
 0.679329038657111,
 0.5548147810734809,
 0.48027408123596266]
#set up the processor and compiler,qb5d97 is the qubit we play around
qb_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=[0.26],\
    rest_place=qubit_phonon_detuning,FSR=13)

qb_compiler = hbar_compiler.HBAR_Compiler(qb_processor.num_qubits,\
    qb_processor.params, qb_processor.pulse_dict)

qb_simulation=hbar_simulation_class.Simulation(qb_processor,qb_compiler)
qb_simulation.swap_time_list=pi_time_list

# %%
#calibrate probe amplitude for qubit spec
param_probe={'Omega':0.015,
    'sigma': 0.5,
    'duration':15,
    'amplitude_starkshift':0}

y_list=[]
sweep_list=np.linspace(0.005,0.04,10)
for sweep_data in sweep_list:
    param_probe['Omega']=sweep_data
    param_probe['duration']=15
    qb_simulation.ideal_phonon_fock(0)
    param_probe['amplitude_starkshift']=interaction_1_freq-qubit_freq
    qb_simulation.detuning_list=np.linspace(
        param_probe['amplitude_starkshift']-0.2,
        param_probe['amplitude_starkshift']+0.2,41)
    qb_simulation.spec_measurement(param_probe)
    y_list.append(qb_simulation.y_array)

figure,ax = plt.subplots(figsize=(8,6))
for i,sweep_data in enumerate(sweep_list):
    ax.plot(qb_simulation.detuning_list ,y_list[i],label='probe omega={}MHz'.format(sweep_data))
figure.legend()
figure.show()

# %%
# calibrating the relation between alpha from density matrix and number splitting.
param_probe={'Omega':0.017,
    'sigma': 0.5,
    'duration':13,
    'amplitude_starkshift':0}
param_drive={'Omega':0.2,
    'sigma':0.5,
    'duration':10,
    'rotate_direction':0,
    'detuning':-qubit_phonon_detuning
    }
density_matrix_alpha_list=[]
fit_alpha_list=[]
peak_position=[]
for i in range(10):
    peak_position.append(2.849-0.096*i)

for n,drive_amplitude in enumerate(np.linspace(0.1,0.6,6)):
    param_drive['Omega']=drive_amplitude
    qb_simulation.generate_coherent_state(param_drive)
    qb_simulation.fit_wigner()
    density_matrix_alpha_list.append(qb_simulation.alpha)

    param_probe['amplitude_starkshift']=interaction_3_freq-qubit_freq
    qb_simulation.detuning_list=np.linspace(
        param_probe['amplitude_starkshift']-1,
        param_probe['amplitude_starkshift']+0.2,201)
    qb_simulation.spec_measurement(param_probe)
    result=qb_simulation.fitter.sum_lorentz_fit(peak_position,2+int(n*1.2))
    fit_alpha_list.append(result[-1])

# %%
density_matrix_alpha_list=np.array(density_matrix_alpha_list)
density_matrix_alpha_list=np.abs(density_matrix_alpha_list)
plt.plot(density_matrix_alpha_list,fit_alpha_list,label='data')
plt.xlabel('alpha from density matrix')
linear_result=np.polyfit(density_matrix_alpha_list,fit_alpha_list,1)
plt.ylabel('alpha from number splitting')
plt.plot(density_matrix_alpha_list,linear_result[0]*density_matrix_alpha_list+linear_result[1],label='fit')
plt.legend()

# %%
#load simulated fock state data
simulate_data_list=[]
for i in [0,1,2,0.5]:
    data_fock=np.load('simulated_data//fock_{}_wigner.npy'.format(i))
    simulate_data_list.append(data_fock)
data_axis=np.load('simulated_data//axis.npy')

#load measured data
measured_data_list=[]
for i in [0,1,2,0.5]:
    data_fock=np.load('wigner_data//fock_{}_measured_data.npy'.format(i))
    measured_data_list.append(data_fock)
data_axis=np.linspace(-32*0.06,32*0.06,40)

# %%
fock_number=0
def fock_fit(x):
    ratio,normalize=x
    data_type='measurement'
    data_ideal=normalize*qt.wigner(qt.fock(10,fock_number),ratio*data_axis,ratio*data_axis)
    if data_type=='simulation':#simulation data
        data_be_fitted=simulate_data_list[fock_number]
    if data_type=='measurement':
        data_be_fitted=measured_data_list[fock_number]/4
    residual=np.sum(np.square(data_be_fitted-data_ideal))
    return residual
minimize_result=scipy.optimize.minimize(fock_fit,(1,1),bounds=[(0.8,2),(0.3,3)])
# %%
factor_ratio,normalize=minimize_result['x']
xx,yy=np.meshgrid(data_axis,data_axis)
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(6,6))
data_ideal=normalize * qt.wigner(qt.fock(10,fock_number),factor_ratio*data_axis,factor_ratio*data_axis)
im1 = ax1.pcolormesh(yy,xx, data_ideal, cmap='seismic',vmin=-1, vmax=1)
im2 = ax2.pcolormesh(yy,xx, measured_data_list[fock_number]/4, cmap='seismic',vmin=-1, vmax=1)
ax1.axis('equal')
ax2.axis('equal')
fig.legend()
fig.show()
# %%
factor_ratio
# %%
qt.plot_wigner(fock(5,0),alpha_max=2.1,cmap='seismic')
# %%
