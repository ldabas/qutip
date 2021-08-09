#%%
from copy import deepcopy
import enum
from re import T
from numpy.core.function_base import linspace
from numpy.testing._private.utils import measure
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
pi_time_list=[0.9616075295292871,
 0.6799131117266513,
 0.5551799620639973,
 0.4808137005549264,
 0.4301277653668094,
 0.3928772171874855,
 0.3641189085741248,
 0.3409357557557456]
#set up the processor and compiler,qb5d97 is the qubit we play around
qb_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=[0.26],\
    rest_place=qubit_phonon_detuning,FSR=13)

qb_compiler = hbar_compiler.HBAR_Compiler(qb_processor.num_qubits,\
    qb_processor.params, qb_processor.pulse_dict)

qb_simulation=hbar_simulation_class.Simulation(qb_processor,qb_compiler)
qb_simulation.swap_time_list=pi_time_list

# %%
'''
For higher order phonon rabi
'''
ydata_2D_list=[]
pi_time_list=[]
for n in range(9):
    qb_simulation.t_list=np.linspace(0.01,10,150)
    qb_simulation.ideal_phonon_fock(n)
    qb_simulation.phonon_rabi_measurement()  
    pi_time_list.append(qb_simulation.fit_result[-1]['swap_time'])
    qb_simulation.swap_time_list=pi_time_list
    ydata_2D_list.append(qb_simulation.y_array)
pi_time_list.pop(0)
# %%
ydata_2D_list=np.array(ydata_2D_list)
np.save('simulated_data//ideal_fock_state_phonon_rabi.npy',ydata_2D_list)
#%%
ydata_2D_list=np.load('simulated_data//ideal_fock_state_phonon_rabi.npy')
# %%
qb_simulation.t_list=np.linspace(0.01,10,150)
qb_simulation.generate_fock_state(3)
qb_simulation.plot_phonon_wigner()
qb_simulation.phonon_rabi_measurement()  
Measurement_data=qb_simulation.y_array
# %%

# %%
#check page for least square method with matrix 
#(English)  https://en.wikipedia.org/wiki/Ordinary_least_squares 
#(Chinese)  https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95
fitted_population=np.linalg.pinv(ydata_2D_list.transpose()).dot( Measurement_data.transpose())
plt.plot(fitted_population,drawstyle= 'steps-mid')

# %%
figure, ax= plt.subplots(figsize=(8,6))
fitted=fitted_population.dot(ydata_2D_list)
measured=qb_simulation.y_array
t=qb_simulation.x_array
ax.plot(t,fitted,label='fitted')
ax.plot(t,measured,label='simulated')
plt.legend()
figure.show()

# %%
qt.plot_wigner(qt.fock(10,2),alpha_max=1.6,cmap='seismic')

#%%
figure, ax= plt.subplots(figsize=(8,6))
fitted=fitted_population.dot(ydata_2D_list)
measured=qb_simulation.y_array
t=qb_simulation.x_array
for i, y in enumerate(ydata_2D_list):
    ax.plot(t,y,label='ideal fock{}'.format(i))

plt.legend()
figure.show()

# %%
