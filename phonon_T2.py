# %% For this page, I will show the result of how qubit T2 affect measured phonon T2.
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
phonon_num=1
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
#%%
qb_simulation.t_list=np.linspace(0.1,200,100)
# %%
qb_simulation.phonon_ramsey_measurement(artificial_detuning=-qubit_phonon_detuning+0.05)


# %%
fitted_T2_list=[]
y_array_list=[]
for qubit_t2 in np.linspace(2,15,10):
    t2=[qubit_t2]+[134]*(phonon_num)
    #pi time list for different fock state
    qb_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims, Omega=50/(2*np.pi),g=[0.26,0.099],\
        rest_place=qubit_phonon_detuning,FSR=1.1)

    qb_compiler = hbar_compiler.HBAR_Compiler(qb_processor.num_qubits,\
        qb_processor.params, qb_processor.pulse_dict)

    qb_simulation=hbar_simulation_class.Simulation(qb_processor,qb_compiler)
    qb_simulation.swap_time_list=pi_time_list
    qb_simulation.t_list=np.linspace(0.1,200,100)
    qb_simulation.phonon_ramsey_measurement(artificial_detuning=-qubit_phonon_detuning+0.05)
    y_array_list.append(qb_simulation.y_array)
    fitted_T2_list.append(qb_simulation.fit_result)
# %%
fitted_T2_list
# %%
t2_list=[]
for result in fitted_T2_list:
    t2_list.append(result[0]['T2'])
fig, ax=plt.subplots(figsize=(8,6))
ax.plot(np.linspace(2,15,10),t2_list)
ax.set_xlabel('qubit T2(us)')
ax.set_ylabel('phonon Ramsey(us)')
fig.show()
# %%
(13.1+9.7)/2
# %%
# %%
#qubit dimission, we only consider g and e states here
qubit_dim=2
#phonon dimission
phonon_dim=3
#how many phonon modes we consider here
phonon_num=1
#the frequency difference between qubit and phonon (qubit minus phonon)
qubit_freq=5972.95
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
#%%
qb_simulation.t_list=np.linspace(0.1,200,100)
qb_simulation.phonon_T1_measurement()
# %%
