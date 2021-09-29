#%%
#this file is for calculating the background of wigner tomography by Ramsey sequence
from copy import deepcopy
import enum
from re import T
from numpy.core.defchararray import mod
from numpy.core.function_base import linspace
from scipy.ndimage.measurements import label
from scipy.sparse import data
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
%matplotlib qt
# %%
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_simulation_class)
reload(hbar_fitting)

#qubit dimission, we only consider g and e states here
qubit_dim=2
#phonon dimission
phonon_dim=15
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
qb_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims, Omega=50/(2*np.pi),g=[0.26],\
    rest_place=qubit_phonon_detuning,FSR=13)

qb_compiler = hbar_compiler.HBAR_Compiler(qb_processor.num_qubits,\
    qb_processor.params, qb_processor.pulse_dict)

qb_simulation=hbar_simulation_class.Simulation(qb_processor,qb_compiler)
qb_simulation.swap_time_list=pi_time_list
#%%
#try find a good driving amplitude
param_drive={'Omega':0.52,
    'sigma':0.5,
    'duration':12,
    'rotate_direction':np.pi,
    'detuning':-qubit_phonon_detuning
    }
qb_simulation.generate_coherent_state(param_drive)
qb_simulation.fit_wigner()
starkshift_param={'detuning':interaction_1_freq-phonon_freq,
                'duration':7}
abs(qb_simulation.alpha)
# %%
phase_list=[]
time_list=[7,7.01,7.02,7.03]
for time in time_list:
    qb_simulation.ideal_phonon_fock(0)
    starkshift_param={'detuning':interaction_1_freq-phonon_freq,
                    'duration':time}
    qb_simulation.wigner_measurement_1D(param_drive,starkshift_param,steps=30,
    phase_calibration=True,if_echo=False)
    phase_list.append(qb_simulation.fit_result[-1]['phi'])
phase_fit=np.polyfit(time_list,phase_list,1)
#%%
duration_list=np.linspace(7,8,30)
calibration_phase_list=phase_fit[0]*duration_list+phase_fit[1]
#%%
calibration_phase=-1.221
qb_simulation.calibration_phase=calibration_phase
qb_simulation.generate_fock_state(2)
starkshift_param={'detuning':interaction_1_freq-phonon_freq,
                    'duration':7}
qb_simulation.wigner_measurement_1D(param_drive,starkshift_param,steps=40,
phase_calibration=False,if_echo=False,first_pulse_phases=[0,np.pi/2,np.pi,np.pi/2*3])
# %%
qb_simulation.generate_fock_state(0) 
qb_simulation.wigner_measurement_time_calibrate(param_drive,duration_list,interaction_1_freq-phonon_freq,
calibration_phases=calibration_phase_list,if_echo=False,first_pulse_phases=[0,np.pi/2,np.pi,np.pi/2*3])
fig, ax=plt.subplots(figsize=(8,6))
ax.plot(qb_simulation.x_array,qb_simulation.y_array)
ax.plot([duration_list[0],duration_list[-1]],[0.5,0.5])
fig.show()

# %%
phase_fit=[ 13.38601886, -94.92311186]
# %%
