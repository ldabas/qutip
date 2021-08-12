# %%
from copy import deepcopy
import enum
from re import T
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
#%%
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_simulation_class)
reload(hbar_fitting)

#qubit dimission, we only consider g and e states here
qubit_dim=2
#phonon dimission
phonon_dim=17
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
#Omega=50 because we use width of pi pulse as 20ns
qb_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims, Omega=50/(2*np.pi),g=[0.26],\
    rest_place=qubit_phonon_detuning,FSR=13)

qb_compiler = hbar_compiler.HBAR_Compiler(qb_processor.num_qubits,\
    qb_processor.params, qb_processor.pulse_dict)

qb_simulation=hbar_simulation_class.Simulation(qb_processor,qb_compiler)
qb_simulation.swap_time_list=pi_time_list



# %%
'''
qubit T1 measurement to test code
'''
qb_simulation.qubit_T1_measurement()



# %%
'''
For higher order phonon rabi
'''
pi_time_list=[]
for n in range(4):
    qb_simulation.t_list=np.linspace(0.01,10,100)
    qb_simulation.generate_fock_state(n)
    qb_simulation.phonon_rabi_measurement()  
    pi_time_list.append(qb_simulation.fit_result[-1]['swap_time'])
    qb_simulation.swap_time_list=pi_time_list



# %%
#first,let's calibrate the artificial detuning
qb_simulation.t_list=np.linspace(0.01,10,101)
qb_simulation.ideal_phonon_fock(0)
qb_simulation.qubit_ramsey_measurement(artificial_detuning=interaction_1_freq-qubit_freq+0.5,
starkshift_amp=interaction_1_freq-phonon_freq,if_fit=True)
extra_detuning=qb_simulation.fit_result[-1]['delta']-0.5




#%%
#test parity measurement,with generated phonon fock
y_2d_list=[]
for i in range(5):
    qb_simulation.generate_fock_state(i)
    qb_simulation.qubit_ramsey_measurement(artificial_detuning=interaction_1_freq-qubit_freq-extra_detuning,
    starkshift_amp=interaction_1_freq-phonon_freq,if_fit=False)
    y_2d_list.append(qb_simulation.y_array)
figure, ax = plt.subplots(figsize=(8,6))
for i in range(5):
    ax.plot(qb_simulation.t_list,y_2d_list[i],label='Fock{}'.format(i))
plt.legend()




# %%
#plot the sum parity,find the parity measurement time
sum=np.zeros(len(qb_simulation.t_list))
figure, ax = plt.subplots(figsize=(8,6))
for i,y in enumerate(y_2d_list):
    sum=sum+(-1)**(i)*(y-0.5)*2
ax.plot(qb_simulation.t_list,sum)
t_parity=qb_simulation.t_list[np.argsort(sum)[-1]]
ax.plot([t_parity,t_parity],[np.min(sum),np.max(sum)])
ax.set_title('best parity measurement time is {} us'.format(t_parity))
plt.show()


# %%
'''
calculate the driving amplitude in the real measurement
drive amplitude in the experiment is 0.06
for qubit operation, pi_amp=0.64, width of the pi pulse is 20us
'''
Omega_drive=50/(2*np.pi)/2.49986*np.pi/2/0.64*0.06
#%%
#probe phonon, because qubit and phonon dispersive coupling, phonon frequency changed
param_probe={'Omega':0.1,
    'sigma': 0.5,
    'duration':5,
    'amplitude_starkshift':0}
qb_simulation.detuning_list=np.linspace(-qubit_phonon_detuning-0.05,-qubit_phonon_detuning+0.1,20)
qb_simulation.spec_measurement(param_probe,readout_type='read phonon')
#the phonon frequency move to 4.0915MHz.

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

#%% phase calibration
qb_simulation.wigner_measurement_1D(param_drive,starkshift_param,steps=30,
phase_calibration=True,if_echo=True)

#%%
#wigner 1D
calibration_phase=0.131
qb_simulation.calibration_phase=calibration_phase
qb_simulation.generate_fock_state(2)
qb_simulation.wigner_measurement_1D(param_drive,starkshift_param,steps=40,
phase_calibration=False,if_echo=True,first_pulse_phases=[0,np.pi/2,np.pi,np.pi/2*3])
np.save('simulated_data//cut_fock1.npy',qb_simulation.y_array)
#%%
x_simulated=qb_simulation.x_array
y_simulated=qb_simulation.y_array*2-1
x_ideal=np.linspace(-2,2,40)
y_ideal=qt.wigner(qt.fock(10,2),x_ideal,[0])[0]*np.pi/2
x_measurement=np.linspace(-0.06,0.06,40)*32.1/0.9
y_measurement=np.load('wigner_data//fock_{}_measured_data.npy'.format(2))[20]/4
#%%
fig, ax=plt.subplots(figsize=(8,6))
ax.plot(x_ideal,y_ideal,label='ideal')
ax.plot(x_simulated,y_simulated,label='simulated')
ax.plot(x_measurement,y_measurement,label='measurement')
plt.legend()
fig.show()
# %%
#calibrate the relation between phase and time
phase_list=[]
time_list=[7,7.05,7.1]
for time in time_list:
    starkshift_param={'detuning':interaction_1_freq-phonon_freq,
                    'duration':time}
    qb_simulation.wigner_measurement_1D(param_drive,starkshift_param,steps=30,
    phase_calibration=True,if_echo=True)
    phase_list.append(qb_simulation.fit_result[-1]['phi'])
phase_fit=np.polyfit(time_list,phase_list,1)
duration_list=np.linspace(7,7.2,30)
calibration_phase_list=phase_fit[0]*duration_list+phase_fit[1]
# %%
#calibrate the wigner background with time
qb_simulation.generate_fock_state(1)
qb_simulation.wigner_measurement_time_calibrate(param_drive,duration_list,interaction_1_freq-phonon_freq,
calibration_phases=calibration_phase_list,if_echo=True,first_pulse_phases=[0,np.pi/2,np.pi,np.pi/2*3])
# %%
#plot 2D wigner
wigner_data_list=[]
for fock_number in [1]:
    print(param_drive)
    starkshift_param={'detuning':interaction_1_freq-phonon_freq,
                'duration':7.05}
    calibration_phase=0.7378
    qb_simulation.calibration_phase=calibration_phase
    qb_simulation.generate_fock_state(fock_number)
    wigner_data=qb_simulation.wigner_measurement_2D(param_drive,starkshift_param,steps=40,
    if_echo=True,first_pulse_phases=[0,np.pi/2,np.pi,np.pi/2*3])
    wigner_data_list.append(qb_simulation.y_array)
    np.save('simulated_data//fock_{}_wigner_v3.npy'.format(fock_number),wigner_data)


# %%
