
#%%
from copy import deepcopy
from qutip.operators import create, destroy, num, qeye
import numpy as np
from qutip import sigmax, sigmay, sigmaz, tensor, fidelity
from qutip.qip.device.processor import Processor
from qutip.qip.compiler import Instruction
from qutip.qip.compiler import GateCompiler
import matplotlib.pyplot as plt
from os import error
import qutip as qt
from qutip.tensor import tensor
from qutip.states import coherent, fock 
from numpy.core.records import array
from qutip.expect import expect
import numpy as np
from qutip.solver import Options
from qutip import basis
from qutip.qip.circuit import Measurement, QubitCircuit
from qutip.solver import Options
from qutip.operators import num, phase, qeye
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
%matplotlib qt

#define the processor here
class demo_processor(Processor):
    def __init__(self,N,t1,t2,dims,Omega=20):
        super(demo_processor,self).__init__(N,t1,t2,dims)
        self.set_up_params(Omega)
        self.set_up_ops()
        self.set_up_drift() 
        
    def set_up_params(self, Omega):
        self.params = {}
        self.params["Omega"] = Omega  # default rabi frequency
       
    def set_up_ops(self):
        self.pulse_dict = {}  # A dictionary that maps the name of a pulse to its index in the pulse list.
        index = 0
        
        # X-axis rotate
        self.add_control(create(self.dims[0])+destroy(self.dims[0]), 0, label="X-axis_R") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["X-axis_R"] = index
        index += 1
        
        # Y-axis rotate
        self.add_control(1j*create(self.dims[0])-1j*destroy(self.dims[0]), 0, label="Y-axis_R") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["Y-axis_R"] = index
        index += 1
        
        # Z-axis rotate
        self.add_control(num(self.dims[0]), 0, label="Z-axis_R") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["Z-axis_R"] = index
        index += 1

        #wait
        self.add_control(0*qeye(self.dims[0]), 0, label="Wait_T") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["Wait_T"] = index
        index +=1

    def set_up_drift(self):
        pass
    def load_circuit(self, circuit, schedule_mode=False, compiler=None):
        tlist, coeffs = compiler.compile(circuit, schedule_mode=schedule_mode)
        # save the time sequence and amplitude for all pulses
        self.set_all_tlist(tlist)
        self.coeffs = coeffs
        return tlist, self.coeffs

#define the modulated gaussian shape
def gauss_dist(t, sigma, amplitude, duration,omega,phase):
    #gaussian shape pulse
    return amplitude*np.exp(-0.5*((t-duration/2)/sigma)**2)*np.cos(omega*t+phase)

#define gaussian shape x rotation, normally for quick pulse
def gauss_rx_compiler(gate, args):
    """
    Compiler for the X-axis_Rotate gate
    """
    targets = gate.targets  # target qubit
    if not gate.arg_value:
        gate.arg_value={}

    parameters = args["params"]
    # rabi frequency of the qubit operation
    Omega=gate.arg_value.get('Omega',parameters["Omega"])*np.pi*2
    # the phase want to rotate, correspond to amplitude and time of pulse
    rotate_phase=gate.arg_value.get('rotate_phase',np.pi)
    #the ratio we need to change the amplitude based on phase we want to rotate
    amp_ratio=rotate_phase/(np.pi)
    #the detuning of the frequency of the driving field
    detuning=gate.arg_value.get('detuning',0) *2*np.pi
    #rotate direction of the qubit, 0 means X, np.pi/2 means Y axis
    rotate_direction=gate.arg_value.get('rotate_direction',0)

    gate_sigma = 1/Omega
    amplitude = Omega/2.49986*np.pi/2*amp_ratio #  2.49986 is just used to compensate the finite pulse duration so that the total area is fixed
    duration = 4 * gate_sigma
    tlist = np.linspace(0, duration, 300)
    coeff1 = gauss_dist(tlist, gate_sigma, amplitude, duration,detuning,rotate_direction)
    coeff2 = gauss_dist(tlist, gate_sigma, amplitude, duration,detuning,rotate_direction+np.pi/2)
    pulse_info =[ ("X-axis_R", coeff1),("Y-axis_R", coeff2) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

#define waiting pulse
def wait_complier(gate,args):
    targets = gate.targets  # target qubit
    parameters = args["params"]
    time_step=3e-3 #assume T1 at least 1us for each part of the system
    duration=gate.arg_value
    tlist=np.linspace(0,duration,int(duration/time_step))
    coeff=0*tlist
    pulse_info =[ ("Wait_T", coeff) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

class demo_Compiler(GateCompiler):  # compiler class
    def __init__(self, num_qubits, params, pulse_dict):
        super(demo_Compiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict)
        # pass our compiler function as a compiler for RX (rotation around X) gate.
        self.gate_compiler["X_R"] = gauss_rx_compiler
        self.gate_compiler["Wait"]=wait_complier
        self.args.update({"params": params})

class Simulation():
    '''
    Setting the simulated experiment
    '''
    def __init__(self,processor,compiler,t_list=np.linspace(0.1,10,100),
        initial_state=None):
        self.qubit_probe_params={}
        self.phonon_drive_params={}
        self.processor=processor
        self.compiler=compiler
        self.t_list=t_list
        self.fit_result=[]
        if not initial_state:
            self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        else:
            self.initial_state=initial_state

    def run_circuit(self,circuit):
        #use copied processor to initialize
        self.copied_processor=deepcopy(self.processor)
        self.copied_processor.load_circuit(circuit, compiler=self.compiler)
        option=Options()
        option.store_final_state=True
        option.store_states=False
        result=self.copied_processor.run_state(init_state =self.initial_state,options=option)
        state=result.final_state
        return state 

    def set_up_1D_experiment(self,title='simulaiton',xlabel='t(us)'):
        '''
        set up experiment for 1D system. 
        self.final_state_list is the list catch the density matrix of final state
        self.y_array is the list catch the qubit excited population of final state
        and then set up to plot the figure

        '''
        self.final_state_list=[]
        self.y_array=np.zeros(len(self.x_array))
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(8,6))
        self.line, = self.ax.plot(self.x_array, self.y_array)
        plt.title(title,fontsize=25)
        plt.xlabel(xlabel,fontsize=18)
        plt.ylabel("qubit expected population",fontsize=18)
        plt.ylim((0,1))

    def post_process(self,circuit,i,readout_type='read qubit'):
        '''
        simulate the circuit and get the data
        refresh the plot
        '''
        final_state=self.run_circuit(circuit)
        self.final_state_list.append(final_state)
        if readout_type=='read qubit':
            expected_population=expect(num(self.processor.dims[0]),final_state.ptrace(0))
        elif readout_type=='read phonon':
            expected_population=expect(num(self.processor.dims[1]),final_state.ptrace(1))
            plt.ylim((0,np.max(self.y_array)*1.1))
            plt.ylabel("phonon expected num",fontsize=18)
        else:
            raise NameError('readout object select wrong')
        expected_population=np.abs(expected_population)
    
        self.y_array[i]=expected_population
        self.line.set_ydata(self.y_array)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def qubit_T1_measurement(self):
        '''
        simulation of qubit T1 
        '''
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='phonon T1')
        i=0
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0)
            circuit.add_gate('Wait',targets=0,arg_value=t)
            self.post_process(circuit,i)
            i=i+1

# %%
t1=[10]
#T2 list of the system 104
t2=[15]
#set up the processor and compiler,qb5d97 is the qubit we play around
test_processor=demo_processor(1,t1,t2,[2])
test_compiler = demo_Compiler(test_processor.num_qubits,\
    test_processor.params, test_processor.pulse_dict)
test_simulation=Simulation(test_processor,test_compiler)
# %%
test_simulation.qubit_T1_measurement()
# %%
test_simulation.copied_processor.plot_pulses()
# %%
