
import numpy as np
from qutip.qip.compiler import Instruction
from qutip.qip.compiler import GateCompiler
import matplotlib.pyplot as plt

#define gauss block shape
def gauss_block(t,sigma,amplitude,duration):
    if t<2*sigma:
        return amplitude*np.exp(-0.5*((t-2*sigma)/sigma)**2)
    elif t>duration-2*sigma:
        return amplitude*np.exp(-0.5*((t+2*sigma-duration)/sigma)**2)
    else:
        return amplitude
gauss_block=np.vectorize(gauss_block)

# define modulated gauss block shape
def gauss_block_modulated(t,sigma,amplitude,duration,omega,phase):
    return gauss_block(t,sigma,amplitude,duration)*np.cos(omega*t+phase)

#define gauss_block shape modulated which used for qubit XY plane rotation, mostly used 
def gauss_block_modulated_XY_rotate_compiler(gate,args):   
    """
    Compiler for the X-axis_Rotate gate, with gaussian block shape
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]

    #set pulse strength (rabi frequency)
    Omega=gate.arg_value.get('Omega',parameters["Omega"])*np.pi*2

    #set pulse raising time (sigma)
    gate_sigma=gate.arg_value.get('sigma',0.01)

    #the detuning of the frequency of the driving field
    detuning=gate.arg_value.get('detuning',0)*2*np.pi

    #rotate direction of the qubit, 0 means X, np.pi/2 means Y axis
    rotate_direction=gate.arg_value.get('rotate_direction',0)

    #set pulse duration
    duration=gate.arg_value['duration']
    
    time_step=3e-3
    tlist = np.linspace(0, duration, int(duration/time_step))
    coeff1 = gauss_block_modulated(tlist, gate_sigma, Omega, duration,detuning, rotate_direction)
    coeff2 = gauss_block_modulated(tlist, gate_sigma, Omega, duration,detuning, rotate_direction+np.pi/2)

    pulse_info =[ ("X-axis_R", coeff1),("Y-axis_R",coeff2) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

def gauss_rx_rz_compiler(gate, args):
    """
    Compiler for the XY-plane_Rotate gate, with a changable frequency shift.
    The propose of this gate is using probing qubit at certain interaction place (frequency),
    to separate state preparation and probing part, for better state preparation
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]

    #set pulse strength (rabi frequency)
    Omega=gate.arg_value.get('Omega',parameters["Omega"])*np.pi*2

    #set pulse raising time (sigma)
    gate_sigma=gate.arg_value.get('sigma',0.01)

    #the detuning of the frequency of the driving field
    detuning=gate.arg_value.get('detuning',0)*2*np.pi

    #rotate direction of the qubit, 0 means X, np.pi/2 means Y axis
    rotate_direction=gate.arg_value.get('rotate_direction',0)

    #set pulse duration
    duration=gate.arg_value['duration']

    sigma_starkshift=gate.arg_value.get('sigma_starkshift',0.01)
    amplitude_starkshift=gate.arg_value.get('amplitude_starkshift',0)*np.pi*2
    
    time_step=3e-3
    tlist = np.linspace(0, duration, int(duration/time_step))
    coeff1 = gauss_block_modulated(tlist, gate_sigma, Omega, duration,detuning, rotate_direction)
    coeff2 = gauss_block_modulated(tlist, gate_sigma, Omega, duration,detuning, rotate_direction+np.pi/2)
    coeff3 = gauss_block(tlist,sigma_starkshift,amplitude_starkshift,duration)

    pulse_info =[ ("X-axis_R", coeff1),("Y-axis_R",coeff2),("Z-axis_R",coeff3) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

def swap_phonon_compiler(gate, args):
    """
    Compiler for the X-axis_Rotate gate
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]
    g= parameters["g"]  # find the coupling strength for the target qubit
    gate_sigma =0.01
    amplitude =parameters['phonon_omega_z'][targets[1]-1]
    duration = 1/g/4*2*np.pi
    tlist = np.linspace(0, duration, 1000)
    coeff = gauss_block(tlist,gate_sigma, amplitude, duration)
    pulse_info =[ ("Z-axis_R", coeff) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

#define sigma_z gaussian block shape pulse
def starkshift_compiler(gate, args):
    """
    Compiler for the X-axis_Rotate gate
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]
    g= parameters["g"]  # find the coupling strength for the target qubit
    gate_sigma =0.02
    duration=gate.arg_value['duration']
    detuning=gate.arg_value.get('detuning',0)
    amplitude=parameters['phonon_omega_z'][targets[1]-1]+detuning*2*np.pi
    time_step=3e-3
    tlist = np.linspace(0, duration, int(duration/time_step))
    coeff = gauss_block(tlist,gate_sigma, amplitude, duration)
    pulse_info =[ ("Z-axis_R", coeff) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

#define gauss shape 
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

class HBAR_Compiler(GateCompiler):  # compiler class
    def __init__(self, num_qubits, params, pulse_dict):
        super(HBAR_Compiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict)
        # pass our compiler function as a compiler for RX (rotation around X) gate.
        self.gate_compiler["X_R"] = gauss_rx_compiler
        self.gate_compiler["Wait"]=wait_complier
        self.gate_compiler["swap"]=swap_phonon_compiler
        self.gate_compiler['Z_R_GB']=starkshift_compiler
        self.gate_compiler['XY_R_GB']= gauss_block_modulated_XY_rotate_compiler
        self.gate_compiler['XYZ_R_GB']=gauss_rx_rz_compiler
        self.args.update({"params": params})