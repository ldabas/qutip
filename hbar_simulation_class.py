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
import hbar_fitting

class Simulation():
    '''
    Setting the simulated experiment
    '''
    def __init__(self,processor,compiler,t_list=np.linspace(0.1,10,100),
        detuning_list=np.linspace(-0.3,1,100),swap_time_list=[],artificial_detuning=0,
        reading_time=None,initial_state=None):
        self.qubit_probe_params={}
        self.phonon_drive_params={}
        self.processor=processor
        self.compiler=compiler
        self.t_list=t_list
        self.detuning_list=detuning_list
        self.swap_time_list=swap_time_list
        self.artificial_detuning=artificial_detuning
        self.reading_time=reading_time
        self.fit_result=[]
        if not initial_state:
            self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        else:
            self.initial_state=initial_state

    def run_circuit(self,circuit):
        self.processor.load_circuit(circuit, compiler=self.compiler)
        option=Options()
        option.store_final_state=True
        option.store_states=False
        result=self.processor.run_state(init_state =self.initial_state,options=option)
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
    
    def ideal_phonon_fock(self,fock_number):
        '''
        set the initail state as phonon in specific fock state and qubit in g state
        '''
        self.initial_state=basis(self.processor.dims, [0]+[fock_number]+[0]*(self.processor.N-2))

    def ideal_coherent_state(self,alpha):
        '''
        set the initail state as phonon in specific coherent state and qubit in g state
        '''
        self.initial_state=tensor(fock(self.processor.dims[0],0),coherent(self.processor.dims[1],alpha))

    def ideal_qubit_state(self,expected_z):
        self.initial_state=tensor(np.sqrt(1-expected_z)*fock(self.processor.dims[0],0)+np.sqrt(expected_z)*fock(self.processor.dims[0],1),
        basis(self.processor.dims[1:],[0]*(self.processor.N-1)))

    def generate_fock_state(self,fock_number,detuning=0):
        '''
        simulation of using qubit phonon swap to generate phonon fock state
        '''
        self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        circuit = QubitCircuit((self.processor.N))
        for swap_t in self.swap_time_list[:fock_number]:
            circuit.add_gate("X_R", targets=0)
            circuit.add_gate('Z_R_GB',targets=[0,1],arg_value={'duration':swap_t,'detuning':detuning})
        if fock_number!=0:
            self.initial_state=self.run_circuit(circuit)
        print('fidelity of phonon fock {} :'.format(fock_number),expect(self.initial_state.ptrace(1),fock(self.processor.dims[1],fock_number)))
    
    def generate_coherent_state(self,phonon_drive_params=None):
        '''
        simulation of driving phonon mediated by qubit, expect a coherent state
        '''
        if not(phonon_drive_params==None):
            self.phonon_drive_params=phonon_drive_params
        self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        circuit = QubitCircuit((self.processor.N))
        circuit.add_gate("XY_R_GB", targets=0,arg_value=self.phonon_drive_params)
        self.initial_state=self.run_circuit(circuit)

    def qubit_pi_pulse(self):
        '''
        simulation of giving pi pulse on qubit
        '''
        # self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        circuit = QubitCircuit((self.processor.N))
        circuit.add_gate("X_R", targets=0)
        self.initial_state=self.run_circuit(circuit)

    def phonon_T1_measurement(self):
        '''
        simulation of phonon T1, first excite qubit and then swap it to phonon. Wait some time and 
        swap back to readout
        '''
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='phonon T1')
        i=0
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0)
            circuit.add_gate('swap',targets=[0,1])
            circuit.add_gate('Wait',targets=0,arg_value=t)
            circuit.add_gate('swap',targets=[0,1])
            self.post_process(circuit,i)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
        self.fit_result.append(self.fitter.fit_T1())


    def phonon_rabi_measurement(self,detuning=0):
        '''
        simulation for qubit phonon rabi oscillation.
        We excite qubit first, then put qubit and phonon on resonance. Sweeping resonance time and readout
        '''
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='phonon rabi')
        i=0
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0)
            circuit.add_gate('Z_R_GB',targets=[0,1],arg_value={'duration':t,'detuning':detuning})
            self.post_process(circuit,i)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
        self.fit_result.append(self.fitter.fit_phonon_rabi())

    def qubit_rabi_measurement(self,qubit_probe_params={}):
        '''
        This is qubit time domain rabi. We use gaussain square pulse to excite qubit while sweeping 
        pulse length.
        '''
        if not(qubit_probe_params=={}):
            self.qubit_probe_params=qubit_probe_params
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='qubit rabi')
        i=0
        for t in tqdm(self.x_array):
            self.qubit_probe_params['duration']=t
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("XY_R_GB", targets=0,arg_value=self.qubit_probe_params)
            self.post_process(circuit,i)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
        self.fit_result.append(self.fitter.fit_phonon_rabi())
    
    def qubit_shift_wait(self,qubit_probe_params={},if_fit=True):
        '''
        We want to see if there is any population in the phonone, how will it change
        the qubit at different time. For a specific initial state, we put qubit closer to phonon, then just
        wait for different time and readout qubit.
        '''
        if not(qubit_probe_params=={}):
            self.qubit_probe_params=qubit_probe_params
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='qubit rabi')
        i=0
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            self.qubit_probe_params['duration']=t
            self.qubit_probe_params['Omega']=0
            circuit.add_gate("XYZ_R_GB", targets=0,arg_value=self.qubit_probe_params)
            self.post_process(circuit,i)
            i=i+1
        if if_fit:
            self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
            self.fit_result.append(self.fitter.fit_phonon_rabi())

    def qubit_ramsey_measurement(self,artificial_detuning=None,starkshift_amp=0,if_fit=True):
        '''
        This is for qubit ramsey measurement.
        do a qubit half pi pulse first, then wait, and another half pi pulse. 
        At the waiting time, we can also shift qubit to interaction point
        '''

        self.x_array=self.t_list
        self.set_up_1D_experiment(title='qubit Ramsey')
        if not(artificial_detuning==None):
            self.artificial_detuning=artificial_detuning
        i=0
        starkshift_param={'detuning':starkshift_amp}
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2})
            starkshift_param['duration']=t
            circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=starkshift_param)
            circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2,\
                'rotate_direction':2*np.pi*self.artificial_detuning*t})
            self.post_process(circuit,i)
            i=i+1
        if if_fit:
            self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
            self.fit_result.append(self.fitter.fit_T2())

    def spec_measurement(self,qubit_probe_params={},readout_type='read qubit'):
        if not(qubit_probe_params=={}):
            self.qubit_probe_params=qubit_probe_params
        self.x_array=self.detuning_list
        self.set_up_1D_experiment(title='qubit spec',xlabel='detuning (MHz)')
        i=0
        for detuning in tqdm(self.x_array):
            self.qubit_probe_params['detuning']=detuning
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("XYZ_R_GB", targets=0,arg_value=self.qubit_probe_params)
            self.post_process(circuit,i,readout_type)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)


    def wigner_measurement_1D(self,phonon_drive_params=None,steps=40,
    displacement_type='simulated',second_pulse_flip=False,set_alpha_range=None):
        '''
        displacement_type can be choose as 'simulated' or 'ideal'
        the second_pulse_flip is choose if the second half pi pulse change to opposite direction
        '''
        stored_initial_state=self.initial_state
        if set_alpha_range:
            self.alpha=set_alpha_range
        else:
            #calibration of the alpha axis based on wigner fitting
            self.generate_coherent_state(phonon_drive_params)
            self.fit_wigner()
       
        self.x_array=np.linspace(-np.abs(self.alpha),np.abs(self.alpha),steps)
        
        #set experiment up
        self.initial_state=stored_initial_state
        self.set_up_1D_experiment(title='wigner measurement',xlabel='alpha')
        Omega_max=self.phonon_drive_params['Omega']
        Omega_list=np.linspace(0,Omega_max,steps)
        
        for i, alpha in enumerate(self.x_array):
            self.phonon_drive_params['Omega']=Omega_list[i]
            circuit = QubitCircuit((self.processor.N))
            if displacement_type=='simulated':
                circuit.add_gate("XY_R_GB", targets=0,arg_value=self.phonon_drive_params)
            elif displacement_type=='ideal':
                displacement_operator=tensor(qeye(self.processor.dims[0]),
                qt.displace(self.processor.dims[1],alpha))
                if self.initial_state.shape[1]==1:
                    self.initial_state=displacement_operator*stored_initial_state
                else:
                    self.initial_state=displacement_operator*stored_initial_state*displacement_operator
            else:
                raise NameError('displacement type select wrong')
            circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2})
            circuit.add_gate('Wait',targets=0,arg_value=self.reading_time)
            if second_pulse_flip:
                circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2,\
                'rotate_direction':2*np.pi*self.artificial_detuning*self.reading_time+np.pi})
            else:
                circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2,\
                'rotate_direction':2*np.pi*self.artificial_detuning*self.reading_time})
            self.post_process(circuit,i)
            i=i+1

    def wigner_measurement_2D(self,phonon_drive_params=None,steps=40):
        '''
        steps is the number of the point in the ploting axis also the simulation times
        '''
        stored_initial_state=self.initial_state
        self.generate_coherent_state(phonon_drive_params)
        self.fit_wigner()
        axis=np.linspace(-np.abs(self.alpha),np.abs(self.alpha),steps)
        self.initial_state=stored_initial_state
        Omega_alpha_ratio=self.phonon_drive_params['Omega']/self.alpha
        storage_list_2D=[]
        for x in tqdm(axis):
            self.x_array=axis
            self.set_up_1D_experiment(title='wigner measurement',xlabel='alpha')
            for i, y in enumerate(axis):
                circuit = QubitCircuit((self.processor.N))
                self.phonon_drive_params['Omega']=np.sqrt(x**2+y**2)*Omega_alpha_ratio
                self.phonon_drive_params['rotate_direction']=np.angle(x+1j*y)+0.5*np.pi
                circuit.add_gate("XY_R_GB", targets=0,arg_value=self.phonon_drive_params)
                circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2})
                circuit.add_gate('Wait',targets=0,arg_value=self.reading_time)
                circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2,\
                    'rotate_direction':2*np.pi*self.artificial_detuning*self.reading_time})
                self.post_process(circuit,i)
            storage_list_2D.append(self.y_array*2-1)

        xx,yy=np.meshgrid(axis,axis)
        plt.figure(figsize=(6,6))
        plt.contourf(xx, yy, storage_list_2D,40,cmap='RdBu_r')
        plt.gca().set_aspect('equal')
        plt.colorbar().set_label("qubit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        return storage_list_2D
    
    def fit_wigner(self):
        wigner_array=np.linspace(-10,10,201)
        wigner_2D=qt.wigner(self.initial_state.ptrace(1),wigner_array,wigner_array)
        position=np.where(wigner_2D==np.amax(wigner_2D))
        self.alpha=(1j*wigner_array[position[0]]+wigner_array[position[1]])[0]
        alpha_fidelity=expect(self.initial_state.ptrace(1),coherent(self.processor.dims[1],self.alpha))
        print(f'alpha is {self.alpha}, fidelity is {alpha_fidelity}')

    def generate_cat(self):
        # initial_state is |g,0>
        circuit = QubitCircuit((self.processor.N))
        self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))

        # prepare qubit in superpostion, get |g,0>+|e,0>
        circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2})
        
        # selectively drive the phonon, get |g,alpha>+|e,0|
        circuit.add_gate("XY_R_GB", targets=0,arg_value=self.phonon_drive_params)
        self.initial_state=self.run_circuit(circuit)
        qt.plot_wigner_fock_distribution(self.initial_state.ptrace(1))

        # selectively give the qubit pi pulse, get |g>(|alpha>+|0>)
        circuit = QubitCircuit((self.processor.N))
        circuit.add_gate("XY_R_GB", targets=0,arg_value=self.qubit_probe_params)
        self.initial_state=self.run_circuit(circuit)
        qt.plot_wigner_fock_distribution(self.initial_state.ptrace(1))
        qt.plot_wigner_fock_distribution(self.initial_state.ptrace(0))


