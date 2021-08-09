import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gauss1D
from scipy import signal
from scipy.optimize import curve_fit
from lmfit.models import LorentzianModel,ConstantModel
from scipy.stats import poisson

def Lorentz(x,x0,w,A,B):
    return A*(1-1/(1+((x-x0)/w)**2))+B
    
def Cos(x, a, f, os,phi):
    return os + a * np.cos( (f * x )+phi)

def Exp_decay(x, A, tau, ofs):
    return A * np.exp(-x / tau) + ofs

def Exp_sine(x, a, tau, ofs, freq , phase):
    return ofs + a * (np.exp(-x/ tau) * np.cos(2 * np.pi * (freq * x + phase)))

def Exp_plus_sine(x, a0, a1,tau1,tau2, ofs, freq , phase):
#     print(a, tau, ofs, freq, phase)
    return ofs + a0 * np.exp(-x/ tau1)*(np.cos(2 * np.pi * (freq * x + phase)))\
        +a1*np.exp(-x/ tau2)

def continue_fourier_transform(time_array,amp_array,freq_array):
    ft_list=[]
    dt=time_array[1]-time_array[0]
    num=len(time_array)
    for w in freq_array:
        phase=np.power((np.zeros(num)+np.exp(-2*np.pi*1j*w*dt)),np.array(range(num)))      
        ft_data=np.sum(amp_array*phase)
        ft_list.append(np.abs(ft_data))
    return np.array(ft_list)


class fitter(object):
    def __init__(self,x_array,y_array):
        self.x_array=x_array
        self.y_array=y_array
    def fit_T1(self):
        #fit for exp decay, e.g. T1
        x_array=self.x_array
        y_array=self.y_array
        minimum_amp=np.min(y_array)
        normalization=y_array[-1]
        popt,pcov =curve_fit(Exp_decay,x_array,y_array,[-(normalization-minimum_amp),20,normalization])
        fig,ax=plt.subplots()
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_decay(x_array,*popt),label='fitted')
        plt.title(('T1 = %.3f us '% (popt[1])))
        plt.legend()
        plt.show()
        return {'T1':popt[1]}
    
    def fit_phonon_rabi(self):
        #fit for phonon_qubit oscillation
        x_array=self.x_array
        y_array=self.y_array
        minimum_point=signal.argrelextrema(y_array, np.less)[0]
        delay_range=x_array[-1]-x_array[0]
        minimum_amp=np.min(y_array)
        max_amp=np.max(y_array)
        if len(minimum_point)>1:
            freq_guess=1/(x_array[minimum_point[1]]-x_array[minimum_point[0]])
        elif len(minimum_point)==1:
            freq_guess=1/(x_array[minimum_point[0]]-x_array[0])
        else:
            freq_guess=1/(x_array[-1]-x_array[0])

        popt,pcov =curve_fit(Exp_plus_sine,x_array,y_array,[-(max_amp-minimum_amp),0.5,
                                                            delay_range/3,delay_range/3,0,freq_guess,0])
     
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_plus_sine(x_array,*popt),label='fitted')
        plt.legend()
        plt.title('swap time = %.3f us '% (1/popt[-2]/2))
        plt.show()
        return {'swap_time':1/popt[-2]/2}

    def fit_T2(self):
        x_array=self.x_array
        y_array=self.y_array
        y_smooth=signal.savgol_filter(y_array,51,4)
        y_smooth=signal.savgol_filter(y_smooth,51,4)

        minimum_number=len(*signal.argrelextrema(y_smooth, np.less))
        amp_range=x_array[-1]-x_array[0]
        minimum_amp=np.min(y_array)
        normalization=np.average(y_array)
        popt,pcov =curve_fit(Exp_sine,x_array,y_array,[(normalization-minimum_amp),amp_range/3,normalization,minimum_number/amp_range,0])
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_sine(x_array,*popt),label='fitted')
        plt.legend()
        plt.title('T2 = %.3f us, detuning is %.3f MHz '% (popt[1],popt[3]))
        plt.show()
        return {'T2': popt[1],
            'delta': popt[3]
            }
    
    def fit_single_peak(self):
        x_array=self.x_array
        y_array=self.y_array
        max_point=x_array[np.argsort(y_array)[-1]]
        popt,pcov =curve_fit(Lorentz,x_array,y_array,[max_point,0.1,np.max(y_array),np.min(y_array)])
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Lorentz(x_array,*popt),label='fitted')
        plt.legend()
        plt.title(('w0 = %.5f MHz '% (popt[0])))
        plt.show()
        return  {'w0': popt[0],
        'width':popt[1],
        }

    def fit_phase_calibration(self):
        x_array=self.x_array
        y_array=self.y_array
        minimum_amp=np.min(y_array)
        max_amp=np.max(y_array)
        popt,pcov =curve_fit(Cos,x_array,y_array,[(max_amp-minimum_amp)/2,1,(max_amp+minimum_amp)/2,0])
        fig,ax=plt.subplots()
        if popt[0]<0:
            phase=-popt[-1]+np.pi
        else:
            phase=-popt[-1]
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Cos(x_array,*popt),label='fitted')
        plt.title(('phi = %.3f'% (phase)))
        plt.legend()
        plt.show()
        print(popt)
        return {'phi':phase}



    def fit_multi_peak(self,peaks):
        x_array=self.x_array
        y_array=self.y_array
        

        #find peaks position first
        max_point=signal.argrelextrema(y_array, np.greater)[0]
        print(x_array[max_point])

        # peak_position=[]
        peak_position=x_array[max_point[np.argsort(y_array[max_point])[-peaks:]]]
        print('peaks position:', peak_position)
        # for i in max_point:
        #     if y_array[i]> threshold:
        #         peak_position.append(x_array[i])

        #define the peak fitting model
        def add_peak(prefix, center, amplitude=0.3, sigma=0.05):
            peak = LorentzianModel(prefix=prefix)
            pars = peak.make_params()
            pars[prefix + 'center'].set(center)
            pars[prefix + 'amplitude'].set(amplitude)
            pars[prefix + 'sigma'].set(sigma, min=0)
            return peak, pars
        
        model = ConstantModel(prefix='bkg')
        background=(y_array[0]+y_array[-1])/2
        params = model.make_params(c=background)
        rough_peak_positions =peak_position
        for i, cen in enumerate(rough_peak_positions):
            peak, pars = add_peak('lz%d' % (i), cen,amplitude=y_array[
                max_point[np.argsort(y_array[max_point])[-peaks:]][i]])
            model = model + peak
            params.update(pars)

        init = model.eval(params, x=x_array)
        result = model.fit(y_array, params, x=x_array)
        comps = result.eval_components()

        fig,ax=plt.subplots()
        ax.plot(x_array, y_array, label='data')
        # ax.plot(x_array,y_smooth,label='smooth')
        ax.plot(x_array, result.best_fit, label='best fit')

        for name, comp in comps.items():
            if "lz" in name:
                plt.plot(x_array, comp+comps['bkg'], '--', label=name)
        plt.legend(loc='upper right')
        plt.show()

        peak_height_list=[]
        peak_center_list=[]
        peak_width_list=[]
        for i in range(len(peak_position)):
            peak_height_list.append( result.params['lz%dheight'%(i)].value)
            peak_center_list.append( result.params['lz%dcenter'%(i)].value)
            peak_width_list.append(result.params['lz%dfwhm'%(i)].value)


        return [peak_center_list,peak_height_list,peak_width_list]

    def sum_lorentz_fit(self,peak_positions,peak_number,fit_poisson=True, init_sigma=[0.035, 0.025, 0.041],init_amp=[0.,0.7], center_var=0.2, smooth=[11,3], debug=False,background=None,plt_nr=0):
        '''
        can specify init, min, max values for sigma, and min, max values for the amplitude
        must specify initial peak positions. min, max for peak centers are +- 1 init sigma
        '''
        #read file
        freq=self.x_array
        y=self.y_array
        print('len(y): ' + str(len(y)))
        if background is None:
            background=(y[0]+y[-1])/2
        y_smooth=signal.savgol_filter(y,smooth[0],smooth[1])
        #we define the initial value for the peak position
        peak_positions = peak_positions[:peak_number]
        # name freq as xdat, amp as ydat
        xdat =freq
        ydat =y

        #define the initial value for peak height
        def find_nearest(array, value):
            array = np.array(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        initial_peak_height=[]

        for i in range(peak_number):
            initial_peak_height.append(find_nearest(freq,peak_positions[i]))


        #define a function for adding peak in model. We need to give initial value for peak center
        def add_peak(prefix, center, amplitude, sigma, c_bounds, amp_bounds, sig_bounds):
            peak = LorentzianModel(prefix=prefix)
            peak.set_param_hint(prefix + 'center', value=center, min=c_bounds[0], max=c_bounds[1])
            peak.set_param_hint(prefix + 'amplitude', value=amplitude/0.32*sigma, min=0)
            peak.set_param_hint(prefix + 'height', value=amplitude, min=amp_bounds[0], max=amp_bounds[1])
            peak.set_param_hint(prefix + 'sigma', value=sigma, min=sig_bounds[0], max=sig_bounds[1])

            pars = peak.make_params()

            # pars[prefix + 'center'].set(center, min=c_bounds[0], max=c_bounds[1])
            # pars[prefix + 'amplitude'].set(amplitude, min=amp_bounds[0], max=amp_bounds[1])
            # pars[prefix + 'sigma'].set(sigma, min=sig_bounds[0], max=sig_bounds[1])
            return peak, pars

        #set up model
        model = ConstantModel(prefix='bkg')
        params = model.make_params(c=background)

        #add peaks for model, based on the peaks and threshold find before

        for i, cen in enumerate(peak_positions):
            init_center = [peak_positions[i]-center_var, peak_positions[i]+center_var]
            peak, pars = add_peak('lz%d' % (i+1), peak_positions[i],initial_peak_height[i], init_sigma[0], 
                                c_bounds=init_center, amp_bounds=init_amp, sig_bounds=init_sigma[1:])
            model = model + peak
            params.update(pars)

        #fit the model
        init = model.eval(params, x=xdat)
        # result = model.fit(ydat, params, x=xdat)
        result = model.fit(y_smooth, params, x=xdat)
        comps = result.eval_components()
        if debug:
            print(result.fit_report())
        #plot the result 
        figure, ax_spec = plt.subplots(figsize=(8,6))
        ax_spec.plot(xdat, ydat, label='data')
        ax_spec.plot(xdat, result.best_fit, label='best fit')
        
        for name, comp in comps.items():
            if "lz" in name:
                ax_spec.plot(xdat, comp+comps['bkg'], '--', label=name)
        ax_spec.legend()
        figure.show()

        #find height for each peak, and normalize it to get population of each fock state
        peak_height_list=np.zeros((10))
        peak_center_list=np.zeros((10))
        peak_width_list=np.zeros((10))
        for i in range(len(peak_positions)):
            peak_height_list[i] = result.params['lz%dheight'%(i+1)].value
            peak_center_list[i] = result.params['lz%dcenter'%(i+1)].value
            peak_width_list[i] = result.params['lz%dsigma'%(i+1)].value
        bkgc = result.params['bkgc'].value
        peak_height_list=np.abs(np.array(peak_height_list))
        total_height = np.sum(peak_height_list)
        #plot population distributation
        xaxis=range(len(peak_height_list))
        peak_height_list=peak_height_list/np.sum(peak_height_list)

        figure, ax_distributation = plt.subplots(figsize=(8,6))
        ax_distributation.plot(xaxis,peak_height_list,drawstyle= 'steps-mid',label='relative peak height')
        plt.ylim(0,1)

        if fit_poisson:
            #fit it with poisson distributation
            def fit_function(k, lamb):
                '''fit poisson function, parameter lamb is the fit parameter'''
                return poisson.pmf(k, lamb)

            # fit poisson distribution
            parameters, cov_matrix = curve_fit(fit_function,xaxis,peak_height_list,[3])
            alpha = np.sqrt(parameters[0])
            alpha_err = np.sqrt(np.diag(cov_matrix)[0])
            ax_distributation.plot(
                xaxis,
                fit_function(xaxis, *parameters),
                marker='o', linestyle='',
                label='Fit result',
            )
            print('alpha = %.2f'% alpha)
            print('alpha_err =%.3f'% alpha_err)
        
        ax_distributation.legend()
        figure.show()
        return peak_height_list, peak_center_list, peak_width_list, alpha