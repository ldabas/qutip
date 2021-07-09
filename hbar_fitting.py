import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gauss1D
from scipy import signal
from scipy.optimize import curve_fit
from lmfit.models import LorentzianModel,ConstantModel

def Lorentz(x,x0,w,A,B):
    return A*(1-1/(1+((x-x0)/w)**2))+B

def Cos(x, a, f, os):
    return os + a * np.cos(2 * np.pi * (f * x ))

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
    
    def fit_multi_peak(self,peaks):
        x_array=self.x_array
        y_array=self.y_array
        

        #find peaks position first
        y_smooth=signal.savgol_filter(y_array,11,4)
        max_point=signal.argrelextrema(y_smooth, np.greater)[0]
        print(x_array[max_point])
        # peak_position=[]
        peak_position=x_array[max_point[np.argsort(y_smooth[max_point])[-peaks:]]]
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
        params = model.make_params(c=0)
        rough_peak_positions =peak_position
        for i, cen in enumerate(rough_peak_positions):
            peak, pars = add_peak('lz%d' % (i), cen,amplitude=y_smooth[
                max_point[np.argsort(y_smooth[max_point])[-peaks:]][i]])
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

