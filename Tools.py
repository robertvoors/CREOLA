from scipy import interpolate
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.optimize

def calc_stats(time, signal):
   std = np.std(signal)
   mean = np.mean(signal)
   #print "MEAN: ", mean 
   #print "STD: ", std
   return mean, std

def make_plot(time, signal):
#   plt.close()
   plt.plot(time,signal)
#   plt.show()

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
	


def read_csv(filename):
   with open(filename) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=';')
      header = []
      header = next(csv_reader)
      date_recorded_string = next(csv_reader)
      for i in range(5):
         bl = next(csv_reader)
      header = next(csv_reader)
      units = next(csv_reader)
      rows = []
      for row in csv_reader:
         rows.append(row)
      data = np.array(rows)
    
   time = data[:,0].astype(np.float)
   signal = data[:,1].astype(np.float)
   return time, signal


def createList(r1, r2, dr=1):
    return np.arange(r1, r2+dr, dr)


def load_data(prefix, im_start, im_end):
   idxlist = createList(im_start, im_end)
   times1 = []
   sigch1 = []
   for idx in idxlist:
      idx_str = str(idx).zfill(3)
      filename = '../CSV_DATA/'+prefix+"_"+idx_str+'.csv'
      time, signal = read_csv(filename)
      times1.append(time)
      sigch1.append(signal)
   times1 = np.asarray(times1)
   sigch1 = np.asarray(sigch1)
   return times1, sigch1


def get_means_stds(prefix, im_start, im_end):
   idxlist = createList(im_start, im_end)
   stds = []
   means = []
   for idx in idxlist:
      idx_str = str(idx).zfill(3)
      filename = '../CSV_DATA/'+prefix+'_'+idx_str+'.csv'
      time, signal = read_csv(filename)
      mean, std = calc_stats(time, signal)
      stds.append(std)
      means.append(mean)
   stds = np.asarray(stds)
   means = np.asarray(means)
   return means, stds


def get_EVOA_cal_results():
   # Measurement results:
   volt_in = np.linspace(0,5,num=21)
   power_out = np.asarray([10.280, 10.264, 10.199, 10.056, 9.792, 9.362, 8.701, 7.785, 6.624, 5.283, 3.894, 2.614, 1.578, 0.849, 0.404, 0.167, 0.055, 0.009, -0.008, -0.013, -0.014])
   # "Correct" for negative output
   power_out = power_out - np.min(power_out)
   att = 1.0 - (power_out / np.max(power_out))
   return volt_in, att

def get_EVOA_attenuation(in_volt):
   # This routine returns attenuation level(s) from the EVOA, as function of input voltage.
   # Note, that this is addditional attenuation, as the minimum attenuation of the EVOA is 1.5 dB (TBC).
   # Measurement results:
   volt_in, att = get_EVOA_cal_results()
   # create 1-D interpolation function. It is a smooth function, so a quadratic interpolation is sufficient. There is no need to do a fucntion fit.
   f = interpolate.interp1d(volt_in, att, kind='quadratic') 
   att_out = f(in_volt)
   return att_out


def get_EVOA_voltage(in_attenuation):
   # This routine returns voltage level(s) as input to the EVOA, as function of required attentuation.
   # Note, that this is addditional attenuation, as the minimum attenuation of the EVOA is 1.5 dB (TBC).
   # Measurement results:
   volt_in, att = get_EVOA_cal_results()
   # create 1-D interpolation function. It is a smooth function, so a quadratic interpolation is sufficient. There is no need to do a fucntion fit.
   f = interpolate.interp1d(att, volt_in, kind='quadratic') 
   volt_out = f(in_attenuation)
   return volt_out



