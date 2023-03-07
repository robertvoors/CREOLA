import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle
import scipy.optimize
from scipy import stats, interpolate, signal
import sys
import Tools as mytools
import copy


# ============================================================
# This file contains a number of routines for the analysis of the power measurements.
# The routines make use of generic routines written by RV, whih are stored in Tools - imported as mytools.
# The mostly used are:
# - load_data(PREFIX,start,stop) :  returns 2 1-D numpy arrays: times and signals                     (time series)
# - get_means_stds(PREFIX,start,stop) :  returns 2 1-D numpy arrays: means and standard_deviations    (statistics)
# - fit_sin : returns a dictionary with two key-value pairs: 'freq' and 'amp'
# In addition there are:
# - get_EVOA_cal_results
# - get_EVOA_attenuation

# 
# Thorlabs PDA30B2 
# detector gain range settings in dB
gains = np.asarray([0,10,20,30,40,50,60,70])  
# detector RMS noise, according to user manual in micro-Volts
specnoise = np.asarray([240, 214, 212, 211, 211, 212, 218, 243])

plotroot = '../Plots/'




def Write_noise_measurements_toFile_1feb():
   # This routine writes noise measurements with the Thorlabs detector to a pickle file
   # Measurements were done at all gains [from 0 to 70dB] and for 3 post-amplifier settings [1, 10 and 100 V/V] 
   print '*** Write_noise_measurements_toFile_1feb   ***'

   # Read in data
   means1, stds1     = mytools.get_means_stds('1feb_v2_ch1', 10, 17)
   means10, stds10   = mytools.get_means_stds('1feb_v2_ch1', 19, 26)
   means100, stds100 = mytools.get_means_stds('1feb_v2_ch1', 27, 34)
 
   # write to file: dark current and noise
   # first, turn it into a dictionary
   dark_noise_signal = {}
   signal_dict = {}
   signal_dict['AMP1'] = means1
   signal_dict['AMP10'] = means10
   signal_dict['AMP100'] = means100
   noise_dict = {}
   noise_dict['AMP1'] = stds1
   noise_dict['AMP10'] = stds10
   noise_dict['AMP100'] = stds100
   dark_noise_signal['signal'] = signal_dict
   dark_noise_signal['noise'] = noise_dict
   
   # save to pkl file
   with open('dark_noise_signal_1feb.pkl','wb') as f:
      pickle.dump(dark_noise_signal, f)


def Get_noise_data_1feb(noise_file='dark_noise_signal_1feb.pkl'):
	print '*** Get_noise_data_1feb   ***'
	with open(noise_file,'rb') as f:
	   dark_noise_sig = pickle.load(f)
	dark_signal = dark_noise_sig['signal']
	dark_noise = dark_noise_sig['noise']
	return dark_signal, dark_noise


def Plot_noise_1feb():
   print '***  Plot_noise_1feb  ****'
   means1, stds1     = mytools.get_means_stds('1feb_v2_ch1', 10, 17)
   means10, stds10   = mytools.get_means_stds('1feb_v2_ch1', 19, 26)
   means100, stds100 = mytools.get_means_stds('1feb_v2_ch1', 27, 34)
 
   plt.close()
   plt.plot(gains, stds1*1000000, label = 'NO AMP gain')
   plt.plot(gains, stds10*100000, label = 'AMP gain 10 V/V')
   #plt.plot(gains, stds100*10000, label = 'AMP gain 100 V/V')
   plt.plot(gains, specnoise, label = 'spec')
   plt.show()
   plt.xlabel('PDA gain [dB]')
   plt.ylabel('RMS noise [muV]')
   plt.legend()
   plt.savefig(plotroot+'RMSnoise_1feb.png',dpi=300)

   plt.close()
   plt.plot(gains, means1*1000000, label = 'NO AMP gain')
   plt.plot(gains, means10*100000, label = 'AMP gain 10 V/V')
   #plt.plot(gains, means100*10000, label = 'AMP gain 100 V/V')
   plt.yscale('log')
   plt.show()
   plt.legend()
   plt.xlabel('PDA gain [dB]')
   plt.ylabel('Mean signal [muV]')
   plt.savefig(plotroot+'MeanSig_1feb.png',dpi=300)


def Plot_noise_30jan():
   print  '***  Plot_noise_30jan  ****'
   means_amp1, stds_amp1     = mytools.get_means_stds('30jan_ch1', 105, 112)
   means_amp10, stds_amp10   = mytools.get_means_stds('30jan_ch1', 88, 95)
   means_amp100, stds_amp100 = mytools.get_means_stds('30jan_ch1', 97, 104)

   plt.plot(gains, stds_amp1*1000000, label = 'NO AMP gain')
   plt.plot(gains, stds_amp10*100000, label = 'AMP gain 10 V/V')
   plt.plot(gains, stds_amp100*10000, label = 'AMP gain 100 V/V')
   plt.plot(gains, specnoise, label = 'spec')
   plt.show()
   plt.xlabel('PDA gain [dB]')
   plt.ylabel('RMS noise [muV]')
   plt.legend()
   plt.savefig(plotroot+'RMSnoise_30jan.png',dpi=300)

   plt.close()
   plt.plot(gains, means_amp1*1000000, label = 'NO AMP gain')
   plt.plot(gains, means_amp10*100000, label = 'AMP gain 10 V/V')
   plt.plot(gains, means_amp100*10000, label = 'AMP gain 100 V/V')
   plt.yscale('log')
   plt.show()
   plt.legend()
   plt.xlabel('PDA gain [dB]')
   plt.ylabel('Mean signal [muV]')
   plt.savefig(plotroot+'MeanSig_30jan.png',dpi=300)


def Make_EVOA_attentuation_plots():
   '***  Make_EVOA_attentuation_plots  ***'

   # This routine returns voltage level(s) as input to the EVOA, as function of required attentuation.
   # Note, that this is addditional attenuation, as the minimum attenuation of the EVOA is 1.5 dB (TBC).
   # [1] Measurement results:
   volt_in, att = mytools.get_EVOA_cal_results()
   # create 1-D interpolation function. It is a smooth function, so a quadratic interpolation is sufficient. There is no need to do a fucntion fit.
   f = interpolate.interp1d(att, volt_in, kind='quadratic') 

   plt.close()
   plt.plot(att, volt_in,'.',label="measured")
   plt.ylabel("Volt")
   plt.xlabel("Attenuation - input")
   plt.title("EVOA variable attenuation as function of Voltage input")
   plt.legend(loc='best')
   plt.show()
   plt.savefig(plotroot+'EVOA_Volt_per_Attenuation.png', dpi=300)

   plt.plot(volt_in, att,'.',label="measured")
   plt.xlabel("Volt - input")
   plt.ylabel("Attenuation")
   plt.title("EVOA variable attenuation as function of Voltage input")
   plt.legend(loc='best')
   plt.show()
   plt.savefig(plotroot+'EVOA_Attenuation_per_Volt.png', dpi=300)
   

def Plot_EVOA_attenuation(in_volt):
   print '***  Plot_EVOA_attenuation  ***'

   # Get data from EVOA calibration
   volt_in, att = mytools.get_EVOA_cal_results()
   # calculate attenuation
   att_out = mytools.get_EVOA_attenuation(in_volt)

   # make plots
   plt.close()
   plt.plot(volt_in, att,'.',label="measured")
   plt.plot(in_volt, att_out, 'o',label="interpolated")
   plt.xlabel("Volt - input")
   plt.ylabel("Attenuation")
   plt.title("EVOA variable attenuation as function of Voltage input")
   plt.legend(loc='best')
   plt.show()
   plt.savefig(plotroot+'EVOA_Attenuation_per_Volt_BWTEST.png', dpi=300)

def Plot_bandwidth_test_1feb():
	print '***  Plot_bandwidth_test_1feb  ***'

	time1, sig1 = mytools.load_data('1feb_ch1',65,75)
	time2, sig2 = mytools.load_data('1feb_ch1',76,86)
	time3, sig3 = mytools.load_data('1feb_ch1',87,97)

	freqs1 = []
	freqs2 = []
	freqs3 = []
	amps1 = []
	amps2 = []
	amps3 = []

	def append_amp_freq(time,sig):
	   t = time[i][:25000]
	   g = sig[i][:25000]
	   res = mytools.fit_sin(t,g)
	   freq = res['period']
	   amp = res['amp']
	   return freq, amp

	for i in range(11):
	   # case 1
	   freq, amp = append_amp_freq(time1,sig1)
	   freqs1.append(freq)
	   amps1.append(amp)
	   # case 2
	   freq, amp = append_amp_freq(time2,sig2)
	   freqs2.append(freq)
	   amps2.append(amp)
	   # case 3
	   freq, amp = append_amp_freq(time3,sig3)
	   freqs3.append(freq)
	   amps3.append(amp)

	freqs1 = np.asarray(freqs1) 
	freqs2 = np.asarray(freqs2)
	freqs3 = np.asarray(freqs3)
	amps1 = np.asarray(amps1)
	amps2 = np.asarray(amps2)
	amps3 = np.asarray(amps3)

	freqs = [1,3,6,10,30,60,120,250,500,1000,2000]

	plt.close()
	plt.plot(freqs,1./freqs1,label='case1')
	plt.plot(freqs,1./freqs2,label='case2')
	plt.plot(freqs,1./freqs3,label='case3')
	plt.legend(loc='best')
	#plt.ylim([0.099,.101])
	plt.xlabel('Input frequency [Hz]')
	plt.ylabel('Fitted frequency [Hz]')
	#plt.show()
	plt.xscale('log')
	plt.savefig(plotroot+'BandwidthTEST_Frequency_fit.png',dpi=300)

	plt.close()
	plt.plot(freqs,freqs*freqs1,label='case1')
	plt.plot(freqs,freqs*freqs2,label='case2')
	plt.plot(freqs,freqs*freqs3,label='case3')
	plt.legend(loc='best')
	#plt.ylim([0.099,.101])
	plt.xlabel('Input frequency [Hz]')
	plt.ylabel('Fitted/input frequency')
	plt.xscale('log')
	#plt.show()
	plt.savefig(plotroot+'BandwidthTEST_Frequency_fit2.png',dpi=300)

	plt.close()
	plt.plot(freqs,np.abs(amps1),label='case1')
	plt.plot(freqs,np.abs(amps2),label='case2')
	plt.plot(freqs,np.abs(amps3),label='case3')
	plt.legend(loc='best')
	#plt.ylim([0.09,.11])
	plt.xlabel('Input frequency [Hz]')
	plt.ylabel('Fitted amplitude [V]')
	#plt.show()
	plt.xscale('log')
	plt.savefig(plotroot+'BandwidthTEST_Amplitude_fit.png',dpi=300)

	plt.close()
	plt.plot(freqs,np.abs(amps1),label='case1')
	plt.plot(freqs,np.abs(amps2)*10*1.7783,label='case2')
	plt.plot(freqs,np.abs(amps3)*100*1.7783,label='case3')
	plt.legend(loc='best')
	#plt.ylim([0.09,.11])
	plt.xlabel('Input frequency [Hz]')
	plt.ylabel('Fitted amplitude [V]')
	#plt.show()
	plt.xscale('log')
	plt.savefig(plotroot+'BandwidthTEST_Amplitude_SCALED_fit.png',dpi=300)





def correct_for_dc(gain1, gain10, gain100):
	# Correct signal for dark current
	gain1_dc = copy.deepcopy(gain1)
	gain10_dc = copy.deepcopy(gain10)
	gain100_dc = copy.deepcopy(gain100)

	def get_noise_data(noise_file='dark_noise_signal.pkl'):
		with open(noise_file,'rb') as f:
			dark_noise_sig = pickle.load(f)
			dark_signal = dark_noise_sig['signal']
			dark_noise = dark_noise_sig['noise']
		return dark_signal, dark_noise
	
	dark_signal, dark_noise = get_noise_data()
	for i in range(8):
		gain1_dc[i][:]   = gain1[i][:]   - dark_signal['AMP1'][i]
		gain10_dc[i][:]  = gain10[i][:]  - dark_signal['AMP10'][i]
		gain100_dc[i][:] = gain100[i][:] - dark_signal['AMP100'][i]
	return gain1_dc, gain10_dc, gain100_dc



def Plot_sinewave_fits_30jan():
	print '***  Plot_sinewave_fits_30jan  ***'

	def get_sine_wave_data():
		times1, gain1     = mytools.load_data('30jan_ch1',140,147)
		times10, gain10   = mytools.load_data('30jan_ch1',150,157)
		times100, gain100 = mytools.load_data('30jan_ch1',160,167)
		return times1, gain1, times10, gain10, times100, gain100

	gains_lin = 10.**(gains/20.)

	# get sine wave data
	times1, gain1, times10, gain10, times100, gain100 = get_sine_wave_data()
	# correct for dark current
	gain1_dc, gain10_dc, gain100_dc = correct_for_dc(gain1, gain10, gain100)
	
	freqs1, freqs10, freqs100 = [], [], []
	amps1, amps10, amps100 = [], [], []
	freqs1_dc, freqs10_dc, freqs100_dc = [], [], []
	amps1_dc, amps10_dc, amps100_dc = [], [], []

	def append_amp_freq(time,sig):
	   t = time[i][:25000]
	   g = sig[i][:25000]
	   res = mytools.fit_sin(t,g)
	   freq = res['period']
	   amp = res['amp']
	   return freq, amp

	for i in range(8):
	   # Gain 1
	   freq, amp = append_amp_freq(times1,gain1)
	   freqs1.append(freq)
	   amps1.append(amp)
	   # Gain 10
	   freq, amp = append_amp_freq(times10,gain10)
	   freqs10.append(freq)
	   amps10.append(amp)
	   # Gain 100
	   freq, amp = append_amp_freq(times1,gain1)
	   freqs100.append(freq)
	   amps100.append(amp)

	   # Dark corrected 
	   # Gain 1
	   freq, amp = append_amp_freq(times1,gain1_dc)
	   freqs1_dc.append(freq)
	   amps1_dc.append(amp)
	   # Gain 10
	   freq, amp = append_amp_freq(times10,gain10_dc)
	   freqs10_dc.append(freq)
	   amps10_dc.append(amp)
	   # Gain 100
	   freq, amp = append_amp_freq(times1,gain1_dc)
	   freqs100_dc.append(freq)
	   amps100_dc.append(amp)

	# Plot amplitude fits
	plt.close()
	plt.plot(gains,np.abs(amps1),label='AMP1')
	plt.plot(gains[:7],np.abs(amps10[:7]),label='AMP10')
	plt.plot(gains[:5],np.abs(amps100[:5]),label='AMP100')
	plt.legend(loc='best')
	#plt.ylim([0.09,.11])
	plt.xlabel('Gain [dB]')
	plt.ylabel('Fitted amplitude [V]')
	#plt.show()
	plt.savefig(plotroot+'Amplitude_fit.png',dpi=300)

	# Plot normalized amplitude fits
	plt.close()
	plt.plot(gains,100.*np.abs(amps1)/gains_lin,label='AMP1')
	plt.plot(gains[:7],10.*np.abs(amps10[:7])/gains_lin[:7],label='AMP10')
	plt.plot(gains[:5],np.abs(amps100[:5])/gains_lin[:5],label='AMP100')
	plt.legend(loc='best')
	plt.xlabel('Gain [dB]')
	plt.ylabel('Normalized fitted amplitude')
	plt.savefig(plotroot+'Normalized_Amplitude_fit.png',dpi=300)
	#plt.ylim([0.09,.11])
	#plt.show()

	# Plot DC corrected amplitude fits
	plt.close()
	plt.plot(gains,np.abs(amps1_dc),label='AMP1')
	plt.plot(gains[:7],np.abs(amps10_dc[:7]),label='AMP10')
	plt.plot(gains[:5],np.abs(amps100_dc[:5]),label='AMP100')
	plt.legend(loc='best')
	#plt.ylim([0.09,.11])
	plt.xlabel('Gain [dB]')
	plt.ylabel('Fitted amplitude [V]')
	#plt.show()
	plt.savefig(plotroot+'Amplitude_fit_dc.png',dpi=300)

	# Plot normalized DC corrected amplitude fits
	plt.close()
	plt.plot(gains,100.*np.abs(amps1_dc)/gains_lin,label='AMP1')
	plt.plot(gains[:7],10.*np.abs(amps10_dc[:7])/gains_lin[:7],label='AMP10')
	plt.plot(gains[:5],np.abs(amps100_dc[:5])/gains_lin[:5],label='AMP100')
	plt.legend(loc='best')
	plt.xlabel('Gain [dB]')
	plt.ylabel('Normalized fitted amplitude')
	plt.savefig(plotroot+'Normalized_Amplitude_fit_dc.png',dpi=300)
	#plt.ylim([0.09,.11])
	#plt.show()

	# plot frequency fits
	plt.close()
	plt.plot(gains,freqs1,label='AMP1')
	plt.plot(gains[:7],freqs10[:7],label='AMP10')
	plt.plot(gains[:5],freqs100[:5],label='AMP100')
	plt.legend(loc='best')
	plt.ylim([0.099,.101])
	plt.xlabel('Gain [dB]')
	plt.ylabel('Fitted frequency [Hz]')
	#plt.show()
	plt.savefig(plotroot+'Frequency_fit.png',dpi=300)

	# 	# plot frequency fits, Dark corrected
	plt.close()
	plt.plot(gains,freqs1_dc,label='AMP1')
	plt.plot(gains[:7],freqs10_dc[:7],label='AMP10')
	plt.plot(gains[:5],freqs100_dc[:5],label='AMP100')
	plt.legend(loc='best')
	plt.ylim([0.099,.101])
	plt.xlabel('Gain [dB]')
	plt.ylabel('Fitted frequency [Hz]')
	#plt.show()
	plt.savefig(plotroot+'Frequency_fit_dc.png',dpi=300)


def Plot_percentile_data_30jan():
	print '***  Plot_percentile_data_30jan  ****'

	def get_sine_wave_data():
		times1, gain1     = mytools.load_data('30jan_ch1',140,147)
		times10, gain10   = mytools.load_data('30jan_ch1',150,157)
		times100, gain100 = mytools.load_data('30jan_ch1',160,167)
		return times1, gain1, times10, gain10, times100, gain100

	# get sine wave data
	times1, gain1, times10, gain10, times100, gain100 = get_sine_wave_data()
	# correct for dark current
	gain1_dc, gain10_dc, gain100_dc = correct_for_dc(gain1, gain10, gain100)

	frac1, frac10, frac100 = [], [], []
	frac1_dc, frac10_dc, frac100_dc = [], [], []
	for i in range(8):
	   # get mean value of 1st percentile and 99th percentile
	   # fraction of the two is a measure of the amplitude range; this sohuld be a factor of 10 (10dB)
	   pct1 = stats.scoreatpercentile(gain1[i][:25000],1)
	   pct99 = stats.scoreatpercentile(gain1[i][:25000],99)
	   frac = pct99/pct1
	   frac1.append(frac)
	   pct1 = stats.scoreatpercentile(gain10[i][:25000],1)
	   pct99 = stats.scoreatpercentile(gain10[i][:25000],99)
	   frac = pct99/pct1
	   frac10.append(frac)
	   pct1 = stats.scoreatpercentile(gain100[i][:25000],1)
	   pct99 = stats.scoreatpercentile(gain100[i][:25000],99)
	   frac = pct99/pct1
	   frac100.append(frac)
	   # dark corrected
	   pct1 = stats.scoreatpercentile(gain1_dc[i][:25000],1)
	   pct99 = stats.scoreatpercentile(gain1_dc[i][:25000],99)
	   frac = pct99/pct1
	   frac1_dc.append(frac)
	   pct1 = stats.scoreatpercentile(gain10_dc[i][:25000],1)
	   pct99 = stats.scoreatpercentile(gain10_dc[i][:25000],99)
	   frac = pct99/pct1
	   frac10_dc.append(frac)
	   pct1 = stats.scoreatpercentile(gain100_dc[i][:25000],1)
	   pct99 = stats.scoreatpercentile(gain100_dc[i][:25000],99)
	   frac = pct99/pct1
	   frac100_dc.append(frac)

	# Plot amlitude fraction based on 1st and 99th percentile 
	plt.close()
	plt.plot(gains, frac1,label='AMP1')
	plt.plot(gains[:7], frac10[:7],label='AMP10')
	plt.plot(gains[:5], frac100[:5],label='AMP100')
	plt.plot([0,70],[10,10],label='Requirement')
	plt.xlabel('detector gain [dB]')
	plt.ylabel('amplification amplitude / fraction')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig(plotroot+'amplitide_fraction.png',dpi=300)
	# DC corrected
	plt.close()
	plt.plot(gains, frac1_dc,label='AMP1')
	plt.plot(gains[:7], frac10_dc[:7],label='AMP10')
	plt.plot(gains[:5], frac100_dc[:5],label='AMP100')
	plt.plot([0,70],[10,10],label='Requirement')
	plt.xlabel('detector gain [dB]')
	plt.ylabel('amplification amplitude / fraction [dark corr.]')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig(plotroot+'amplitide_fraction_dc.png',dpi=300)

	# Plot time series
	for i in range(8):
	   plt.close()
	   plt.plot(times1[i][:5000], gain1[i][:5000], label='AMP1')
	   plt.plot(times10[i][:5000], gain10[i][:5000]/10, label='AMP10')
	   plt.plot(times100[i][:5000], gain100[i][1200:6200]/100, label='AMP100')
	 #  plt.show()
	   plt.xlabel('Time [s]')
	   plt.ylabel('Amplitude / amplification')
	   plt.legend(loc='best')
	   plt.savefig(plotroot+'amp_test_'+str(i)+'.png',dpi=300)

	   plt.close()
	   plt.plot(times1[i][:5000], gain1_dc[i][:5000], label='AMP1')
	   plt.plot(times10[i][:5000], gain10_dc[i][:5000]/10, label='AMP10')
	   plt.plot(times100[i][:5000], gain100_dc[i][1200:6200]/100, label='AMP100')
	 #  plt.show()
	   plt.xlabel('Time [s]')
	   plt.ylabel('Amplitude / amplification')
	   plt.legend(loc='best')
	   plt.savefig(plotroot+'amp_test_darkcorrect_'+str(i)+'.png',dpi=300)
	  


def Compare_EVOA_calibrations():
	print '*** Compare_EVOA_calibrations   ***'

	def EVOA_cal_from_handyscope_data():
		evoa_cal = mytools.load_data('1feb_ch1',22,42)
		sig = evoa_cal[1]
		means = []
		for i in range(21):
			mean = np.mean(sig[i])
			means.append(mean)
		return means

	# cal from handyscope data
	evoa_cal = EVOA_cal_from_handyscope_data()
	volt_in = np.linspace(0,4.75,num=20)
	# subtract dark current
	#evoa_cal = evoa_cal[:20] - 0.5*(evoa_cal[19]+evoa_cal[20])
	evoa_cal = evoa_cal[:20] - 0.5*(evoa_cal[19]+evoa_cal[19])
	evoa_cal = evoa_cal / np.max(evoa_cal)
	evoa_cal = 1 - evoa_cal

	# cal from reading display at EVOA
	v,a = mytools.get_EVOA_cal_results()

	plt.close()
	plt.plot(volt_in, evoa_cal,label="EVOA display")
	plt.plot(v,a, label="HandyScope reading")
	plt.xlabel("Voltage input")
	plt.ylabel("Attenutation")
	plt.legend()
	plt.savefig(plotroot+'EVOA_cal_comparison.png',dpi=300)
	#plt.show()

	plt.close()
	plt.plot(volt_in, evoa_cal-a[:20])
	plt.xlabel("Voltage input")
	plt.ylabel("Difference in attenutation")
	plt.savefig(plotroot+'EVOA_cal_comparison2.png',dpi=300)
	#plt.show()

def Plot_SNR_6feb():
	print '*** Plot_SNR_6feb  ***'

	means_0dB_1, stds_0dB_1 = mytools.get_means_stds('6feb_PM5_ch1', 36,39)
	means_0dB_2, stds_0dB_2 = mytools.get_means_stds('6feb_PM5_ch1', 41,44)
	means_40dB, stds_40dB = mytools.get_means_stds('6feb_PM5_ch1', 45,52)

	means_0dB = np.concatenate([means_0dB_1, means_0dB_2])
	stds_0dB = np.concatenate([stds_0dB_1, stds_0dB_2])

	dark_means, dark_stds = mytools.get_means_stds('6feb_PM2_ch1', 61,68)
	dark_means_0dB = dark_means[0]
	dark_stds_0dB = dark_stds[0]
	dark_means_40dB = dark_means[4]
	dark_stds_40dB = dark_stds[4]

	# Modelled signal level at detector in nW
	signal_at_detector = [4.204, 8.387, 16.74, 33.339, 138.5, 276.3, 551.3, 1100]

	# correct for dark current
	#means_0dB_dc = means_0dB-dark_means_0dB*0.99
	#means_40dB_dc = means_40dB-dark_means_40dB*0.99
	means_0dB_dc = means_0dB-dark_means_0dB
	means_40dB_dc = means_40dB-dark_means_40dB
	
	# Make plots
	plt.close()
#	plt.plot(signal_at_detector, means_0dB-means_0dB[0])
#	plt.plot(signal_at_detector, (means_40dB-means_40dB[0])/100.)
#	plt.plot(signal_at_detector, means_0dB_dc)
	plt.plot(signal_at_detector, means_40dB_dc/signal_at_detector)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

	# plot SNR
	plt.close()
	plt.plot(signal_at_detector, np.abs(means_0dB_dc/stds_0dB), '--' , label="0dB")
	plt.plot(signal_at_detector, np.abs(means_40dB_dc/stds_40dB), label="40dB")
	plt.plot([16.87,16.87], [1e-2,7e0],':',label='min ACQ')
	plt.plot([48.66,48.66], [1e-2,2e1],':',label='min RX')
	plt.plot([612.56,612.56], [1e-2,3e2],':', label='max RX')
	plt.xlabel('signal at detector [nW]')
	plt.ylabel('SNR')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.title('SNR @1562nm')
	plt.show()
	plt.savefig(plotroot+'SNR.png',dpi=300)
	


def Write_RAW_sinewave_measurements_toFile_6feb():
	print '***  Write_RAW_sinewave_measurements_toFile_6feb    ***'

	# Four measurement sets: 60 and 45 dB attenuation and 1 V/V and 10V/V amplification

	# Set 1: 60dB, 1VV
	time_60db_1VV_dark,   signal_60db_1VV_dark     = mytools.load_data('6feb_ch1',38,45 )
	time_60db_1VV_0hz,    signal_60db_1VV_0hz      = mytools.load_data('6feb_ch1',46,53)
	time_60db_1VV_1hz,    signal_60db_1VV_1hz      = mytools.load_data('6feb_ch1', 141,148)
	time_60db_1VV_3hz,    signal_60db_1VV_3hz      = mytools.load_data('6feb_ch1', 149,156)
	time_60db_1VV_6hz,    signal_60db_1VV_6hz      = mytools.load_data('6feb_ch1', 167,174)
	time_60db_1VV_10hz,   signal_60db_1VV_10hz     = mytools.load_data('6feb_ch1', 175,182)
	time_60db_1VV_30hz,   signal_60db_1VV_30hz     = mytools.load_data('6feb_ch1', 183,190)
	time_60db_1VV_60hz,   signal_60db_1VV_60hz     = mytools.load_data('6feb_ch1', 191,198)
	time_60db_1VV_120hz,  signal_60db_1VV_120hz    = mytools.load_data('6feb_ch1', 199,206)
	time_60db_1VV_250hz,  signal_60db_1VV_250hz    = mytools.load_data('6feb_ch1', 207,214)
	time_60db_1VV_500hz,  signal_60db_1VV_500hz    = mytools.load_data('6feb_ch1', 215,222)
	time_60db_1VV_1000hz, signal_60db_1VV_1000hz   = mytools.load_data('6feb_ch1', 223,230)
	time_60db_1VV_2000hz, signal_60db_1VV_2000hz   = mytools.load_data('6feb_ch1', 231,238)
	# Set 2: 45dB, 1VV
	time_45db_1VV_dark,   signal_45db_1VV_dark     = mytools.load_data('6feb_ch1', 239,246)
	time_45db_1VV_0hz,    signal_45db_1VV_0hz      = mytools.load_data('6feb_ch1', 247,254)
	time_45db_1VV_1hz,    signal_45db_1VV_1hz      = mytools.load_data('6feb_ch1', 255,262)
	time_45db_1VV_3hz,    signal_45db_1VV_3hz      = mytools.load_data('6feb_ch1', 263,270)
	time_45db_1VV_6hz,    signal_45db_1VV_6hz      = mytools.load_data('6feb_ch1', 271,278)
	time_45db_1VV_10hz,   signal_45db_1VV_10hz     = mytools.load_data('6feb_ch1', 279,286)
	time_45db_1VV_30hz,   signal_45db_1VV_30hz     = mytools.load_data('6feb_ch1', 287,294)
	time_45db_1VV_60hz,   signal_45db_1VV_60hz     = mytools.load_data('6feb_ch1', 295,302)
	time_45db_1VV_120hz,  signal_45db_1VV_120hz    = mytools.load_data('6feb_ch1', 303,310)
	time_45db_1VV_250hz,  signal_45db_1VV_250hz    = mytools.load_data('6feb_ch1', 311,318)
	time_45db_1VV_500hz,  signal_45db_1VV_500hz    = mytools.load_data('6feb_ch1', 319,326)
	time_45db_1VV_1000hz, signal_45db_1VV_1000hz   = mytools.load_data('6feb_ch1', 327,334)
	time_45db_1VV_2000hz, signal_45db_1VV_2000hz   = mytools.load_data('6feb_ch1', 335,342)
	# Set 3: 45dB, 10VV
	time_45db_10VV_dark,   signal_45db_10VV_dark   = mytools.load_data('6feb_PM_ch1',  16,23)
	time_45db_10VV_0hz,    signal_45db_10VV_0hz    = mytools.load_data('6feb_PM_ch1',  24,31)
	time_45db_10VV_1hz,    signal_45db_10VV_1hz    = mytools.load_data('6feb_PM_ch1',  32,39)
	time_45db_10VV_3hz,    signal_45db_10VV_3hz    = mytools.load_data('6feb_PM_ch1',  40,47)
	time_45db_10VV_6hz,    signal_45db_10VV_6hz    = mytools.load_data('6feb_PM_ch1',  48,55)
	time_45db_10VV_10hz,   signal_45db_10VV_10hz   = mytools.load_data('6feb_PM_ch1',  56,63)
	time_45db_10VV_30hz,   signal_45db_10VV_30hz   = mytools.load_data('6feb_PM2_ch1', 5,12)
	time_45db_10VV_60hz,   signal_45db_10VV_60hz   = mytools.load_data('6feb_PM2_ch1', 13,20)
	time_45db_10VV_100hz,  signal_45db_10VV_100hz  = mytools.load_data('6feb_PM2_ch1', 21,28)
	time_45db_10VV_250hz,  signal_45db_10VV_250hz  = mytools.load_data('6feb_PM2_ch1', 29,36)
	time_45db_10VV_500hz,  signal_45db_10VV_500hz  = mytools.load_data('6feb_PM2_ch1', 37,44)
	time_45db_10VV_1000hz, signal_45db_10VV_1000hz = mytools.load_data('6feb_PM2_ch1', 45,52)
	time_45db_10VV_2000hz, signal_45db_10VV_2000hz = mytools.load_data('6feb_PM2_ch1', 53,60)
	# Set 4: 60dB, 10VV
	time_60db_10VV_dark,   signal_60db_10VV_dark   = mytools.load_data('6feb_PM2_ch1', 61,68)
	time_60db_10VV_0hz,    signal_60db_10VV_0hz    = mytools.load_data('6feb_PM2_ch1', 69,76)
	time_60db_10VV_1hz,    signal_60db_10VV_1hz    = mytools.load_data('6feb_PM2_ch1', 77,84)
	time_60db_10VV_3hz,    signal_60db_10VV_3hz    = mytools.load_data('6feb_PM2_ch1', 85,92)
	time_60db_10VV_6hz,    signal_60db_10VV_6hz    = mytools.load_data('6feb_PM3_ch1', 3,11)
	time_60db_10VV_10hz,   signal_60db_10VV_10hz   = mytools.load_data('6feb_PM4_ch1', 6,13)
	time_60db_10VV_30hz,   signal_60db_10VV_30hz   = mytools.load_data('6feb_PM4_ch1', 22,29)
	time_60db_10VV_60hz,   signal_60db_10VV_60hz   = mytools.load_data('6feb_PM4_ch1', 30,37)
	time_60db_10VV_120hz,  signal_60db_10VV_120hz  = mytools.load_data('6feb_PM4_ch1', 38,45)
	time_60db_10VV_250hz,  signal_60db_10VV_250hz  = mytools.load_data('6feb_PM4_ch1', 46,53)
	time_60db_10VV_500hz,  signal_60db_10VV_500hz  = mytools.load_data('6feb_PM4_ch1', 54,61)
	time_60db_10VV_1000hz, signal_60db_10VV_1000hz = mytools.load_data('6feb_PM5_ch1', 16,23)
	time_60db_10VV_2000hz, signal_60db_10VV_2000hz = mytools.load_data('6feb_PM5_ch1', 24,31)
	
	time_60db_1VV    = {'dark': time_60db_1VV_dark,    '0Hz': time_60db_1VV_0hz,    '1Hz': time_60db_1VV_1hz,    '3Hz': time_60db_1VV_3hz,    '6Hz': time_60db_1VV_6hz,    '10Hz': time_60db_1VV_10hz,    '30Hz': time_60db_1VV_30hz,    '60Hz': time_60db_1VV_60hz,    '120Hz': time_60db_1VV_120hz,    '250Hz': time_60db_1VV_250hz,    '500Hz': time_60db_1VV_500hz,    '1000Hz': time_60db_1VV_1000hz,    '2000Hz': time_60db_1VV_2000hz}
	signal_60db_1VV  = {'dark': signal_60db_1VV_dark,  '0Hz': signal_60db_1VV_0hz,  '1Hz': signal_60db_1VV_1hz,  '3Hz': signal_60db_1VV_3hz,  '6Hz': signal_60db_1VV_6hz,  '10Hz': signal_60db_1VV_10hz,  '30Hz': signal_60db_1VV_30hz,  '60Hz': signal_60db_1VV_60hz,  '120Hz': signal_60db_1VV_120hz,  '250Hz': signal_60db_1VV_250hz,  '500Hz': signal_60db_1VV_500hz,  '1000Hz': signal_60db_1VV_1000hz,  '2000Hz': signal_60db_1VV_2000hz}
	time_45db_1VV    = {'dark': time_45db_1VV_dark,    '0Hz': time_45db_1VV_0hz,    '1Hz': time_45db_1VV_1hz,    '3Hz': time_45db_1VV_3hz,    '6Hz': time_45db_1VV_6hz,    '10Hz': time_45db_1VV_10hz,    '30Hz': time_45db_1VV_30hz,    '60Hz': time_45db_1VV_60hz,    '120Hz': time_45db_1VV_120hz,    '250Hz': time_45db_1VV_250hz,    '500Hz': time_45db_1VV_500hz,    '1000Hz': time_45db_1VV_1000hz,    '2000Hz': time_45db_1VV_2000hz}
	signal_45db_1VV  = {'dark': signal_45db_1VV_dark,  '0Hz': signal_45db_1VV_0hz,  '1Hz': signal_45db_1VV_1hz,  '3Hz': signal_45db_1VV_3hz,  '6Hz': signal_45db_1VV_6hz,  '10Hz': signal_45db_1VV_10hz,  '30Hz': signal_45db_1VV_30hz,  '60Hz': signal_45db_1VV_60hz,  '120Hz': signal_45db_1VV_120hz,  '250Hz': signal_45db_1VV_250hz,  '500Hz': signal_45db_1VV_500hz,  '1000Hz': signal_45db_1VV_1000hz,  '2000Hz': signal_45db_1VV_2000hz}
	time_45db_10VV   = {'dark': time_45db_10VV_dark,   '0Hz': time_45db_10VV_0hz,   '1Hz': time_45db_10VV_1hz,   '3Hz': time_45db_10VV_3hz,   '6Hz': time_45db_10VV_6hz,   '10Hz': time_45db_10VV_10hz,   '30Hz': time_45db_10VV_30hz,   '60Hz': time_45db_10VV_60hz,   '100Hz': time_45db_10VV_100hz,   '250Hz': time_45db_10VV_250hz,   '500Hz': time_45db_10VV_500hz,   '1000Hz': time_45db_10VV_1000hz,   '2000Hz': time_45db_10VV_2000hz}
	signal_45db_10VV = {'dark': signal_45db_10VV_dark, '0Hz': signal_45db_10VV_0hz, '1Hz': signal_45db_10VV_1hz, '3Hz': signal_45db_10VV_3hz, '6Hz': signal_45db_10VV_6hz, '10Hz': signal_45db_10VV_10hz, '30Hz': signal_45db_10VV_30hz, '60Hz': signal_45db_10VV_60hz, '100Hz': signal_45db_10VV_100hz, '250Hz': signal_45db_10VV_250hz, '500Hz': signal_45db_10VV_500hz, '1000Hz': signal_45db_10VV_1000hz, '2000Hz': signal_45db_10VV_2000hz}
	time_60db_10VV   = {'dark': time_60db_10VV_dark,   '0Hz': time_60db_10VV_0hz,   '1Hz': time_60db_10VV_1hz,   '3Hz': time_60db_10VV_3hz,   '6Hz': time_60db_10VV_6hz,   '10Hz': time_60db_10VV_10hz,   '30Hz': time_60db_10VV_30hz,   '60Hz': time_60db_10VV_60hz,   '120Hz': time_60db_10VV_120hz,   '250Hz': time_60db_10VV_250hz,   '500Hz': time_60db_10VV_500hz,   '1000Hz': time_60db_10VV_1000hz,   '2000Hz': time_60db_10VV_2000hz}
	signal_60db_10VV = {'dark': signal_60db_10VV_dark, '0Hz': signal_60db_10VV_0hz, '1Hz': signal_60db_10VV_1hz, '3Hz': signal_60db_10VV_3hz, '6Hz': signal_60db_10VV_6hz, '10Hz': signal_60db_10VV_10hz, '30Hz': signal_60db_10VV_30hz, '60Hz': signal_60db_10VV_60hz, '120Hz': signal_60db_10VV_120hz, '250Hz': signal_60db_10VV_250hz, '500Hz': signal_60db_10VV_500hz, '1000Hz': signal_60db_10VV_1000hz, '2000Hz': signal_60db_10VV_2000hz}

	# write to file
	# first, turn it into a dictionary
	all_sinewave_data = {}
	all_time_data = {}
	all_signal_data = {}
	all_time_data['60db_1VV']  = time_60db_1VV
	all_time_data['45db_1VV']  = time_45db_1VV
	all_time_data['45db_10VV'] = time_45db_10VV
	all_time_data['60db_10VV'] = time_60db_10VV
	all_signal_data['60db_1VV']   = signal_60db_1VV
	all_signal_data['45db_1VV']   = signal_45db_1VV
	all_signal_data['45db_10VV']  = signal_45db_10VV
	all_signal_data['60db_10VV']  = signal_60db_10VV
	all_sinewave_data['time'] = all_time_data
	all_sinewave_data['signal'] = all_signal_data

	# save to pkl file
	with open('all_raw_sinewave_data_6feb.pkl','wb') as f:
		pickle.dump(all_sinewave_data, f)
	


def Plot_sinewaves_and_fits():
	print '*** Plot_sinewaves_and_fits  ***'

	filename = 'all_raw_sinewave_data_6feb.pkl'
	with open(filename,'rb') as f:
	   print 'Reading ', filename
	   all_sinewave_data = pickle.load(f)
	time = all_sinewave_data['time']
	signal = all_sinewave_data['signal']

	def test_attn():
		cycles = 2. # how many sine cycles
		resolution = 1500. # how many datapoints to generate

		freq_scale = 10.
		length = np.pi * 2 * cycles
		my_wave = np.sin(np.arange(0, length, freq_scale *length / resolution))	
		amp = 2.0
		offset = 2.25
		my_wave = my_wave * amp + offset
		my_volt  = mytools.get_EVOA_attenuation(my_wave)

		return my_volt

		#plt.close()	
		#plt.plot(my_volt)
		#plt.show()

	my_volt = test_attn()
	
	time_60db_1VV    = time['60db_1VV']    
	time_45db_1VV    = time['45db_1VV']    
	time_45db_10VV   = time['45db_10VV']   
	time_60db_10VV   = time['60db_10VV']  
	signal_60db_1VV  = signal['60db_1VV'] 
	signal_45db_1VV  = signal['45db_1VV']   
	signal_45db_10VV = signal['45db_10VV']  
	signal_60db_10VV = signal['60db_10VV'] 

	plt.close()

	#plt.plot(time['60db_10VV']['dark'][0][:], signal['60db_10VV']['dark'][0][:])
	#plt.plot(time['60db_10VV']['0Hz'][0][:], signal['60db_10VV']['0Hz'][0][:])
	#plt.plot(time['60db_10VV']['1Hz'][0][:], signal['60db_10VV']['1Hz'][0][:])

	series = '45db_10VV'
	i = 5
	nt = 30000
	plt.plot(time[series]['dark'][i][:nt], signal[series]['dark'][i][:nt],label='dark')
	plt.plot(time[series]['0Hz'][i][:nt],  signal[series]['0Hz'][i][:nt],label='no attenuation')
	plt.plot(time[series]['1Hz'][i][:nt],  signal[series]['1Hz'][i][:nt],label='1Hz attenuation')
	#plt.plot(time[series]['3Hz'][i][:],  signal[series]['3Hz'][i][:])
	#plt.plot(time[series]['6Hz'][i][:],  signal[series]['6Hz'][i][:])
	plt.plot(np.arange(len(my_volt))/75.-0.151, 0.1272+1.428*my_volt,label='Predicted attenutation')
	plt.xlabel('time [s]')
	plt.ylabel('signal [V]')
	plt.ylim([0,2.25])
	plt.legend(loc='upper right')
	plt.show()
	plt.savefig(plotroot+'SineWave_replication.png',dpi=300)

	plt.close()
	i = 5
	series = '60db_1VV'
	plt.plot(time[series]['1Hz'][i][:nt],  signal[series]['1Hz'][i][:nt] * np.power(10, 1.5) * 10,label=series)
	series = '60db_10VV'
	plt.plot(time[series]['1Hz'][i][:nt],  signal[series]['1Hz'][i][:nt] * np.power(10,1.5),label=series)
	series = '45db_1VV'
	plt.plot(time[series]['1Hz'][i][:nt],  signal[series]['1Hz'][i][:nt] * 10,label=series)
	series = '45db_10VV'
	plt.plot(time[series]['1Hz'][i][:nt],  signal[series]['1Hz'][i][:nt],label=series)

	plt.xlabel('time [s]')
	plt.ylabel('signal [V]')
#	plt.ylim([0,2.25])
	plt.legend(loc='upper right')
	plt.show()
	plt.savefig(plotroot+'SineWave_samples.png',dpi=300)


def Write_sinewave_measurements_toFile_6feb():
	print '***   Write_sinewave_measurements_toFile_6feb  ***'

	# Four measurement sets: 60 and 45 dB attenuation and 1 V/V and 10V/V amplification
	# Set 1: 60dB, 1VV
	time_60db_1VV_dark,   signal_60db_1VV_dark     = mytools.load_data('6feb_ch1',38,45 )
	time_60db_1VV_0hz,    signal_60db_1VV_0hz      = mytools.load_data('6feb_ch1',46,53)
	time_60db_1VV_1hz,    signal_60db_1VV_1hz      = mytools.load_data('6feb_ch1', 141,148)
	time_60db_1VV_3hz,    signal_60db_1VV_3hz      = mytools.load_data('6feb_ch1', 149,156)
	time_60db_1VV_6hz,    signal_60db_1VV_6hz      = mytools.load_data('6feb_ch1', 167,174)
	time_60db_1VV_10hz,   signal_60db_1VV_10hz     = mytools.load_data('6feb_ch1', 175,182)
	time_60db_1VV_30hz,   signal_60db_1VV_30hz     = mytools.load_data('6feb_ch1', 183,190)
	time_60db_1VV_60hz,   signal_60db_1VV_60hz     = mytools.load_data('6feb_ch1', 191,198)
	time_60db_1VV_120hz,  signal_60db_1VV_120hz    = mytools.load_data('6feb_ch1', 199,206)
	time_60db_1VV_250hz,  signal_60db_1VV_250hz    = mytools.load_data('6feb_ch1', 207,214)
	time_60db_1VV_500hz,  signal_60db_1VV_500hz    = mytools.load_data('6feb_ch1', 215,222)
	time_60db_1VV_1000hz, signal_60db_1VV_1000hz   = mytools.load_data('6feb_ch1', 223,230)
	time_60db_1VV_2000hz, signal_60db_1VV_2000hz   = mytools.load_data('6feb_ch1', 231,238)
	# Set 2: 45dB, 1VV
	time_45db_1VV_dark,   signal_45db_1VV_dark     = mytools.load_data('6feb_ch1', 239,246)
	time_45db_1VV_0hz,    signal_45db_1VV_0hz      = mytools.load_data('6feb_ch1', 247,254)
	time_45db_1VV_1hz,    signal_45db_1VV_1hz      = mytools.load_data('6feb_ch1', 255,262)
	time_45db_1VV_3hz,    signal_45db_1VV_3hz      = mytools.load_data('6feb_ch1', 263,270)
	time_45db_1VV_6hz,    signal_45db_1VV_6hz      = mytools.load_data('6feb_ch1', 271,278)
	time_45db_1VV_10hz,   signal_45db_1VV_10hz     = mytools.load_data('6feb_ch1', 279,286)
	time_45db_1VV_30hz,   signal_45db_1VV_30hz     = mytools.load_data('6feb_ch1', 287,294)
	time_45db_1VV_60hz,   signal_45db_1VV_60hz     = mytools.load_data('6feb_ch1', 295,302)
	time_45db_1VV_120hz,  signal_45db_1VV_120hz    = mytools.load_data('6feb_ch1', 303,310)
	time_45db_1VV_250hz,  signal_45db_1VV_250hz    = mytools.load_data('6feb_ch1', 311,318)
	time_45db_1VV_500hz,  signal_45db_1VV_500hz    = mytools.load_data('6feb_ch1', 319,326)
	time_45db_1VV_1000hz, signal_45db_1VV_1000hz   = mytools.load_data('6feb_ch1', 327,334)
	time_45db_1VV_2000hz, signal_45db_1VV_2000hz   = mytools.load_data('6feb_ch1', 335,342)
	# Set 3: 45dB, 10VV
	time_45db_10VV_dark,   signal_45db_10VV_dark   = mytools.load_data('6feb_PM_ch1',  16,23)
	time_45db_10VV_0hz,    signal_45db_10VV_0hz    = mytools.load_data('6feb_PM_ch1',  24,31)
	time_45db_10VV_1hz,    signal_45db_10VV_1hz    = mytools.load_data('6feb_PM_ch1',  32,39)
	time_45db_10VV_3hz,    signal_45db_10VV_3hz    = mytools.load_data('6feb_PM_ch1',  40,47)
	time_45db_10VV_6hz,    signal_45db_10VV_6hz    = mytools.load_data('6feb_PM_ch1',  48,55)
	time_45db_10VV_10hz,   signal_45db_10VV_10hz   = mytools.load_data('6feb_PM_ch1',  56,63)
	time_45db_10VV_30hz,   signal_45db_10VV_30hz   = mytools.load_data('6feb_PM2_ch1',  5,12)
	time_45db_10VV_60hz,   signal_45db_10VV_60hz   = mytools.load_data('6feb_PM2_ch1', 13,20)
	time_45db_10VV_100hz,  signal_45db_10VV_100hz  = mytools.load_data('6feb_PM2_ch1', 21,28)
	time_45db_10VV_250hz,  signal_45db_10VV_250hz  = mytools.load_data('6feb_PM2_ch1', 29,36)
	time_45db_10VV_500hz,  signal_45db_10VV_500hz  = mytools.load_data('6feb_PM2_ch1', 37,44)
	time_45db_10VV_1000hz, signal_45db_10VV_1000hz = mytools.load_data('6feb_PM2_ch1', 45,52)
	time_45db_10VV_2000hz, signal_45db_10VV_2000hz = mytools.load_data('6feb_PM2_ch1', 53,60)
	# Set 4: 60dB, 10VV
	time_60db_10VV_dark,   signal_60db_10VV_dark   = mytools.load_data('6feb_PM2_ch1', 61,68)
	time_60db_10VV_0hz,    signal_60db_10VV_0hz    = mytools.load_data('6feb_PM2_ch1', 69,76)
	time_60db_10VV_1hz,    signal_60db_10VV_1hz    = mytools.load_data('6feb_PM2_ch1', 77,84)
	time_60db_10VV_3hz,    signal_60db_10VV_3hz    = mytools.load_data('6feb_PM2_ch1', 85,92)
	time_60db_10VV_6hz,    signal_60db_10VV_6hz    = mytools.load_data('6feb_PM3_ch1',  3,11)
	time_60db_10VV_10hz,   signal_60db_10VV_10hz   = mytools.load_data('6feb_PM4_ch1',  6,13)
	time_60db_10VV_30hz,   signal_60db_10VV_30hz   = mytools.load_data('6feb_PM4_ch1', 22,29)
	time_60db_10VV_60hz,   signal_60db_10VV_60hz   = mytools.load_data('6feb_PM4_ch1', 30,37)
	time_60db_10VV_120hz,  signal_60db_10VV_120hz  = mytools.load_data('6feb_PM4_ch1', 38,45)
	time_60db_10VV_250hz,  signal_60db_10VV_250hz  = mytools.load_data('6feb_PM4_ch1', 46,53)
	time_60db_10VV_500hz,  signal_60db_10VV_500hz  = mytools.load_data('6feb_PM4_ch1', 54,61)
	time_60db_10VV_1000hz, signal_60db_10VV_1000hz = mytools.load_data('6feb_PM5_ch1', 16,23)
	time_60db_10VV_2000hz, signal_60db_10VV_2000hz = mytools.load_data('6feb_PM5_ch1', 24,31)

	# calculate dark current signals:
	dark_60db_1VV = []
	dark_45db_1VV = []
	dark_45db_10VV = []
	dark_60db_10VV = []
	for i in range(8):
		dark = np.mean(signal_60db_1VV_dark[i][:])
		dark_60db_1VV.append(dark)
		dark = np.mean(signal_45db_1VV_dark[i][:])
		dark_45db_1VV.append(dark)
		dark = np.mean(signal_45db_10VV_dark[i][:])
		dark_45db_10VV.append(dark)
		dark = np.mean(signal_60db_10VV_dark[i][:])
		dark_60db_10VV.append(dark)
	
	def append_amp_freq(time,sig,dark):
	   time = np.asarray(time)
	   sig = np.asarray(sig)
	   all_amp = []
	   all_freq = []
	   for i in range(8):
	      t = time[i][:25000]
	      g = sig[i][:25000]-dark[i]
	      res = mytools.fit_sin(t,g)
	      freq = res['period']
	      amp = res['amp']
	      all_freq.append(freq)
	      all_amp.append(amp)
	   all_freq = np.abs(np.asarray(all_freq))
	   all_amp = np.abs(np.asarray(all_amp))
	   return all_amp, all_freq

	freq_60db_1VV_1hz, freq_60db_1VV_3hz, freq_60db_1VV_6hz, freq_60db_1VV_10hz, freq_60db_1VV_30hz, freq_60db_1VV_60hz, freq_60db_1VV_120hz, freq_60db_1VV_250hz, freq_60db_1VV_500hz, freq_60db_1VV_1000hz, freq_60db_1VV_2000hz = [], [], [], [], [], [], [], [], [], [], []
	amp_60db_1VV_1hz, amp_60db_1VV_3hz, amp_60db_1VV_6hz, amp_60db_1VV_10hz, amp_60db_1VV_30hz, amp_60db_1VV_60hz, amp_60db_1VV_120hz, amp_60db_1VV_250hz, amp_60db_1VV_500hz, amp_60db_1VV_1000hz, amp_60db_1VV_2000hz = [], [], [], [], [], [], [], [], [], [], []

	freq_45db_1VV_1hz, freq_45db_1VV_3hz, freq_45db_1VV_6hz, freq_45db_1VV_10hz, freq_45db_1VV_30hz, freq_45db_1VV_60hz, freq_45db_1VV_120hz, freq_45db_1VV_250hz, freq_45db_1VV_500hz, freq_45db_1VV_1000hz, freq_45db_1VV_2000hz = [], [], [], [], [], [], [], [], [], [], []
	amp_45db_1VV_1hz, amp_45db_1VV_3hz, amp_45db_1VV_6hz, amp_45db_1VV_10hz, amp_45db_1VV_30hz, amp_45db_1VV_60hz, amp_45db_1VV_120hz, amp_45db_1VV_250hz, amp_45db_1VV_500hz, amp_45db_1VV_1000hz, amp_45db_1VV_2000hz = [], [], [], [], [], [], [], [], [], [], []

	freq_45db_10VV_1hz, freq_45db_10VV_3hz, freq_45db_10VV_6hz, freq_45db_10VV_10hz, freq_45db_10VV_30hz, freq_45db_10VV_60hz, freq_45db_10VV_100hz, freq_45db_10VV_250hz, freq_45db_10VV_500hz, freq_45db_10VV_1000hz, freq_45db_10VV_2000hz = [], [], [], [], [], [], [], [], [], [], []
	amp_45db_10VV_1hz, amp_45db_10VV_3hz, amp_45db_10VV_6hz, amp_45db_10VV_10hz, amp_45db_10VV_30hz, amp_45db_10VV_60hz, amp_45db_10VV_100hz, amp_45db_10VV_250hz, amp_45db_10VV_500hz, amp_45db_10VV_1000hz, amp_45db_10VV_2000hz = [], [], [], [], [], [], [], [], [], [], []

	freq_60db_10VV_1hz, freq_60db_10VV_3hz, freq_60db_10VV_6hz, freq_60db_10VV_10hz, freq_60db_10VV_30hz, freq_60db_10VV_60hz, freq_60db_10VV_120hz, freq_60db_10VV_250hz, freq_60db_10VV_500hz, freq_60db_10VV_1000hz, freq_60db_10VV_2000hz = [], [], [], [], [], [], [], [], [], [], []
	amp_60db_10VV_1hz, amp_60db_10VV_3hz, amp_60db_10VV_6hz, amp_60db_10VV_10hz, amp_60db_10VV_30hz, amp_60db_10VV_60hz, amp_60db_10VV_120hz, amp_60db_10VV_250hz, amp_60db_10VV_500hz, amp_60db_10VV_1000hz, amp_60db_10VV_2000hz = [], [], [], [], [], [], [], [], [], [], []

	# Set 1: 60dB, 1VV
	amp, freq = append_amp_freq(time_60db_1VV_1hz,   signal_60db_1VV_1hz, dark_60db_1VV)
	freq_60db_1VV_1hz.append(freq)
	amp_60db_1VV_1hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_3hz,   signal_60db_1VV_3hz, dark_60db_1VV)
	freq_60db_1VV_3hz.append(freq)
	amp_60db_1VV_3hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_6hz,   signal_60db_1VV_6hz, dark_60db_1VV)
	freq_60db_1VV_6hz.append(freq)
	amp_60db_1VV_6hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_10hz,   signal_60db_1VV_10hz, dark_60db_1VV)
	freq_60db_1VV_10hz.append(freq)
	amp_60db_1VV_10hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_30hz,   signal_60db_1VV_30hz, dark_60db_1VV)
	freq_60db_1VV_30hz.append(freq)
	amp_60db_1VV_30hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_60hz,   signal_60db_1VV_60hz, dark_60db_1VV)
	freq_60db_1VV_60hz.append(freq)
	amp_60db_1VV_60hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_120hz,   signal_60db_1VV_120hz, dark_60db_1VV)
	freq_60db_1VV_120hz.append(freq)
	amp_60db_1VV_120hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_250hz,   signal_60db_1VV_250hz, dark_60db_1VV)
	freq_60db_1VV_250hz.append(freq)
	amp_60db_1VV_250hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_500hz,   signal_60db_1VV_500hz, dark_60db_1VV)
	freq_60db_1VV_500hz.append(freq)
	amp_60db_1VV_500hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_1000hz,   signal_60db_1VV_1000hz, dark_60db_1VV)
	freq_60db_1VV_1000hz.append(freq)
	amp_60db_1VV_1000hz.append(amp)
	amp, freq = append_amp_freq(time_60db_1VV_2000hz,   signal_60db_1VV_2000hz, dark_60db_1VV)
	freq_60db_1VV_2000hz.append(freq)
	amp_60db_1VV_2000hz.append(amp)
	# Set 2: 45dB, 1VV
	amp, freq = append_amp_freq(time_45db_1VV_1hz,   signal_45db_1VV_1hz, dark_45db_1VV)
	freq_45db_1VV_1hz.append(freq)
	amp_45db_1VV_1hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_3hz,   signal_45db_1VV_3hz, dark_45db_1VV)
	freq_45db_1VV_3hz.append(freq)
	amp_45db_1VV_3hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_6hz,   signal_45db_1VV_6hz, dark_45db_1VV)
	freq_45db_1VV_6hz.append(freq)
	amp_45db_1VV_6hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_10hz,   signal_45db_1VV_10hz, dark_45db_1VV)
	freq_45db_1VV_10hz.append(freq)
	amp_45db_1VV_10hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_30hz,   signal_45db_1VV_30hz, dark_45db_1VV)
	freq_45db_1VV_30hz.append(freq)
	amp_45db_1VV_30hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_60hz,   signal_45db_1VV_60hz, dark_45db_1VV)
	freq_45db_1VV_60hz.append(freq)
	amp_45db_1VV_60hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_120hz,   signal_45db_1VV_120hz, dark_45db_1VV)
	freq_45db_1VV_120hz.append(freq)
	amp_45db_1VV_120hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_250hz,   signal_45db_1VV_250hz, dark_45db_1VV)
	freq_45db_1VV_250hz.append(freq)
	amp_45db_1VV_250hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_500hz,   signal_45db_1VV_500hz, dark_45db_1VV)
	freq_45db_1VV_500hz.append(freq)
	amp_45db_1VV_500hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_1000hz,   signal_45db_1VV_1000hz, dark_45db_1VV)
	freq_45db_1VV_1000hz.append(freq)
	amp_45db_1VV_1000hz.append(amp)
	amp, freq = append_amp_freq(time_45db_1VV_2000hz,   signal_45db_1VV_2000hz, dark_45db_1VV)
	freq_45db_1VV_2000hz.append(freq)
	amp_45db_1VV_2000hz.append(amp)
	# Set 3: 45dB, 10VV
	amp, freq = append_amp_freq(time_45db_10VV_1hz,   signal_45db_10VV_1hz, dark_45db_10VV)
	freq_45db_10VV_1hz.append(freq)
	amp_45db_10VV_1hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_3hz,   signal_45db_10VV_3hz, dark_45db_10VV)
	freq_45db_10VV_3hz.append(freq)
	amp_45db_10VV_3hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_6hz,   signal_45db_10VV_6hz, dark_45db_10VV)
	freq_45db_10VV_6hz.append(freq)
	amp_45db_10VV_6hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_10hz,   signal_45db_10VV_10hz, dark_45db_10VV)
	freq_45db_10VV_10hz.append(freq)
	amp_45db_10VV_10hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_30hz,   signal_45db_10VV_30hz, dark_45db_10VV)
	freq_45db_10VV_30hz.append(freq)
	amp_45db_10VV_30hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_60hz,   signal_45db_10VV_60hz, dark_45db_10VV)
	freq_45db_10VV_60hz.append(freq)
	amp_45db_10VV_60hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_100hz,   signal_45db_10VV_100hz, dark_45db_10VV)
	freq_45db_10VV_100hz.append(freq)
	amp_45db_10VV_100hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_250hz,   signal_45db_10VV_250hz, dark_45db_10VV)
	freq_45db_10VV_250hz.append(freq)
	amp_45db_10VV_250hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_500hz,   signal_45db_10VV_500hz, dark_45db_10VV)
	freq_45db_10VV_500hz.append(freq)
	amp_45db_10VV_500hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_1000hz,   signal_45db_10VV_1000hz, dark_45db_10VV)
	freq_45db_10VV_1000hz.append(freq)
	amp_45db_10VV_1000hz.append(amp)
	amp, freq = append_amp_freq(time_45db_10VV_2000hz,   signal_45db_10VV_2000hz, dark_45db_10VV)
	freq_45db_10VV_2000hz.append(freq)
	amp_45db_10VV_2000hz.append(amp)
	# Set 4: 60dB, 10VV
	amp, freq = append_amp_freq(time_60db_10VV_1hz,   signal_60db_10VV_1hz, dark_60db_10VV)
	freq_60db_10VV_1hz.append(freq)
	amp_60db_10VV_1hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_3hz,   signal_60db_10VV_3hz, dark_60db_10VV)
	freq_60db_10VV_3hz.append(freq)
	amp_60db_10VV_3hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_6hz,   signal_60db_10VV_6hz, dark_60db_10VV)
	freq_60db_10VV_6hz.append(freq)
	amp_60db_10VV_6hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_10hz,   signal_60db_10VV_10hz, dark_60db_10VV)
	freq_60db_10VV_10hz.append(freq)
	amp_60db_10VV_10hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_30hz,   signal_60db_10VV_30hz, dark_60db_10VV)
	freq_60db_10VV_30hz.append(freq)
	amp_60db_10VV_30hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_60hz,   signal_60db_10VV_60hz, dark_60db_10VV)
	freq_60db_10VV_60hz.append(freq)
	amp_60db_10VV_60hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_120hz,   signal_60db_10VV_120hz, dark_60db_10VV)
	freq_60db_10VV_120hz.append(freq)
	amp_60db_10VV_120hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_250hz,   signal_60db_10VV_250hz, dark_60db_10VV)
	freq_60db_10VV_250hz.append(freq)
	amp_60db_10VV_250hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_500hz,   signal_60db_10VV_500hz, dark_60db_10VV)
	freq_60db_10VV_500hz.append(freq)
	amp_60db_10VV_500hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_1000hz,   signal_60db_10VV_1000hz, dark_60db_10VV)
	freq_60db_10VV_1000hz.append(freq)
	amp_60db_10VV_1000hz.append(amp)
	amp, freq = append_amp_freq(time_60db_10VV_2000hz,   signal_60db_10VV_2000hz, dark_60db_10VV)
	freq_60db_10VV_2000hz.append(freq)
	amp_60db_10VV_2000hz.append(amp)
	
	freq_60db_1VV  = [freq_60db_1VV_1hz, freq_60db_1VV_3hz, freq_60db_1VV_6hz, freq_60db_1VV_10hz, freq_60db_1VV_30hz, freq_60db_1VV_60hz, freq_60db_1VV_120hz, freq_60db_1VV_250hz, freq_60db_1VV_500hz, freq_60db_1VV_1000hz, freq_60db_1VV_2000hz]
	amp_60db_1VV   = [amp_60db_1VV_1hz, amp_60db_1VV_3hz, amp_60db_1VV_6hz, amp_60db_1VV_10hz, amp_60db_1VV_30hz, amp_60db_1VV_60hz, amp_60db_1VV_120hz, amp_60db_1VV_250hz, amp_60db_1VV_500hz, amp_60db_1VV_1000hz, amp_60db_1VV_2000hz]
	freq_45db_1VV  = [freq_45db_1VV_1hz, freq_45db_1VV_3hz, freq_45db_1VV_6hz, freq_45db_1VV_10hz, freq_45db_1VV_30hz, freq_45db_1VV_60hz, freq_45db_1VV_120hz, freq_45db_1VV_250hz, freq_45db_1VV_500hz, freq_45db_1VV_1000hz, freq_45db_1VV_2000hz]
	amp_45db_1VV   = [amp_45db_1VV_1hz, amp_45db_1VV_3hz, amp_45db_1VV_6hz, amp_45db_1VV_10hz, amp_45db_1VV_30hz, amp_45db_1VV_60hz, amp_45db_1VV_120hz, amp_45db_1VV_250hz, amp_45db_1VV_500hz, amp_45db_1VV_1000hz, amp_45db_1VV_2000hz]
	freq_45db_10VV = [freq_45db_10VV_1hz, freq_45db_10VV_3hz, freq_45db_10VV_6hz, freq_45db_10VV_10hz, freq_45db_10VV_30hz, freq_45db_10VV_60hz, freq_45db_10VV_100hz, freq_45db_10VV_250hz, freq_45db_10VV_500hz, freq_45db_10VV_1000hz, freq_45db_10VV_2000hz]
	amp_45db_10VV  = [amp_45db_10VV_1hz, amp_45db_10VV_3hz, amp_45db_10VV_6hz, amp_45db_10VV_10hz, amp_45db_10VV_30hz, amp_45db_10VV_60hz, amp_45db_10VV_100hz, amp_45db_10VV_250hz, amp_45db_10VV_500hz, amp_45db_10VV_1000hz, amp_45db_10VV_2000hz]
	freq_60db_10VV = [freq_60db_10VV_1hz, freq_60db_10VV_3hz, freq_60db_10VV_6hz, freq_60db_10VV_10hz, freq_60db_10VV_30hz, freq_60db_10VV_60hz, freq_60db_10VV_120hz, freq_60db_10VV_250hz, freq_60db_10VV_500hz, freq_60db_10VV_1000hz, freq_60db_10VV_2000hz]
	amp_60db_10VV  = [amp_60db_10VV_1hz, amp_60db_10VV_3hz, amp_60db_10VV_6hz, amp_60db_10VV_10hz, amp_60db_10VV_30hz, amp_60db_10VV_60hz, amp_60db_10VV_120hz, amp_60db_10VV_250hz, amp_60db_10VV_500hz, amp_60db_10VV_1000hz, amp_60db_10VV_2000hz]
	
	
	freq_60db_1VV = np.asarray(freq_60db_1VV)
	amp_60db_1VV = np.asarray(amp_60db_1VV)
	freq_45db_1VV = np.asarray(freq_45db_1VV)
	amp_45db_1VV = np.asarray(amp_45db_1VV)
	freq_45db_10VV = np.asarray(freq_45db_10VV)
	amp_45db_10VV = np.asarray(amp_45db_10VV)
	freq_60db_10VV = np.asarray(freq_60db_10VV)
	amp_60db_10VV = np.asarray(amp_60db_10VV)
	
	freq_all = [freq_60db_1VV, freq_45db_1VV, freq_45db_10VV, freq_60db_10VV]
	amp_all = [amp_60db_1VV, amp_45db_1VV, amp_45db_10VV, amp_60db_10VV]
	
	
	# write to file
	# first, turn it into a dictionary
	all_sinewave_data = {}
	all_freq_data = {}
	all_amp_data = {}
	all_freq_data['60db_1VV']  = freq_60db_1VV
	all_freq_data['45db_1VV']  = freq_45db_1VV
	all_freq_data['45db_10VV'] = freq_45db_10VV
	all_freq_data['60db_10VV'] = freq_60db_10VV
	all_amp_data['60db_1VV']   = amp_60db_1VV
	all_amp_data['45db_1VV']   = amp_45db_1VV
	all_amp_data['45db_10VV']  = amp_45db_10VV
	all_amp_data['60db_10VV']  = amp_60db_10VV
	all_sinewave_data['frequency'] = all_freq_data
	all_sinewave_data['amplitude'] = all_amp_data
	# save to pkl file
	with open('all_sinewave_data_6feb.pkl','wb') as f:
		pickle.dump(all_sinewave_data, f)


def Make_sinewave_response_plots_6feb():
	print '***  Make_sinewave_response_plots_6feb  ***'

	def get_sinewave_data(noise_file='all_sinewave_data_6feb.pkl'):
		with open(noise_file,'rb') as f:
			all_sinewave_data = pickle.load(f)
			frequency = all_sinewave_data['frequency']
			amplitude = all_sinewave_data['amplitude']
		return frequency, amplitude

	freq, amp = get_sinewave_data()

	frekkie  = np.asarray([1,3,6,10,30,60,120,250,500,1000,2000])
	frekkie2 = np.asarray([1,3,6,10,30,60,100,250,500,1000,2000])

	gain = np.asarray([0.,10.,20.,30.,40.,50.,60.,70.])

	syms = ['p','P','s','>','o','+','<','>']

	# Frequency plots
	plt.close()
	for i in range(0,8):
		plt.plot(frekkie[:], frekkie[:]*freq['60db_1VV'][:,0,i],syms[i] ,label=str(gain[i])+'db')
	plt.ylim([.95,1.05])
	plt.xscale('log')
	plt.xlabel('Input sine wave frequency (Hz)')
	plt.ylabel('Fitted / input frequency')
	plt.legend()
	plt.show()
	plt.savefig(plotroot+'sinewavefit_60db_1VV_freq.png',dpi=300)

	plt.close()
	for i in range(0,8):
		plt.plot(frekkie[:], frekkie[:]*freq['45db_1VV'][:,0,i],syms[i], label=str(gain[i])+'db')
	plt.ylim([.95,1.05])
	plt.xscale('log')
	plt.xlabel('Input sine wave frequency (Hz)')
	plt.ylabel('Fitted / input frequency')
	plt.legend()
	plt.show()
	plt.savefig(plotroot+'sinewavefit_45db_1VV_freq.png',dpi=300)

	plt.close()
	for i in range(0,8):
		plt.plot(frekkie2[:], frekkie2[:]*freq['45db_10VV'][:,0,i],syms[i], label=str(gain[i])+'db')
	plt.ylim([.95,1.05])
	plt.xscale('log')
	plt.xlabel('Input sine wave frequency (Hz)')
	plt.ylabel('Fitted / input frequency')
	plt.legend()
	plt.show()
	plt.savefig(plotroot+'sinewavefit_45db_10VV_freq.png',dpi=300)

	plt.close()
	for i in range(0,8):
		plt.plot(frekkie[:], frekkie[:]*freq['60db_10VV'][:,0,i],syms[i], label=str(gain[i])+'db')
	plt.ylim([.95,1.05])
	plt.xscale('log')
	plt.xlabel('Input sine wave frequency (Hz)')
	plt.ylabel('Fitted / input frequency')
	plt.legend()
	plt.show()
	plt.savefig(plotroot+'sinewavefit_60db_10VV_freq.png',dpi=300)


	fs = 8
	# Amplitude plots
	plt.close()
	for i in range(0,8):
		cf = np.power(10,gain[i]/20)
		plt.plot(frekkie[:], np.power(10,1.5)*1e6*(amp['60db_1VV'][:,0,i]/cf),syms[i] ,label=str(gain[i])+'db')
	plt.ylim([0,500])
	plt.xscale('log')
	plt.plot([0.7,2000],[260,260],'--',label='Norm. inp. amp.')
	plt.xlabel('input sine wave frequency (Hz)')
	plt.ylabel('Normalized amplitude')
	plt.legend(fontsize=fs,loc='best')
	plt.show()
	plt.savefig(plotroot+'sinewavefit_60db_1VV_amp.png',dpi=300)

	plt.close()
	for i in range(0,8):
		cf = np.power(10,gain[i]/20)
		plt.plot(frekkie[:], 1e6*(amp['45db_1VV'][:,0,i]/cf),syms[i], label=str(gain[i])+'db')
	plt.ylim([0,500])
	plt.plot([0.7,2000],[260,260],'--')
	plt.xscale('log')
	plt.plot([0.7,2000],[260,260],'--',label='Normalized input amplitude')
	plt.xlabel('input sine wave frequency (Hz)')
	plt.ylabel('Normalized amplitude')
	plt.legend(fontsize=fs,loc='best')
	plt.show()
	plt.savefig(plotroot+'sinewavefit_45db_1VV_amp.png',dpi=300)

	plt.close()
	for i in range(0,8):
		cf = np.power(10,gain[i]/20)
		plt.plot(frekkie2[:], 1e6*(amp['45db_10VV'][:,0,i]/cf/10.),syms[i], label=str(gain[i])+'db')
	plt.ylim([0,500])
	plt.plot([0.7,2000],[260,260],'--')
	plt.xscale('log')
	plt.plot([0.7,2000],[260,260],'--',label='Normalized input amplitude')
	plt.xlabel('input sine wave frequency (Hz)')
	plt.ylabel('Normalized amplitude')
	plt.legend(fontsize=fs,loc='best')
	plt.show()
	plt.savefig(plotroot+'sinewavefit_45db_10VV_amp.png',dpi=300)

	mask = np.ones(len(frekkie),dtype=bool)
	mask[[2]]=False
	plt.close()
	for i in range(0,8):
		cf = np.power(10,gain[i]/20)
		#plt.plot(frekkie[:], np.power(10,1.5)*1e6*(amp['60db_10VV'][:,0,i]/cf/10.),syms[i], label=str(gain[i])+'db')
		plt.plot(frekkie[mask], np.power(10,1.5)*1e6*(amp['60db_10VV'][mask,0,i]/cf/10.),syms[i], label=str(gain[i])+'db')
	plt.ylim([0,500])
	plt.plot([0.7,2000],[260,260],'--')
	plt.xscale('log')
	plt.plot([0.7,2000],[260,260],'--',label='Normalized input amplitude')
	plt.xlabel('input sine wave frequency (Hz)')
	plt.ylabel('Normalized amplitude')
	plt.legend(fontsize=fs,loc='best')
	plt.show()
	plt.savefig(plotroot+'sinewavefit_60db_10VV_amp.png',dpi=300)


def Test_wavelength_dependence_6feb():
	print '*** Test_wavelength_dependence_6feb ***'

	time1, signal1 = mytools.load_data('6feb_ch1',346,364 )
	time2, signal2 = mytools.load_data('6feb_ch1',366,371 )
	time3, signal3 = mytools.load_data('6feb_ch1',372,375 )
	time4, signal4 = mytools.load_data('6feb_PM_ch1',2,12 )
	time5, signal5 = mytools.load_data('6feb_PM_ch1',13,13 )

	mean1 = []
	mean2 = []
	mean3 = []
	mean4 = []
	mean5 = []
	for i in range(len(time1)):
		mean = stats.scoreatpercentile(signal1[i][:25000],50)
		mean = np.mean(signal1[i][:25000])
		mean1.append(mean)
	for i in range(len(time2)):
		mean = stats.scoreatpercentile(signal2[i][:25000],50)
		mean = np.mean(signal2[i][:25000])
		mean2.append(mean)
	for i in range(len(time3)):
		mean = stats.scoreatpercentile(signal3[i][:25000],50)
		mean = np.mean(signal3[i][:25000])
		mean3.append(mean)
	for i in range(len(time4)):
		mean = stats.scoreatpercentile(signal4[i][:25000],50)
		mean = np.mean(signal4[i][:25000])
		mean4.append(mean)
	for i in range(len(time5)):
		mean = stats.scoreatpercentile(signal5[i][:25000],50)
		mean = np.mean(signal5[i][:25000])
		mean5.append(mean)

	print mean1
	print mean2
	print mean3
	print mean4
	print mean5

	wavel1 = np.linspace(1550.,1568.,19)
	wavel2 = np.linspace(1568.,1563.,6)
	wavel3 = np.linspace(1562.8,1562.2,4)
	wavel4 = np.linspace(1562.,1560.,11)
	wavel5 = np.linspace(1550.,1550.,1)

	print wavel1
	print wavel2
	print wavel3
	print wavel4
	print wavel5

	syms = ['p','P','s','>','o','+','<','>']


	plt.close()
	plt.plot(wavel1,mean1,'o')
	plt.plot(wavel2,mean2,'>')
	plt.plot(wavel3,mean3,'s')
	plt.plot(wavel4,mean4,'p')
	plt.plot(wavel5,mean5,'+')
	plt.xlabel('Laser Wavelength [nm]')
	plt.ylabel('Signal [V]')
	plt.show()
	plt.savefig(plotroot+'wavelength_dependence.png',dpi=300)

def Test_wavelength_dependence_13feb():

	print '*** Test_wavelength_dependence_13feb ***'

	time1, signal1 = mytools.load_data('13feb_AM1_ch1',119,128 )
	time2, signal2 = mytools.load_data('13feb_AM1_ch1',129,138 )
	time3, signal3 = mytools.load_data('13feb_AM1_ch1',139,144 )
	time4, signal4 = mytools.load_data('13feb_AM1_ch1',146,160 )
	time5, signal5 = mytools.load_data('13feb_AM1_ch1',161,177 )

	mean1 = []
	mean2 = []
	mean3 = []
	mean4 = []
	mean5 = []
	for i in range(len(time1)):
		mean = stats.scoreatpercentile(signal1[i][:25000],50)
		#mean = np.mean(signal1[i][:25000])
		mean1.append(mean)
	for i in range(len(time2)):
		mean = stats.scoreatpercentile(signal2[i][:25000],50)
		#mean = np.mean(signal2[i][:25000])
		mean2.append(mean)
	for i in range(len(time3)):
		mean = stats.scoreatpercentile(signal3[i][:25000],50)
		#mean = np.mean(signal3[i][:25000])
		mean3.append(mean)
	for i in range(len(time4)):
		mean = stats.scoreatpercentile(signal4[i][:25000],50)
		#mean = np.mean(signal4[i][:25000])
		mean4.append(mean)
	for i in range(len(time5)):
		mean = stats.scoreatpercentile(signal5[i][:25000],50)
		#mean = np.mean(signal5[i][:25000])
		mean5.append(mean)

	wavel1 = np.linspace(1550.,1568.,10)
	wavel2 = np.linspace(1568.,1550.,10)
	wavel3 = np.linspace(1560.,1561.,6)
	wavel4 = np.linspace(1561.2,1564.,15)
	wavel5 = np.linspace(1561.7,1562.5,17)

	syms = ['p','P','s','>','o','+','<','>']

	plt.close()
	plt.plot(wavel1,mean1,'o')
	plt.plot(wavel2,mean2,'>')
	plt.xlabel('Laser Wavelength [nm]')
	plt.ylabel('Signal [V]')
	plt.show()
	plt.savefig(plotroot+'wavelength_dependence_13feb_up_down.png',dpi=300)

	plt.close()
	plt.plot(wavel1,mean1,'o')
	plt.plot(wavel3,mean3,'s')
	plt.plot(wavel4,mean4,'p')
	plt.xlabel('Laser Wavelength [nm]')
	plt.ylabel('Signal [V]')
	plt.show()
	plt.savefig(plotroot+'wavelength_dependence_13feb_medium_steps.png',dpi=300)

	plt.close()
	plt.plot(wavel3,mean3,'s')
	plt.plot(wavel4,mean4,'p')
	plt.plot(wavel5,mean5,'+')
	plt.xlabel('Laser Wavelength [nm]')
	plt.ylabel('Signal [V]')
	plt.show()
	plt.savefig(plotroot+'wavelength_dependence_13feb_small_steps.png',dpi=300)


	mean2r = list(reversed(mean2))
	print mean1
	print mean2
	print mean2r
	print '-----------------------'
	bla = np.asarray(mean1) - np.asarray(mean2r)
	bla2 = np.asarray(mean1) + np.asarray(mean2r) - 0.26
	print bla
	print np.std(bla) / np.mean(bla2)


def Compare_Femto_and_Thorlabs_sinewave_response():
	print '*** Compare_Femto_and_Thorlabs_sinewave_response ***'
	# Femto
	time1, signal1 = mytools.load_data('13feb_AM2_ch2',1,11)
	time2, signal2 = mytools.load_data('13feb_AM2_ch2',12,12)
	time3, signal3 = mytools.load_data('13feb_AM2_ch2',13,13)
	
	# Thorlabs 40dB
	#time_1, signal_1       = mytools.load_data('13feb_ch1',28,28)
	#time_3, signal_3       = mytools.load_data('13feb_ch1',36,36)
	#time_6, signal_6       = mytools.load_data('13feb_ch1',45,45)
	#time_10, signal_10     = mytools.load_data('13feb_AM1_ch1',12,12)
	#time_30, signal_30     = mytools.load_data('13feb_AM1_ch1',20,20)
	#time_60, signal_60     = mytools.load_data('13feb_AM1_ch1',33,33)
	#time_120, signal_120   = mytools.load_data('13feb_AM1_ch1',41,41)
	#time_250, signal_250   = mytools.load_data('13feb_AM1_ch1',49,49)
	#time_500, signal_500   = mytools.load_data('13feb_AM1_ch1',57,57)
	#time_1000, signal_1000 = mytools.load_data('13feb_AM1_ch1',65,65)
	#time_2000, signal_2000 = mytools.load_data('13feb_AM1_ch1',73,73)
	#time_dark, signal_dark = mytools.load_data('6feb_ch1',42,42)
	#time_0, signal_0 = mytools.load_data('6feb_ch1',50,50)
	
	# Thorlabs 50 dB
	time_1, signal_1       = mytools.load_data('13feb_ch1',29,29)
	time_3, signal_3       = mytools.load_data('13feb_ch1',37,37)
	time_6, signal_6       = mytools.load_data('13feb_ch1',46,46)
	time_10, signal_10     = mytools.load_data('13feb_AM1_ch1',13,13)
	time_30, signal_30     = mytools.load_data('13feb_AM1_ch1',21,21)
	time_60, signal_60     = mytools.load_data('13feb_AM1_ch1',34,34)
	time_120, signal_120   = mytools.load_data('13feb_AM1_ch1',42,42)
	time_250, signal_250   = mytools.load_data('13feb_AM1_ch1',50,50)
	time_500, signal_500   = mytools.load_data('13feb_AM1_ch1',58,58)
	time_1000, signal_1000 = mytools.load_data('13feb_AM1_ch1',66,66)
	time_2000, signal_2000 = mytools.load_data('13feb_AM1_ch1',74,74)
	time_dark, signal_dark = mytools.load_data('6feb_ch1',43,43)
	time_0, signal_0 = mytools.load_data('6feb_ch1',51,51)

	def append_amp_freq(time,sig,dark):
	   time = np.asarray(time)
	   sig = np.asarray(sig)
	   all_amp = []
	   all_freq = []
	   for i in range(11):
	      t = time[i][:25000]
	      g = sig[i][:25000]-dark
	      res = mytools.fit_sin(t,g)
	      freq = res['period']
	      amp = res['amp']
	      all_freq.append(1./freq)
	      all_amp.append(amp)
	   all_freq = np.abs(np.asarray(all_freq))
	   all_amp = np.abs(np.asarray(all_amp))
	   return all_amp, all_freq

	def Tappend_amp_freq(time,sig,dark):
	   time = np.asarray(time)
	   sig = np.asarray(sig)
	   all_amp = []
	   all_freq = []
	   for i in range(1):
	      t = time[i][:25000]
	      g = sig[i][:25000]-dark
	      res = mytools.fit_sin(t,g)
	      freq = res['period']
	      amp = res['amp']
	      all_freq.append(1./freq)
	      all_amp.append(amp)
	   all_freq = np.abs(np.asarray(all_freq))
	   all_amp = np.abs(np.asarray(all_amp))
	   return all_amp, all_freq


	# Thorlabs
	Tdark = np.mean(signal_dark)
	T1_amp, T1_freq       = Tappend_amp_freq(time_1,signal_1,Tdark)
	T3_amp, T3_freq       = Tappend_amp_freq(time_3,signal_3,Tdark)
	T6_amp, T6_freq       = Tappend_amp_freq(time_6,signal_6,Tdark)
	T10_amp, T10_freq     = Tappend_amp_freq(time_10,signal_10,Tdark)
	T30_amp, T30_freq     = Tappend_amp_freq(time_30,signal_30,Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_60,signal_60,Tdark)
	T120_amp, T120_freq   = Tappend_amp_freq(time_120,signal_120,Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_250,signal_250,Tdark)
	T500_amp, T500_freq   = Tappend_amp_freq(time_500,signal_500,Tdark)
	T1000_amp, T1000_freq = Tappend_amp_freq(time_1000,signal_1000,Tdark)
	T2000_amp, T2000_freq = Tappend_amp_freq(time_2000,signal_2000,Tdark)

	Tall_amp = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	Tall_freq = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]

	# Femto
	means_series = []
	means_cont = []
	means_dark = []
	means = np.mean(signal3[0])
	means_cont.append(means)
	means = np.mean(signal2[0])
	means_dark.append(means)
	for i in range(11):
		means = np.mean(signal1[i])
		means_series.append(means)

	all_amp, all_freq = append_amp_freq(time1,signal1,means_dark[0])

	real_freq = np.asarray([1,3,6,10,30,60,120,250,500,1000,2000])

	plt.close()
	plt.plot(all_freq, (all_freq-real_freq) / all_freq)
	plt.show()
	plt.xscale('log')

	plt.close()
	plt.plot(all_freq, all_amp/np.mean(all_amp[:6]),label='Femto')
	plt.plot(Tall_freq, Tall_amp/ np.mean(Tall_amp[:6]),label='Thorlabs')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Normalized signal')
	plt.show()
	plt.legend()
	plt.xscale('log')
	plt.savefig(plotroot+'Compare_Femto_Thorlabs.png',dpi=300)



def Plot_test():
#	times0, gain0     = mytools.load_data('6feb_ch1',247,254)
#	times60, gain60   = mytools.load_data('6feb_ch1',295,302)
	times0, gain0     = mytools.load_data('6feb_PM_ch1',24,31)
	times60, gain60   = mytools.load_data('6feb_PM2_ch1',13,20)
	print np.shape(times0[0])
	plt.close()
	gain_idx = 4
	range = 1000
	plt.plot(times0[gain_idx][:range], gain0[gain_idx][:range])
	plt.plot(times60[gain_idx][:range], gain60[gain_idx][:range])
	plt.show()


def Wavelength_variability_14feb():
	wave_up = np.linspace(1550,1568,num=19)
	#wave_down = np.linspace(1568,1550,num=19)
	sig_up   = [1.586,  1.378,  1.6236, 1.6843, 1.2938, 1.1523, 1.2441, 1.2327, 1.3549, 0.9471, 1.066,  0.9994, 0.8347, 1.1324, 0.8712, 0.8960, 1.1081, 0.9186, 1.0948]
	sig_down = [1.5657, 1.3819, 1.5891, 1.6631, 1.3504, 1.4391, 1.2922, 1.1651, 1.3465, 0.9904, 1.0217, 1.0378, 0.7945, 1.1277, 0.9428, 0.8664, 1.1325, 0.9290, 1.0785]
	evoa = [11.127, 11.070, 11.185, 11.152, 11.047, 10.897, 10.950, 11.08, 10.998, 10.75, 10.81, 10.83, 10.74, 10.65, 10.56, 10.75, 10.62, 10.633, 10.63]

	wave_up_hf = np.linspace(1562.0, 1562.2, num=21)
	sig_up_hf = [8299, 8299, 8334, 8342, 8417, 8440, 8522, 8480, 8521, 8603, 8676, 8773, 8777, 8896, 8958, 9028, 9039, 9125, 9242, 9270, 9366]
	sig_up_hf = np.asarray(sig_up_hf)/1e4

	wave_timeseries = np.linspace(1562.1, 1562.1, 12)
	sig_timeseries = [8698, 8760, 8774, 8740, 8692, 8684, 8625, 8638, 8633, 8614, 8667, 8687]
	sig_timeseries = np.asarray(sig_timeseries) / 1e4

	time4, signal4 = mytools.load_data('13feb_AM1_ch1',148,153 )
	time5, signal5 = mytools.load_data('13feb_AM1_ch1',161,177 )

	mean4 = []
	for i in range(len(time4)):
		mean = stats.scoreatpercentile(signal4[i][:25000],50)
		#mean = np.mean(signal5[i][:25000])
		mean4.append(mean)
	wavel4 = np.linspace(1561.6,1562.6,6)
	mean5 = []
	for i in range(len(time5)):
		mean = stats.scoreatpercentile(signal5[i][:25000],50)
		#mean = np.mean(signal5[i][:25000])
		mean5.append(mean)
	wavel5 = np.linspace(1561.7,1562.5,17)


	plt.close()
	plt.plot(wave_up,sig_up,'.')
	plt.plot(wave_up,sig_down,'p')
	plt.show()

	plt.close()
	plt.plot(wave_up,sig_up/(np.mean(sig_up)),'.')
	plt.plot(wave_up,sig_down/(np.mean(sig_down)),'p')
	plt.plot(wave_up, evoa/np.mean(evoa),'d')
	plt.show()

	plt.close()
	plt.plot(wave_up_hf, sig_up_hf/np.mean(sig_up_hf),'o',label='14Feb')
	plt.plot(wavel4, mean4/np.mean(mean4),'d',label='13Feb')
	plt.plot(wavel5, mean5/np.mean(mean4),'p',label='13Feb')
	plt.plot(wave_timeseries, sig_timeseries/np.mean(sig_up_hf),'.',label='14Feb')
	plt.xlabel('wavelength [nm]')
	plt.ylabel('Normalized signal')
	plt.legend(loc='best')
	plt.show()
	plt.savefig(plotroot+'Wavelength_instability.png', dpi=300)


def Make_plot_for_Thomas():
	
	print '*** Make_plot_for_Thomas ***'

	# Thorlabs 40dB
	time_1, signal_1       = mytools.load_data('13feb_ch1',28,28)
	time_3, signal_3       = mytools.load_data('13feb_ch1',36,36)
	time_6, signal_6       = mytools.load_data('13feb_ch1',45,45)
	time_10, signal_10     = mytools.load_data('13feb_AM1_ch1',12,12)
	time_30, signal_30     = mytools.load_data('13feb_AM1_ch1',20,20)
	time_60, signal_60     = mytools.load_data('13feb_AM1_ch1',33,33)
	time_120, signal_120   = mytools.load_data('13feb_AM1_ch1',41,41)
	time_250, signal_250   = mytools.load_data('13feb_AM1_ch1',49,49)
	time_500, signal_500   = mytools.load_data('13feb_AM1_ch1',57,57)
	time_1000, signal_1000 = mytools.load_data('13feb_AM1_ch1',65,65)
	time_2000, signal_2000 = mytools.load_data('13feb_AM1_ch1',73,73)
	time_dark, signal_dark = mytools.load_data('6feb_PM2_ch1',65,65)
	time_0, signal_0 = mytools.load_data('6feb_ch1',50,50)

	plt.close()
	plt.plot(time_1[0][:50000], signal_1[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('1Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_1Hz.png')
	plt.close()
	plt.plot(time_3[0][:50000], signal_3[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('3Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_3Hz.png')
	plt.close()
	plt.plot(time_6[0][:50000], signal_6[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('6Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_6Hz.png')
	plt.close()
	plt.plot(time_10[0][:50000], signal_10[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('10Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_10Hz.png')
	plt.close()
	plt.plot(time_30[0][:5000], signal_30[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('30Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_30Hz.png')
	plt.close()
	plt.plot(time_60[0][:5000], signal_60[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('60Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_60Hz.png')
	plt.close()
	plt.plot(time_120[0][:5000], signal_120[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('120Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_120Hz.png')
	plt.close()
	plt.plot(time_250[0][:500], signal_250[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('250Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_250Hz.png')
	plt.close()
	plt.plot(time_500[0][:500], signal_500[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('500Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_500Hz.png')
	plt.close()
	plt.plot(time_1000[0][:500], signal_1000[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('1000Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_1000Hz.png')
	plt.close()
	plt.plot(time_2000[0][:500], signal_2000[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('2000Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 40dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_2000Hz.png')

	# Thorlabs 30dB
	time_1, signal_1       = mytools.load_data('13feb_ch1',27,27)
	time_3, signal_3       = mytools.load_data('13feb_ch1',35,35)
	time_6, signal_6       = mytools.load_data('13feb_ch1',44,44)
	time_10, signal_10     = mytools.load_data('13feb_AM1_ch1',11,11)
	time_30, signal_30     = mytools.load_data('13feb_AM1_ch1',19,19)
	time_60, signal_60     = mytools.load_data('13feb_AM1_ch1',32,32)
	time_120, signal_120   = mytools.load_data('13feb_AM1_ch1',40,40)
	time_250, signal_250   = mytools.load_data('13feb_AM1_ch1',48,48)
	time_500, signal_500   = mytools.load_data('13feb_AM1_ch1',56,56)
	time_1000, signal_1000 = mytools.load_data('13feb_AM1_ch1',64,64)
	time_2000, signal_2000 = mytools.load_data('13feb_AM1_ch1',72,72)
	time_dark, signal_dark = mytools.load_data('6feb_PM2_ch1',64,64)
	time_0, signal_0 = mytools.load_data('6feb_ch1',50,50)

	plt.close()
	plt.plot(time_1[0][:50000], signal_1[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('1Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_1Hz_30dB.png')
	plt.close()
	plt.plot(time_3[0][:50000], signal_3[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('3Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_3Hz_30dB.png')
	plt.close()
	plt.plot(time_6[0][:50000], signal_6[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('6Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_6Hz_30dB.png')
	plt.close()
	plt.plot(time_10[0][:50000], signal_10[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('10Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_10Hz_30dB.png')
	plt.close()
	plt.plot(time_30[0][:5000], signal_30[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('30Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_30Hz_30dB.png')
	plt.close()
	plt.plot(time_60[0][:5000], signal_60[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('60Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_60Hz_30dB.png')
	plt.close()
	plt.plot(time_120[0][:5000], signal_120[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('120Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_120Hz_30dB.png')
	plt.close()
	plt.plot(time_250[0][:500], signal_250[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('250Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_250Hz_30dB.png')
	plt.close()
	plt.plot(time_500[0][:500], signal_500[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('500Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_500Hz_30dB.png')
	plt.close()
	plt.plot(time_1000[0][:500], signal_1000[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('1000Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_1000Hz_30dB.png')
	plt.close()
	plt.plot(time_2000[0][:500], signal_2000[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('2000Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 30dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_2000Hz_30dB.png')

	# Thorlabs 10dB
	time_1, signal_1       = mytools.load_data('13feb_ch1',25,25)
	time_3, signal_3       = mytools.load_data('13feb_ch1',33,33)
	time_6, signal_6       = mytools.load_data('13feb_ch1',42,42)
	time_10, signal_10     = mytools.load_data('13feb_AM1_ch1',9,9)
	time_30, signal_30     = mytools.load_data('13feb_AM1_ch1',17,17)
	time_60, signal_60     = mytools.load_data('13feb_AM1_ch1',30,30)
	time_120, signal_120   = mytools.load_data('13feb_AM1_ch1',38,38)
	time_250, signal_250   = mytools.load_data('13feb_AM1_ch1',46,46)
	time_500, signal_500   = mytools.load_data('13feb_AM1_ch1',54,54)
	time_1000, signal_1000 = mytools.load_data('13feb_AM1_ch1',62,62)
	time_2000, signal_2000 = mytools.load_data('13feb_AM1_ch1',70,70)
	time_dark, signal_dark = mytools.load_data('6feb_PM2_ch1',62,62)
	time_0, signal_0 = mytools.load_data('6feb_ch1',50,50)

	plt.close()
	plt.plot(time_1[0][:50000], signal_1[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('1Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_1Hz_10dB.png')
	plt.close()
	plt.plot(time_3[0][:50000], signal_3[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('3Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_3Hz_10dB.png')
	plt.close()
	plt.plot(time_6[0][:50000], signal_6[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('6Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_6Hz_10dB.png')
	plt.close()
	plt.plot(time_10[0][:50000], signal_10[0][:50000]-np.mean(signal_dark[0][:50000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('10Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_10Hz_10dB.png')
	plt.close()
	plt.plot(time_30[0][:5000], signal_30[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('30Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_30Hz_10dB.png')
	plt.close()
	plt.plot(time_60[0][:5000], signal_60[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('60Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_60Hz_10dB.png')
	plt.close()
	plt.plot(time_120[0][:5000], signal_120[0][:5000]-np.mean(signal_dark[0][:5000]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('120Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_120Hz_10dB.png')
	plt.close()
	plt.plot(time_250[0][:500], signal_250[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('250Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_250Hz_10dB.png')
	plt.close()
	plt.plot(time_500[0][:500], signal_500[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('500Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_500Hz_10dB.png')
	plt.close()
	plt.plot(time_1000[0][:500], signal_1000[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('1000Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_1000Hz_10dB.png')
	plt.close()
	plt.plot(time_2000[0][:500], signal_2000[0][:500]-np.mean(signal_dark[0][:500]))
	plt.xlabel('time [s]')
	plt.ylabel('Signal [v]  - dark corrected')
	plt.title('2000Hz; 1562.0 nm; 14dBm; >50dB attenuation; 10V/V; 10dB gain ')
	plt.show()
	plt.savefig(plotroot+'sinewave_response_2000Hz_10dB.png')



	# Thorlabs 50 dB
	#time_1, signal_1       = mytools.load_data('13feb_ch1',29,29)
	#time_3, signal_3       = mytools.load_data('13feb_ch1',37,37)
	#time_6, signal_6       = mytools.load_data('13feb_ch1',46,46)
	#time_10, signal_10     = mytools.load_data('13feb_AM1_ch1',13,13)
	#time_30, signal_30     = mytools.load_data('13feb_AM1_ch1',21,21)
	#time_60, signal_60     = mytools.load_data('13feb_AM1_ch1',34,34)
	#time_120, signal_120   = mytools.load_data('13feb_AM1_ch1',42,42)
	#time_250, signal_250   = mytools.load_data('13feb_AM1_ch1',50,50)
	#time_500, signal_500   = mytools.load_data('13feb_AM1_ch1',58,58)
	#time_1000, signal_1000 = mytools.load_data('13feb_AM1_ch1',66,66)
	#time_2000, signal_2000 = mytools.load_data('13feb_AM1_ch1',74,74)
	#time_dark, signal_dark = mytools.load_data('6feb_ch1',43,43)
	#time_0, signal_0 = mytools.load_data('6feb_ch1',51,51)
		
	#def append_amp_freq(time,sig,dark):
	#   time = np.asarray(time)
	#   sig = np.asarray(sig)
	#   all_amp = []
	#   all_freq = []
	#   for i in range(11):
	#      t = time[i][:25000]
	#      g = sig[i][:25000]-dark
	#      res = mytools.fit_sin(t,g)
	#      freq = res['period']
	#      amp = res['amp']
	#      all_freq.append(1./freq)
	#      all_amp.append(amp)
	#   all_freq = np.abs(np.asarray(all_freq))
	#   all_amp = np.abs(np.asarray(all_amp))
	#   return all_amp, all_freq

	#def Tappend_amp_freq(time,sig,dark):
	#   time = np.asarray(time)
	#   sig = np.asarray(sig)
	#   all_amp = []
	#   all_freq = []
	#   for i in range(1):
	#      t = time[i][:25000]
	#      g = sig[i][:25000]-dark
	#      res = mytools.fit_sin(t,g)
	#      freq = res['period']
	#      amp = res['amp']
	#      all_freq.append(1./freq)
	#      all_amp.append(amp)
	#   all_freq = np.abs(np.asarray(all_freq))
	#   all_amp = np.abs(np.asarray(all_amp))
	#   return all_amp, all_freq


	# Thorlabs
	#Tdark = np.mean(signal_dark)
	#T1_amp, T1_freq       = Tappend_amp_freq(time_1,signal_1,Tdark)
	#T3_amp, T3_freq       = Tappend_amp_freq(time_3,signal_3,Tdark)
	#T6_amp, T6_freq       = Tappend_amp_freq(time_6,signal_6,Tdark)
	#T10_amp, T10_freq     = Tappend_amp_freq(time_10,signal_10,Tdark)
	#T30_amp, T30_freq     = Tappend_amp_freq(time_30,signal_30,Tdark)
	#T60_amp, T60_freq     = Tappend_amp_freq(time_60,signal_60,Tdark)
	#T120_amp, T120_freq   = Tappend_amp_freq(time_120,signal_120,Tdark)
	#T250_amp, T250_freq   = Tappend_amp_freq(time_250,signal_250,Tdark)
	#T500_amp, T500_freq   = Tappend_amp_freq(time_500,signal_500,Tdark)
	#T1000_amp, T1000_freq = Tappend_amp_freq(time_1000,signal_1000,Tdark)
	#T2000_amp, T2000_freq = Tappend_amp_freq(time_2000,signal_2000,Tdark)

	#Tall_amp = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	#Tall_freq = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]

	# Femto
	#means_series = []
	#means_cont = []
	#means_dark = []
	#means = np.mean(signal3[0])
	#means_cont.append(means)
	#means = np.mean(signal2[0])
	#means_dark.append(means)
	#for i in range(11):
	#	means = np.mean(signal1[i])
	#	means_series.append(means)

	#all_amp, all_freq = append_amp_freq(time1,signal1,means_dark[0])

	#real_freq = np.asarray([1,3,6,10,30,60,120,250,500,1000,2000])

	#plt.close()
	#plt.plot(all_freq, (all_freq-real_freq) / all_freq)
	#plt.show()
	#plt.xscale('log')

	#plt.close()
	#plt.plot(all_freq, all_amp/np.mean(all_amp[:6]),label='Femto')
	#plt.plot(Tall_freq, Tall_amp/ np.mean(Tall_amp[:6]),label='Thorlabs')
	#plt.xlabel('Frequency [Hz]')
	#plt.ylabel('Normalized signal')
	#plt.show()
	#plt.legend()
	#plt.xscale('log')
	#plt.savefig(plotroot+'Compare_Femto_Thorlabs.png',dpi=300)

	
def analyse_sine_wave_21feb_Thorlabs():
	time_40db_1VV_dark,   signal_40db_1VV_dark    = mytools.load_data('21feb_ch1',59,59)
	time_40db_1VV_0Hz,    signal_40db_1VV_0Hz     = mytools.load_data('21feb_ch1',58,58)
	time_40db_1VV_1Hz,    signal_40db_1VV_1Hz     = mytools.load_data('21feb_ch1',47,47)
	time_40db_1VV_3Hz,    signal_40db_1VV_3Hz     = mytools.load_data('21feb_ch1',48,48)
	time_40db_1VV_6Hz,    signal_40db_1VV_6Hz     = mytools.load_data('21feb_ch1',49,49)
	time_40db_1VV_10Hz,   signal_40db_1VV_10Hz    = mytools.load_data('21feb_ch1',50,50)
	time_40db_1VV_30Hz,   signal_40db_1VV_30Hz    = mytools.load_data('21feb_ch1',51,51)
	time_40db_1VV_60Hz,   signal_40db_1VV_60Hz    = mytools.load_data('21feb_ch1',52,52)
	time_40db_1VV_120Hz,  signal_40db_1VV_120Hz   = mytools.load_data('21feb_ch1',53,53)
	time_40db_1VV_250Hz,  signal_40db_1VV_250Hz   = mytools.load_data('21feb_ch1',54,54)
	time_40db_1VV_500Hz,  signal_40db_1VV_500Hz   = mytools.load_data('21feb_ch1',55,55)
	time_40db_1VV_1000Hz, signal_40db_1VV_1000Hz  = mytools.load_data('21feb_ch1',56,56)
	time_40db_1VV_2000Hz, signal_40db_1VV_2000Hz  = mytools.load_data('21feb_ch1',57,57)
	#
	time_50db_1VV_dark,   signal_50db_1VV_dark    = mytools.load_data('21feb_ch1',61,61)
	time_50db_1VV_0Hz,    signal_50db_1VV_0Hz     = mytools.load_data('21feb_ch1',60,60)
	time_50db_1VV_1Hz,    signal_50db_1VV_1Hz     = mytools.load_data('21feb_ch1',36,36)
	time_50db_1VV_3Hz,    signal_50db_1VV_3Hz     = mytools.load_data('21feb_ch1',37,37)
	time_50db_1VV_6Hz,    signal_50db_1VV_6Hz     = mytools.load_data('21feb_ch1',38,38)
	time_50db_1VV_10Hz,   signal_50db_1VV_10Hz    = mytools.load_data('21feb_ch1',39,39)
	time_50db_1VV_30Hz,   signal_50db_1VV_30Hz    = mytools.load_data('21feb_ch1',40,40)
	time_50db_1VV_60Hz,   signal_50db_1VV_60Hz    = mytools.load_data('21feb_ch1',41,41)
	time_50db_1VV_120Hz,  signal_50db_1VV_120Hz   = mytools.load_data('21feb_ch1',42,42)
	time_50db_1VV_250Hz,  signal_50db_1VV_250Hz   = mytools.load_data('21feb_ch1',43,43)
	time_50db_1VV_500Hz,  signal_50db_1VV_500Hz   = mytools.load_data('21feb_ch1',44,44)
	time_50db_1VV_1000Hz, signal_50db_1VV_1000Hz  = mytools.load_data('21feb_ch1',45,45)
	time_50db_1VV_2000Hz, signal_50db_1VV_2000Hz  = mytools.load_data('21feb_ch1',46,46)
	#	
	time_60db_1VV_dark,   signal_60db_1VV_dark    = mytools.load_data('21feb_ch1',63,63)
	time_60db_1VV_0Hz,    signal_60db_1VV_0Hz     = mytools.load_data('21feb_ch1',62,62)
	time_60db_1VV_1Hz,    signal_60db_1VV_1Hz     = mytools.load_data('21feb_ch1',25,25)
	time_60db_1VV_3Hz,    signal_60db_1VV_3Hz     = mytools.load_data('21feb_ch1',26,26)
	time_60db_1VV_6Hz,    signal_60db_1VV_6Hz     = mytools.load_data('21feb_ch1',27,27)
	time_60db_1VV_10Hz,   signal_60db_1VV_10Hz    = mytools.load_data('21feb_ch1',28,28)
	time_60db_1VV_30Hz,   signal_60db_1VV_30Hz    = mytools.load_data('21feb_ch1',29,29)
	time_60db_1VV_60Hz,   signal_60db_1VV_60Hz    = mytools.load_data('21feb_ch1',30,30)
	time_60db_1VV_120Hz,  signal_60db_1VV_120Hz   = mytools.load_data('21feb_ch1',31,31)
	time_60db_1VV_250Hz,  signal_60db_1VV_250Hz   = mytools.load_data('21feb_ch1',32,32)
	time_60db_1VV_500Hz,  signal_60db_1VV_500Hz   = mytools.load_data('21feb_ch1',33,33)
	time_60db_1VV_1000Hz, signal_60db_1VV_1000Hz  = mytools.load_data('21feb_ch1',34,34)
	time_60db_1VV_2000Hz, signal_60db_1VV_2000Hz  = mytools.load_data('21feb_ch1',35,35)

	def Tappend_amp_freq(time,sig,dark):
	   time = np.asarray(time)
	   sig = np.asarray(sig)
	   t = time[0][:25000]
	   g = sig[0][:25000]-dark
	   res = mytools.fit_sin(t,g)
	   freq = res['period']
	   amp = res['amp']
	   return amp, freq


	# Thorlabs
	Tdark = np.mean(signal_40db_1VV_dark)
	T1_amp, T1_freq       = Tappend_amp_freq(time_40db_1VV_1Hz,  signal_40db_1VV_1Hz, Tdark)
	T3_amp, T3_freq       = Tappend_amp_freq(time_40db_1VV_3Hz,  signal_40db_1VV_3Hz, Tdark)
	T6_amp, T6_freq       = Tappend_amp_freq(time_40db_1VV_6Hz,  signal_40db_1VV_6Hz, Tdark)
	T10_amp, T10_freq     = Tappend_amp_freq(time_40db_1VV_10Hz,  signal_40db_1VV_10Hz, Tdark)
	T30_amp, T30_freq     = Tappend_amp_freq(time_40db_1VV_30Hz,  signal_40db_1VV_30Hz, Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_40db_1VV_60Hz,  signal_40db_1VV_60Hz, Tdark)
	T120_amp, T120_freq   = Tappend_amp_freq(time_40db_1VV_120Hz,  signal_40db_1VV_120Hz, Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_40db_1VV_250Hz,  signal_40db_1VV_250Hz, Tdark)
	T500_amp, T500_freq   = Tappend_amp_freq(time_40db_1VV_500Hz,  signal_40db_1VV_500Hz, Tdark)
	T1000_amp, T1000_freq = Tappend_amp_freq(time_40db_1VV_1000Hz,  signal_40db_1VV_1000Hz, Tdark)
	T2000_amp, T2000_freq = Tappend_amp_freq(time_40db_1VV_2000Hz,  signal_40db_1VV_2000Hz, Tdark)
	Tall_amp_40dB = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	Tall_freq_40dB = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]
	#
	Tdark = np.mean(signal_60db_1VV_dark)
	T1_amp, T1_freq       = Tappend_amp_freq(time_50db_1VV_1Hz,  signal_50db_1VV_1Hz, Tdark)
	T3_amp, T3_freq       = Tappend_amp_freq(time_50db_1VV_3Hz,  signal_50db_1VV_3Hz, Tdark)
	T6_amp, T6_freq       = Tappend_amp_freq(time_50db_1VV_6Hz,  signal_50db_1VV_6Hz, Tdark)
	T10_amp, T10_freq     = Tappend_amp_freq(time_50db_1VV_10Hz,  signal_50db_1VV_10Hz, Tdark)
	T30_amp, T30_freq     = Tappend_amp_freq(time_50db_1VV_30Hz,  signal_50db_1VV_30Hz, Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_50db_1VV_60Hz,  signal_50db_1VV_60Hz, Tdark)
	T120_amp, T120_freq   = Tappend_amp_freq(time_50db_1VV_120Hz,  signal_50db_1VV_120Hz, Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_50db_1VV_250Hz,  signal_50db_1VV_250Hz, Tdark)
	T500_amp, T500_freq   = Tappend_amp_freq(time_50db_1VV_500Hz,  signal_50db_1VV_500Hz, Tdark)
	T1000_amp, T1000_freq = Tappend_amp_freq(time_50db_1VV_1000Hz,  signal_50db_1VV_1000Hz, Tdark)
	T2000_amp, T2000_freq = Tappend_amp_freq(time_50db_1VV_2000Hz,  signal_50db_1VV_2000Hz, Tdark)
	Tall_amp_50dB = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	Tall_freq_50dB = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]
	#
	Tdark = np.mean(signal_60db_1VV_dark)
	T1_amp, T1_freq       = Tappend_amp_freq(time_60db_1VV_1Hz,  signal_60db_1VV_1Hz, Tdark)
	T3_amp, T3_freq       = Tappend_amp_freq(time_60db_1VV_3Hz,  signal_60db_1VV_3Hz, Tdark)
	T6_amp, T6_freq       = Tappend_amp_freq(time_60db_1VV_6Hz,  signal_60db_1VV_6Hz, Tdark)
	T10_amp, T10_freq     = Tappend_amp_freq(time_60db_1VV_10Hz,  signal_60db_1VV_10Hz, Tdark)
	T30_amp, T30_freq     = Tappend_amp_freq(time_60db_1VV_30Hz,  signal_60db_1VV_30Hz, Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_60db_1VV_60Hz,  signal_60db_1VV_60Hz, Tdark)
	T120_amp, T120_freq   = Tappend_amp_freq(time_60db_1VV_120Hz,  signal_60db_1VV_120Hz, Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_60db_1VV_250Hz,  signal_60db_1VV_250Hz, Tdark)
	T500_amp, T500_freq   = Tappend_amp_freq(time_60db_1VV_500Hz,  signal_60db_1VV_500Hz, Tdark)
	T1000_amp, T1000_freq = Tappend_amp_freq(time_60db_1VV_1000Hz,  signal_60db_1VV_1000Hz, Tdark)
	T2000_amp, T2000_freq = Tappend_amp_freq(time_60db_1VV_2000Hz,  signal_60db_1VV_2000Hz, Tdark)
	Tall_amp_60dB = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	Tall_freq_60dB = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]

	all_data_40dB = [Tall_amp_40dB, Tall_freq_40dB]
	all_data_50dB = [Tall_amp_50dB, Tall_freq_50dB]
	all_data_60dB = [Tall_amp_60dB, Tall_freq_60dB]

	# make some plots
	# First, subtract dark current from 0Hz signal
	Tdark40db = np.mean(signal_40db_1VV_dark)
	Tsig40db_mean = np.mean(signal_40db_1VV_0Hz) - Tdark40db 	
	Tdark50db = np.mean(signal_50db_1VV_dark)
	Tsig50db_mean = np.mean(signal_50db_1VV_0Hz) - Tdark50db 	
	Tdark60db = np.mean(signal_60db_1VV_dark)
	Tsig60db_mean = np.mean(signal_60db_1VV_0Hz) - Tdark60db 	

	noise_40db = np.std(signal_40db_1VV_0Hz)
	noise_50db = np.std(signal_50db_1VV_0Hz)
	noise_60db = np.std(signal_60db_1VV_0Hz)

	print "Mean signal:  ", Tsig40db_mean, Tsig50db_mean, Tsig60db_mean
	print "RMS signal:   ", noise_40db, noise_50db, noise_60db
	print "Dark signal:  ", Tdark40db, Tdark50db, Tdark60db
	print "SNR: ", Tsig40db_mean/noise_40db, Tsig50db_mean/noise_50db, Tsig60db_mean/noise_60db

	return all_data_40dB, all_data_50dB, all_data_60dB
	
def analyse_sine_wave_21feb_Femto():
	time_40db_1VV_dark,   signal_40db_1VV_dark    = mytools.load_data('21feb_ch2',59,59)
	time_40db_1VV_0Hz,    signal_40db_1VV_0Hz     = mytools.load_data('21feb_ch2',58,58)
	time_40db_1VV_1Hz,    signal_40db_1VV_1Hz     = mytools.load_data('21feb_ch2',47,47)
	time_40db_1VV_3Hz,    signal_40db_1VV_3Hz     = mytools.load_data('21feb_ch2',48,48)
	time_40db_1VV_6Hz,    signal_40db_1VV_6Hz     = mytools.load_data('21feb_ch2',49,49)
	time_40db_1VV_10Hz,   signal_40db_1VV_10Hz    = mytools.load_data('21feb_ch2',50,50)
	time_40db_1VV_30Hz,   signal_40db_1VV_30Hz    = mytools.load_data('21feb_ch2',51,51)
	time_40db_1VV_60Hz,   signal_40db_1VV_60Hz    = mytools.load_data('21feb_ch2',52,52)
	time_40db_1VV_120Hz,  signal_40db_1VV_120Hz   = mytools.load_data('21feb_ch2',53,53)
	time_40db_1VV_250Hz,  signal_40db_1VV_250Hz   = mytools.load_data('21feb_ch2',54,54)
	time_40db_1VV_500Hz,  signal_40db_1VV_500Hz   = mytools.load_data('21feb_ch2',55,55)
	time_40db_1VV_1000Hz, signal_40db_1VV_1000Hz  = mytools.load_data('21feb_ch2',56,56)
	time_40db_1VV_2000Hz, signal_40db_1VV_2000Hz  = mytools.load_data('21feb_ch2',57,57)
	#
	time_50db_1VV_dark,   signal_50db_1VV_dark    = mytools.load_data('21feb_ch2',61,61)
	time_50db_1VV_0Hz,    signal_50db_1VV_0Hz     = mytools.load_data('21feb_ch2',60,60)
	time_50db_1VV_1Hz,    signal_50db_1VV_1Hz     = mytools.load_data('21feb_ch2',36,36)
	time_50db_1VV_3Hz,    signal_50db_1VV_3Hz     = mytools.load_data('21feb_ch2',37,37)
	time_50db_1VV_6Hz,    signal_50db_1VV_6Hz     = mytools.load_data('21feb_ch2',38,38)
	time_50db_1VV_10Hz,   signal_50db_1VV_10Hz    = mytools.load_data('21feb_ch2',39,39)
	time_50db_1VV_30Hz,   signal_50db_1VV_30Hz    = mytools.load_data('21feb_ch2',40,40)
	time_50db_1VV_60Hz,   signal_50db_1VV_60Hz    = mytools.load_data('21feb_ch2',41,41)
	time_50db_1VV_120Hz,  signal_50db_1VV_120Hz   = mytools.load_data('21feb_ch2',42,42)
	time_50db_1VV_250Hz,  signal_50db_1VV_250Hz   = mytools.load_data('21feb_ch2',43,43)
	time_50db_1VV_500Hz,  signal_50db_1VV_500Hz   = mytools.load_data('21feb_ch2',44,44)
	time_50db_1VV_1000Hz, signal_50db_1VV_1000Hz  = mytools.load_data('21feb_ch2',45,45)
	time_50db_1VV_2000Hz, signal_50db_1VV_2000Hz  = mytools.load_data('21feb_ch2',46,46)
	#	
	time_60db_1VV_dark,   signal_60db_1VV_dark    = mytools.load_data('21feb_ch2',63,63)
	time_60db_1VV_0Hz,    signal_60db_1VV_0Hz     = mytools.load_data('21feb_ch2',62,62)
	time_60db_1VV_1Hz,    signal_60db_1VV_1Hz     = mytools.load_data('21feb_ch2',25,25)
	time_60db_1VV_3Hz,    signal_60db_1VV_3Hz     = mytools.load_data('21feb_ch2',26,26)
	time_60db_1VV_6Hz,    signal_60db_1VV_6Hz     = mytools.load_data('21feb_ch2',27,27)
	time_60db_1VV_10Hz,   signal_60db_1VV_10Hz    = mytools.load_data('21feb_ch2',28,28)
	time_60db_1VV_30Hz,   signal_60db_1VV_30Hz    = mytools.load_data('21feb_ch2',29,29)
	time_60db_1VV_60Hz,   signal_60db_1VV_60Hz    = mytools.load_data('21feb_ch2',30,30)
	time_60db_1VV_120Hz,  signal_60db_1VV_120Hz   = mytools.load_data('21feb_ch2',31,31)
	time_60db_1VV_250Hz,  signal_60db_1VV_250Hz   = mytools.load_data('21feb_ch2',32,32)
	time_60db_1VV_500Hz,  signal_60db_1VV_500Hz   = mytools.load_data('21feb_ch2',33,33)
	time_60db_1VV_1000Hz, signal_60db_1VV_1000Hz  = mytools.load_data('21feb_ch2',34,34)
	time_60db_1VV_2000Hz, signal_60db_1VV_2000Hz  = mytools.load_data('21feb_ch2',35,35)

	def Tappend_amp_freq(time,sig,dark):
	   time = np.asarray(time)
	   sig = np.asarray(sig)
	   t = time[0][:25000]
	   g = sig[0][:25000]-dark
	   res = mytools.fit_sin(t,g)
	   freq = res['period']
	   amp = res['amp']
	   return amp, freq


	# Femto
	Tdark = np.mean(signal_40db_1VV_dark)
	T1_amp, T1_freq       = Tappend_amp_freq(time_40db_1VV_1Hz,  signal_40db_1VV_1Hz, Tdark)
	T3_amp, T3_freq       = Tappend_amp_freq(time_40db_1VV_3Hz,  signal_40db_1VV_3Hz, Tdark)
	T6_amp, T6_freq       = Tappend_amp_freq(time_40db_1VV_6Hz,  signal_40db_1VV_6Hz, Tdark)
	T10_amp, T10_freq     = Tappend_amp_freq(time_40db_1VV_10Hz,  signal_40db_1VV_10Hz, Tdark)
	T30_amp, T30_freq     = Tappend_amp_freq(time_40db_1VV_30Hz,  signal_40db_1VV_30Hz, Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_40db_1VV_60Hz,  signal_40db_1VV_60Hz, Tdark)
	T120_amp, T120_freq   = Tappend_amp_freq(time_40db_1VV_120Hz,  signal_40db_1VV_120Hz, Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_40db_1VV_250Hz,  signal_40db_1VV_250Hz, Tdark)
	T500_amp, T500_freq   = Tappend_amp_freq(time_40db_1VV_500Hz,  signal_40db_1VV_500Hz, Tdark)
	T1000_amp, T1000_freq = Tappend_amp_freq(time_40db_1VV_1000Hz,  signal_40db_1VV_1000Hz, Tdark)
	T2000_amp, T2000_freq = Tappend_amp_freq(time_40db_1VV_2000Hz,  signal_40db_1VV_2000Hz, Tdark)
	Tall_amp_40dB = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	Tall_freq_40dB = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]
	#
	Tdark = np.mean(signal_50db_1VV_dark)
	T1_amp, T1_freq       = Tappend_amp_freq(time_50db_1VV_1Hz,  signal_50db_1VV_1Hz, Tdark)
	T3_amp, T3_freq       = Tappend_amp_freq(time_50db_1VV_3Hz,  signal_50db_1VV_3Hz, Tdark)
	T6_amp, T6_freq       = Tappend_amp_freq(time_50db_1VV_6Hz,  signal_50db_1VV_6Hz, Tdark)
	T10_amp, T10_freq     = Tappend_amp_freq(time_50db_1VV_10Hz,  signal_50db_1VV_10Hz, Tdark)
	T30_amp, T30_freq     = Tappend_amp_freq(time_50db_1VV_30Hz,  signal_50db_1VV_30Hz, Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_50db_1VV_60Hz,  signal_50db_1VV_60Hz, Tdark)
	T120_amp, T120_freq   = Tappend_amp_freq(time_50db_1VV_120Hz,  signal_50db_1VV_120Hz, Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_50db_1VV_250Hz,  signal_50db_1VV_250Hz, Tdark)
	T500_amp, T500_freq   = Tappend_amp_freq(time_50db_1VV_500Hz,  signal_50db_1VV_500Hz, Tdark)
	T1000_amp, T1000_freq = Tappend_amp_freq(time_50db_1VV_1000Hz,  signal_50db_1VV_1000Hz, Tdark)
	T2000_amp, T2000_freq = Tappend_amp_freq(time_50db_1VV_2000Hz,  signal_50db_1VV_2000Hz, Tdark)
	Tall_amp_50dB = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	Tall_freq_50dB = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]
	#
	Tdark = np.mean(signal_60db_1VV_dark)
	T1_amp, T1_freq       = Tappend_amp_freq(time_60db_1VV_1Hz,  signal_60db_1VV_1Hz, Tdark)
	T3_amp, T3_freq       = Tappend_amp_freq(time_60db_1VV_3Hz,  signal_60db_1VV_3Hz, Tdark)
	T6_amp, T6_freq       = Tappend_amp_freq(time_60db_1VV_6Hz,  signal_60db_1VV_6Hz, Tdark)
	T10_amp, T10_freq     = Tappend_amp_freq(time_60db_1VV_10Hz,  signal_60db_1VV_10Hz, Tdark)
	T30_amp, T30_freq     = Tappend_amp_freq(time_60db_1VV_30Hz,  signal_60db_1VV_30Hz, Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_60db_1VV_60Hz,  signal_60db_1VV_60Hz, Tdark)
	T120_amp, T120_freq   = Tappend_amp_freq(time_60db_1VV_120Hz,  signal_60db_1VV_120Hz, Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_60db_1VV_250Hz,  signal_60db_1VV_250Hz, Tdark)
	T500_amp, T500_freq   = Tappend_amp_freq(time_60db_1VV_500Hz,  signal_60db_1VV_500Hz, Tdark)
	T1000_amp, T1000_freq = Tappend_amp_freq(time_60db_1VV_1000Hz,  signal_60db_1VV_1000Hz, Tdark)
	T2000_amp, T2000_freq = Tappend_amp_freq(time_60db_1VV_2000Hz,  signal_60db_1VV_2000Hz, Tdark)
	Tall_amp_60dB = [T1_amp, T3_amp, T6_amp, T10_amp, T30_amp, T60_amp, T120_amp, T250_amp, T500_amp, T1000_amp, T2000_amp]
	Tall_freq_60dB = [T1_freq, T3_freq, T6_freq, T10_freq, T30_freq, T60_freq, T120_freq, T250_freq, T500_freq, T1000_freq, T2000_freq]

	all_data_40dB = [Tall_amp_40dB, Tall_freq_40dB]
	all_data_50dB = [Tall_amp_50dB, Tall_freq_50dB]
	all_data_60dB = [Tall_amp_60dB, Tall_freq_60dB]


	# make some plots
	# First, subtract dark current from 0Hz signal
	Tdark = np.mean(signal_40db_1VV_dark)
	Tsig_mean = np.mean(signal_40db_1VV_0Hz) - Tdark 	

	noise = np.std(signal_40db_1VV_0Hz)

	print "Mean signal:  ", Tsig_mean
	print "RMS signal:   ", noise
	print "Dark signal:  ", Tdark
	print "SNR: ", Tsig_mean/noise


	return all_data_40dB, all_data_50dB, all_data_60dB
	
def plot_raw_sinewave_21feb():
	
	def Tappend_amp_freq(time,sig,dark):
	   time = np.asarray(time)
	   sig = np.asarray(sig)
	   t = time[0][:25000]
	   g = sig[0][:25000]-dark
	   res = mytools.fit_sin(t,g)
	   freq = res['period']
	   amp = res['amp']
	   return amp, freq


	
	time_40db_1VV_dark,   signal_40db_1VV_dark    = mytools.load_data('21feb_ch1',59,59)
	time_40db_1VV_0Hz,    signal_40db_1VV_0Hz     = mytools.load_data('21feb_ch1',58,58)
	time_40db_1VV_6Hz,    signal_40db_1VV_6Hz     = mytools.load_data('21feb_ch1',49,49)
	time_40db_1VV_60Hz,   signal_40db_1VV_60Hz    = mytools.load_data('21feb_ch1',52,52)
	time_40db_1VV_250Hz,  signal_40db_1VV_250Hz   = mytools.load_data('21feb_ch1',54,54)

	Tdark = np.mean(signal_40db_1VV_dark)
	Toffset = np.mean(signal_40db_1VV_0Hz)
	T6_amp, T6_freq       = Tappend_amp_freq(time_40db_1VV_6Hz,  signal_40db_1VV_6Hz, Tdark)
	T60_amp, T60_freq     = Tappend_amp_freq(time_40db_1VV_60Hz,  signal_40db_1VV_60Hz, Tdark)
	T250_amp, T250_freq   = Tappend_amp_freq(time_40db_1VV_250Hz,  signal_40db_1VV_250Hz, Tdark)

	
	#plt.close()
	#plt.plot(time_40db_1VV_10Hz_1T[0], signal_40db_1VV_10Hz_1T[0])
	#plt.show()

	fs = 20000  # sampling rate, Hz


	ts = time_40db_1VV_6Hz[0][:30000]	# 2 second
	yraw = signal_40db_1VV_6Hz[0][:30000]
	#ytruish = 0.136 + 0.0022*np.sin(2*np.pi * 6.0 * ts + 10)
	ytruish = Toffset -0.5*np.abs(T6_amp) + np.abs(T6_amp)*np.sin(2*np.pi * 6.0 * ts + 10)
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter = scipy.signal.lfilter(b, a, yraw)

	plt.close()
	plt.plot(ts, yraw, label="Raw signal")
	plt.plot(ts, ytruish, label="Fitted signal")
	plt.plot(ts, y_lfilter, alpha=0.8, lw=3, label="SciPy lfilter")
	plt.xlabel("Time / s")
	plt.ylabel("Amplitude")
	plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=2,
           fontsize="smaller")
	plt.xlim([0.5,1.5])
	plt.ylim([.1,.175])
	plt.show()
	plt.savefig('sinefit_40db_6Hz.png',dpi=300)


	ts = time_40db_1VV_60Hz[0][:30000]	# 2 second
	yraw = signal_40db_1VV_60Hz[0][:30000]
	ytruish = Toffset -0.5*np.abs(T60_amp) + np.abs(T60_amp)*np.sin(2*np.pi * 60.0 * ts + 15)
	#ytruish = 0.136 + 0.0022*np.sin(2*np.pi * 60.0 * ts + 10)
	b, a = scipy.signal.iirfilter(4, Wn=120.0, fs=fs, btype="low", ftype="butter")
	y_lfilter = scipy.signal.lfilter(b, a, yraw)

	plt.close()
	plt.plot(ts, yraw, label="Raw signal")
	plt.plot(ts, ytruish, label="Fitted signal")
	plt.plot(ts, y_lfilter, alpha=0.8, lw=3, label="SciPy lfilter")
	plt.xlabel("Time / s")
	plt.ylabel("Amplitude")
	plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=2,
           fontsize="smaller")
	plt.xlim([0.5,0.57])
	plt.ylim([.11,.165])
	plt.show()
	plt.savefig('sinefit_40db_60Hz.png',dpi=300)


	ts = time_40db_1VV_250Hz[0][:30000]	# 2 second
	yraw = signal_40db_1VV_250Hz[0][:30000]
	#ytruish = 0.136 + 0.0022*np.sin(2*np.pi * 250.0 * ts + 10)
	ytruish = Toffset -0.5*np.abs(T250_amp) + np.abs(T250_amp)*np.sin(2*np.pi * 250.0 * ts + 10)
	b, a = scipy.signal.iirfilter(4, Wn=500.0, fs=fs, btype="low", ftype="butter")
	y_lfilter = scipy.signal.lfilter(b, a, yraw)

	plt.close()
	plt.plot(ts, yraw, label="Raw signal")
	plt.plot(ts, ytruish, label="Fitted signal")
	plt.plot(ts, y_lfilter, alpha=0.8, lw=3, label="SciPy lfilter")
	plt.xlabel("Time / s")
	plt.ylabel("Amplitude")
	plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=2,
           fontsize="smaller")
	plt.xlim([0.5,0.52])
	plt.ylim([.1,.175])
	plt.show()
	plt.savefig('sinefit_40db_250Hz.png',dpi=300)


def snr_thorlabs_22feb():
	
	# Read data
	t_T, s_T = mytools.load_data('22feb_ch1',4,52)
	t_F, s_F = mytools.load_data('22feb_ch2',4,52)

	T_dark = np.mean(s_T[48])
	F_dark = np.mean(s_F[48])

	T_amp = 10	# V/V

	F_gain = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, \
			  1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, \
			  1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, \
			  1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]
	
	T_mean = []
	F_mean = []
	T_noise = []
	F_noise = []
	T_snr = []
	F_snr = []
	for i in range(1,46):
		# Thorlabs
		mean = np.mean(s_T[i][1000:]/T_amp) - T_dark/T_amp
		noise = np.std(s_T[i][1000:]/T_amp)
		T_mean.append(mean)
		T_noise.append(noise)
		snr = mean / noise
		T_snr.append(snr)
		# Femto
		mean = np.mean(s_F[i][1000:]) - F_dark
		noise = np.std(s_F[i][1000:])
		snr = mean / noise
		F_snr.append(snr)
		F_mean.append(mean/F_gain[i-1])
		F_noise.append(noise)

	T_mean = np.asarray(T_mean)
	F_mean = np.asarray(F_mean)
	T_noise = np.asarray(T_noise)
	F_noise = np.asarray(F_noise)
	T_snr = np.asarray(T_snr)
	F_snr = np.asarray(F_snr)

	# Convert to dBm
	F_mean_dBm = 10.0 * np.log10(1e3*F_mean)

	plt.close()
	plt.plot(F_mean, T_mean/F_mean/1e6,'.')
	plt.xscale('log')
	plt.show()
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Mean Thorlabs / Femto signals / 1E6 ')
	plt.savefig('Thorlabs_vs_Femto_HighSpeed.png',dpi=300)
	
	plt.close()
	plt.plot(F_mean_dBm, T_mean/F_mean/1e6,'.')
	plt.show()
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Mean Thorlabs / Femto signals / 1E6 ')
	plt.savefig('Thorlabs_vs_Femto_HighSpeed_dBm.png',dpi=300)
	
	plt.close()
	plt.plot(F_mean, F_snr, '.')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Signal-to-Noise')
	plt.show()
	plt.title('Femto OE-200-IN2 HighSpeed')
	plt.savefig('SNR_Femto_HighSPeed.png',dpi=300)

	plt.close()
	plt.plot(F_mean_dBm, F_snr, '.')
	plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal-to-Noise')
	plt.show()
	plt.title('Femto OE-200-IN2 HighSpeed')
	plt.savefig('SNR_Femto_HighSPeed_dBm.png',dpi=300)

	plt.close()
	plt.plot(F_mean, T_snr,'.')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Thorlabs PDA30B2')
	plt.show()
	plt.savefig('SNR_Thorlabs_v1.png',dpi=300)
	
	plt.close()
	plt.plot(F_mean_dBm, T_snr,'.')
	plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Thorlabs PDA30B2')
	plt.show()
	plt.savefig('SNR_Thorlabs_v1_dBm.png',dpi=300)

	#plt.close()
	#plt.plot(F_mean_dBm, 1e3*T_noise,'.',label="series 1")
	##plt.plot(F_mean_H_dBm, T_snr_H,'.',label="series 2")
	##plt.yscale('log')
	#plt.xlabel('Signal at detector [dBm]')
	#plt.ylabel('Noise [RMS mV]')
	#plt.title('Thorlabs PDA30B2')
	#plt.legend()
	##plt.ylim([8, 8.6])
	#plt.show()
	#plt.savefig('SNR_Thorlabs_noise_dBm.png',dpi=300)
	
	plt.close()
	plt.plot(F_mean_dBm, 1e4*T_noise,'.',label="series 1")
	#plt.plot(F_mean_H_dBm, T_snr_H,'.',label="series 2")
	#plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Noise [RMS mV]')
	plt.title('Thorlabs PDA30B2')
	plt.legend()
	plt.ylim([8, 8.6])
	plt.show()
	plt.savefig('SNR_Thorlabs_noise_dBm.png',dpi=300)

	

def snr_femto_22feb():


	# Read data
	t_F_H, s_F_H = mytools.load_data('22feb_ch2',4,52)
	F_dark_H = np.mean(s_F_H[48])
	t_T_H, s_T_H = mytools.load_data('22feb_ch1',4,52)
	T_dark_H = np.mean(s_T_H[48])

	F_gain_H = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, \
			  1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, \
			  1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7, \
			  1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]
	
	F_mean_H = []
	F_noise_H = []
	F_snr_H = []
	T_mean_H = []
	T_noise_H = []
	T_snr_H = []
	for i in range(1,46):
		# Femto
		mean = np.mean(s_F_H[i][1000:]) - F_dark_H
		noise = np.std(s_F_H[i][1000:])
		snr = mean / noise
		F_snr_H.append(snr)
		F_mean_H.append(mean/F_gain_H[i-1])
		F_noise_H.append(noise)
		# Thorlabs
		mean = np.mean(s_T_H[i][1000:]) - T_dark_H
		noise = np.std(s_T_H[i][1000:])
		snr = mean / noise
		T_snr_H.append(snr)
		T_mean_H.append(mean)
		T_noise_H.append(noise)

	F_mean_H = np.asarray(F_mean_H)
	F_noise_H = np.asarray(F_noise_H)
	F_snr_H = np.asarray(F_snr_H)
	T_mean_H = np.asarray(T_mean_H)
	T_noise_H = np.asarray(T_noise_H)
	T_snr_H = np.asarray(T_snr_H)

	# Convert to dBm
	F_mean_H_dBm = 10.0 * np.log10(1e3*F_mean_H)

	
	# Read data
	t_T1, s_T1 = mytools.load_data('22feb_ch1',70,79)
	t_F1, s_F1 = mytools.load_data('22feb_ch2',70,79)
	t_T2, s_T2 = mytools.load_data('22feb_ch1',81,91)
	t_F2, s_F2 = mytools.load_data('22feb_ch2',81,91)
	t_T = np.concatenate((t_T1, t_T2))
	s_T = np.concatenate((s_T1, s_T2))
	t_F = np.concatenate((t_F1, t_F2))
	s_F = np.concatenate((s_F1, s_F2))

	T_dark = np.mean(s_T[0])
	F_dark = np.mean(s_F[20])

	F_gain = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, \
			  1e8, 1e8, 1e8, 1e8, 1e8, \
			  1e7, 1e7, 1e7, 1e7, 1e7, \
			  1e6, 1e6, 1e6]

	T_mean = []
	F_mean = []
	T_noise = []
	F_noise = []
	F_noise_W = []
	T_snr = []
	F_snr = []
	for i in range(1,20):
		# Thorlabs
		mean = np.mean(s_T[i][1000:]) - T_dark
		noise = np.std(s_T[i][1000:])
		T_mean.append(mean)
		T_noise.append(noise)
		snr = mean / noise
		T_snr.append(snr)
		# Femto
		mean = np.mean(s_F[i][1000:]) - F_dark
		noise = np.std(s_F[i][1000:])
		snr = mean / noise
		F_snr.append(snr)
		F_mean.append(mean/F_gain[i-1])
		F_noise.append(noise)
		F_noise_W.append(noise/F_gain[i-1])

	T_mean = np.asarray(T_mean)
	F_mean = np.asarray(F_mean)
	T_noise = np.asarray(T_noise)
	F_noise = np.asarray(F_noise)
	F_noise_W = np.asarray(F_noise_W)
	T_snr = np.asarray(T_snr)
	F_snr = np.asarray(F_snr)

	# Convert to dBm
	F_mean_H_dBm = 10.0 * np.log10(1e3*F_mean_H)
	F_mean_dBm = 10.0 * np.log10(1e3*F_mean)






	plt.close()
	plt.plot(F_mean, T_mean/F_mean /1e6,'.')
	plt.xscale('log')
	plt.show()
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Mean Thorlabs / Femto signals / 1e6 ')
	plt.savefig('Thorlabs_vs_Femto_LowNoise.png',dpi=300)

	plt.close()
	plt.plot(F_mean_dBm, T_mean/F_mean /1e6,'.')
	plt.show()
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Mean Thorlabs / Femto signals / 1e6 ')
	plt.savefig('Thorlabs_vs_Femto_LowNoise_dBm.png',dpi=300)


	plt.close()
	plt.plot(F_mean_dBm, F_noise,'.',label="series 1")
	#plt.plot(F_mean_H_dBm, T_snr_H,'.',label="series 2")
	#plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Noise [RMS V]')
	plt.title('Femto OE-200-IN2   LowNoise ')
	plt.legend()
	plt.show()
	plt.savefig('SNR_Femto_noise_dBm.png',dpi=300)

	plt.close()
	plt.plot(F_mean_dBm, F_noise_W,'.',label="series 1")
	#plt.plot(F_mean_H_dBm, T_snr_H,'.',label="series 2")
	#plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Noise [RMS W]')
	plt.title('Femto OE-200-IN2   LowNoise ')
	plt.legend()
	plt.show()
	plt.savefig('SNR_Femto_noiseW_dBm.png',dpi=300)

	
	plt.close()
	plt.plot(F_mean, F_snr, '.')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Femto OE-200-IN2  LowNoise')
	plt.show()
	plt.savefig('SNR_Femto_LowNoise.png',dpi=300)

	plt.close()
	plt.plot(F_mean_dBm, F_snr, '.')
	plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Femto OE-200-IN2  LowNoise')
	plt.show()
	plt.savefig('SNR_Femto_LowNoise_dBm.png',dpi=300)

	plt.close()
	plt.plot(F_mean_H, F_snr_H, '.',label="HighSpeed")
	plt.plot(F_mean, F_snr, '.', label="LowNoise")
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Femto OE-200-IN2  LowNoise')
	plt.legend()
	plt.show()
	plt.savefig('SNR_Femto_all.png',dpi=300)

	plt.close()
	plt.plot(F_mean_H_dBm, F_snr_H, '.',label="HighSpeed")
	plt.plot(F_mean_dBm, F_snr, '.', label="LowNoise")
	plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Femto OE-200-IN2  LowNoise')
	plt.legend()
	plt.show()
	plt.savefig('SNR_Femto_all_dBm.png',dpi=300)

	plt.close()
	plt.plot(F_mean, T_snr,'.')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Thorlabs PDA30B2')
	plt.show()
	plt.savefig('SNR_Thorlabs_v2.png',dpi=300)

	plt.close()
	plt.plot(F_mean_dBm, T_snr,'.')
	plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Thorlabs PDA30B2')
	plt.show()
	plt.savefig('SNR_Thorlabs_v2_dBm.png',dpi=300)
	
	plt.close()
	plt.plot(F_mean, T_snr,'.',label="series 1")
	plt.plot(F_mean_H, T_snr_H,'.',label="series 2")
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Signal at detector [W]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Thorlabs PDA30B2')
	plt.legend()
	plt.show()
	plt.savefig('SNR_Thorlabs_all.png',dpi=300)

	plt.close()
	plt.plot(F_mean_dBm, T_snr,'.',label="series 1")
	plt.plot(F_mean_H_dBm, T_snr_H,'.',label="series 2")
	plt.yscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal-to-Noise')
	plt.title('Thorlabs PDA30B2')
	plt.legend()
	plt.show()
	plt.savefig('SNR_Thorlabs_all_dBm.png',dpi=300)


def plot_snr_27feb():
	# Read data
	# Thorlabs is Ch1
	# Femto is Ch2
	
	t11, s11 = mytools.load_data('27feb_PM_ch1',60,63)
	t21, s21 = mytools.load_data('27feb_PM_ch2',60,63)
	t12, s12 = mytools.load_data('27feb_PM_ch1',72,75)
	t22, s22 = mytools.load_data('27feb_PM_ch2',72,75)
	t13, s13 = mytools.load_data('27feb_PM_ch1',82,85)
	t23, s23 = mytools.load_data('27feb_PM_ch2',82,85)
	t14, s14 = mytools.load_data('27feb_PM_ch1',93,96)
	t24, s24 = mytools.load_data('27feb_PM_ch2',93,96)
	t15, s15 = mytools.load_data('27feb_PM_ch1',103,106)
	t25, s25 = mytools.load_data('27feb_PM_ch2',103,106)
	tdt, sdt = mytools.load_data('27feb_PM_ch1',107,107)
	tdf, sdf = mytools.load_data('27feb_PM_ch2',107,107)

	sig_T = []
	noise_T = []
	sig_F = []
	noise_F = []
	dark_T = []
	dark_F = []
	
	for i in range(len(s11)):
		sig = np.mean(s11[i][1000:])
		noise = np.std(s11[i][1000:])
		sig_T.append(sig)
		noise_T.append(noise)
		sig = np.mean(s21[i][1000:])
		noise = np.std(s21[i][1000:])
		sig_F.append(sig)
		noise_F.append(noise)
	for i in range(len(s12)):
		sig = np.mean(s12[i][1000:])
		noise = np.std(s12[i][1000:])
		sig_T.append(sig)
		noise_T.append(noise)
		sig = np.mean(s22[i][1000:])
		noise = np.std(s22[i][1000:])
		sig_F.append(sig)
		noise_F.append(noise)
	for i in range(len(s13)):
		sig = np.mean(s13[i][1000:])
		noise = np.std(s13[i][1000:])
		sig_T.append(sig)
		noise_T.append(noise)
		sig = np.mean(s23[i][1000:])
		noise = np.std(s23[i][1000:])
		sig_F.append(sig)
		noise_F.append(noise)
	for i in range(len(s14)):
		sig = np.mean(s14[i][1000:])
		noise = np.std(s14[i][1000:])
		sig_T.append(sig)
		noise_T.append(noise)
		sig = np.mean(s24[i][1000:])
		noise = np.std(s24[i][1000:])
		sig_F.append(sig)
		noise_F.append(noise)
	for i in range(len(s15)):
		sig = np.mean(s15[i][1000:])
		noise = np.std(s15[i][1000:])
		sig_T.append(sig)
		noise_T.append(noise)
		sig = np.mean(s25[i][1000:])
		noise = np.std(s25[i][1000:])
		sig_F.append(sig)
		noise_F.append(noise)
	for i in range(len(sdt)):
		sig = np.mean(sdt[i][1000:])
		dark_T.append(sig)
		sig = np.mean(sdf[i][1000:])
		dark_F.append(sig)
	
	# Signals in Volts
	# Let's assume that the Femto signal is correct and the V/W conversion is too.
	# So, there is no need to use the absolute Thorlabs signal.
	# The Thorlabs signal is only used for calcualting the noise.
	sig_T_raw = np.asarray(sig_T)
	sig_T_dark = dark_T[0]
	sig_T = np.asarray(sig_T) - np.asarray(dark_T)
	noise_T = np.asarray(noise_T)
	snr_T = sig_T / noise_T

	sig_F_raw = np.asarray(sig_F)
	sig_F_dark = dark_F[0]
	sig_F = np.asarray(sig_F) - np.asarray(dark_F)
	noise_F = np.asarray(noise_F)
	snr_F = sig_F / noise_F
	# Signals in Watts
	Gain_F = 1E6								# V/W
	Gain_T_VA = 0.75E5							# V/A
	Gain_T_AW =  0.85							# A/W
	Amp_VV = 10.0								# Voltage amplification
	Gain_T = Gain_T_VA * Gain_T_AW * Amp_VV 	# V/W
	sig_T_W = sig_T / Gain_T
	noise_T_W = np.asarray(noise_T) / Gain_T
	sig_F_W = sig_F / Gain_F
	noise_F_W = noise_F / Gain_F

	# Convert to dBm
	sig_T_dBm = 10.0 * np.log10(1e3*sig_T_W)
	sig_F_dBm = 10.0 * np.log10(1e3*sig_F_W)


	#Now, let's model the noise
	# noise = sqrt(signal + readoutnoise + darknoise)
	# From doing measurements at different Ranges we an estimate - very roughly -
	# that the dark noise of the Thorlabs detector is 1.24 mV and of the Femto detector 0.15 mV
	# And that the dark signal of the Thorlabs is 139 mV and of the Femto less than 0.1 mV
	### Note that these measurements were done with a +/- 4V range.

	#Thorlabs
	dark_noise_T = 1.26e-3 								# V
	dark_noise_T_W = dark_noise_T / Gain_T 				# W
	read_out_noise_T = 1.28e-3							# V
	read_out_noise_T_W = read_out_noise_T / Gain_T	# W
	sig_CF_T = 2.4e-7									# Scaling factor ; Amp to electrons TBD.
	sim_noise_T = np.sqrt(sig_T*sig_CF_T + dark_noise_T**2 + read_out_noise_T**2 )	# V
	sim_noise_T_W = np.sqrt(sig_T*sig_CF_T + dark_noise_T**2 + read_out_noise_T**2) / Gain_T	# W
	
	
	# different factors
	sim_noise_photon_T    = np.sqrt(sig_T*sig_CF_T)		# V
	sim_noise_photon_T_W  = sim_noise_photon_T / Gain_T	# W
	sim_noise_dark_T      = dark_noise_T				# V
	sim_noise_dark_T_W    = dark_noise_T / Gain_T		# W
	sim_noise_readout_T   = read_out_noise_T			# V
	sim_noise_readout_T_W = read_out_noise_T / Gain_T	# W


	# Femto
	dark_noise_F = 1.28e-4 								# V
	dark_noise_F_W = dark_noise_F / Gain_F 				# W
	read_out_noise_F = 8.7e-4							# V
	read_out_noise_F_W = read_out_noise_F / Gain_F	# W
	sig_CF_F = 2.4e-7									# Scaling factor ; Amp to electrons TBD.
	sim_noise_F = np.sqrt(sig_F*sig_CF_F + dark_noise_F**2 + read_out_noise_F**2 )	# V
	sim_noise_F_W = np.sqrt(sig_F*sig_CF_F + dark_noise_F**2 + read_out_noise_F**2) / Gain_F	# W
	# different factors
	sim_noise_photon_F    = np.sqrt(sig_F*sig_CF_F)		# V
	sim_noise_photon_F_W  = sim_noise_photon_F / Gain_F	# W
	sim_noise_dark_F      = dark_noise_F				# V
	sim_noise_dark_F_W    = dark_noise_F / Gain_F		# W
	sim_noise_readout_F   = read_out_noise_F			# V
	sim_noise_readout_F_W = read_out_noise_F / Gain_F	# W


	# Thorlabs
	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	#plt.plot(sig_T_dBm[1:], photon_signal_T[1:], 'o-',label="photon signal")
	#plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm)], [dark_signal_T,dark_signal_T], 'o-',label="dark signal")
	plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm)], [sig_T_dark,sig_T_dark], 'o-',label="dark signal")
	plt.plot(sig_T_dBm[1:], sig_T[1:], 'o-',label="photon signal")
	plt.plot(sig_T_dBm[1:], sig_T_raw[1:], 'd-', label='total_signal')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal at detetor [V]')
	plt.yscale('log')
	plt.ylim([1e-2,1e1])
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Sig_Thorlabs_dBm_feb28.png',dpi=300)



	# Femto
	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	#plt.plot(sig_T_dBm[1:], photon_signal_T[1:], 'o-',label="photon signal")
	#plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm)], [dark_signal_T,dark_signal_T], 'o-',label="dark signal")
	plt.plot([np.min(sig_F_dBm[1:]),np.max(sig_F_dBm)], [sig_F_dark,sig_F_dark], 'o-',label="dark signal")
	plt.plot(sig_F_dBm[1:], sig_F[1:], 'o-',label="photon signal")
	plt.plot(sig_F_dBm[1:], sig_F_raw[1:], 'd-', label='total_signal')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal at detetor [V]')
	plt.yscale('log')
	#plt.ylimit([1e-2,1e1])
	plt.title('Femto')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Sig_Femto_dBm_feb28.png',dpi=300)



#	sys.exit()


	plt.close()
	plot_scale_V = 1e3
	plot_scale_W = 1e9
	plt.plot(plot_scale_W*sig_T_W[1:], plot_scale_V*noise_T[1:],'o',label="total noise")
	plt.plot(plot_scale_W*sig_T_W[1:], plot_scale_V*sim_noise_T[1:],label="simulated noise")
	plt.plot(plot_scale_W*sig_T_W[1:], plot_scale_V*sim_noise_photon_T[1:],label="photon noise")
	plt.plot([np.min(plot_scale_W*sig_T_W[1:]),np.max(plot_scale_W*sig_T_W[1:])], [plot_scale_V*sim_noise_dark_T,plot_scale_V*sim_noise_dark_T],label="dark noise")
	plt.plot([np.min(plot_scale_W*sig_T_W[1:]),np.max(plot_scale_W*sig_T_W[1:])], [plot_scale_V*sim_noise_readout_T,plot_scale_V*sim_noise_readout_T],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Thorlabs_feb28.png',dpi=300)


	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	plt.plot(sig_T_dBm[1:], plot_scale_V*noise_T[1:],'o',label="total noise")
	plt.plot(sig_T_dBm[1:], plot_scale_V*sim_noise_T[1:],label="simulated noise")
	plt.plot(sig_T_dBm[1:], plot_scale_V*sim_noise_photon_T[1:],label="photon noise")
	plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm[1:])], [plot_scale_V*sim_noise_dark_T,plot_scale_V*sim_noise_dark_T],label="dark noise")
	plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm[1:])], [plot_scale_V*sim_noise_readout_T,plot_scale_V*sim_noise_readout_T],label="read-out noise")
	#plt.xscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Thorlabs_dBm_feb28.png',dpi=300)


	plt.close()
	plot_scale = 1e9
	plt.plot(plot_scale*sig_T_W[1:], plot_scale*noise_T_W[1:],'o',label="total noise")
	plt.plot(plot_scale*sig_T_W[1:], plot_scale*sim_noise_T_W[1:],label="simulated noise")
	plt.plot(plot_scale*sig_T_W[1:], plot_scale*sim_noise_photon_T_W[1:],label="photon noise")
	plt.plot([np.min(plot_scale*sig_T_W[1:]),np.max(plot_scale*sig_T_W[1:])], [plot_scale*sim_noise_dark_T_W,plot_scale*sim_noise_dark_T_W],label="dark noise")
	plt.plot([np.min(plot_scale*sig_T_W[1:]),np.max(plot_scale*sig_T_W[1:])], [plot_scale*sim_noise_readout_T_W,plot_scale*sim_noise_readout_T_W],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [nW]')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Thorlabs_W_feb28.png',dpi=300)

	
	plt.close()
	plot_scale_W = 1e9
	#plt.plot(plot_scale_W*sig_F_W[1:], snr_T[1:],'o')
	plt.plot(plot_scale_W*sig_T_W[1:], snr_T[1:],'o')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('SNR')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	#plt.legend()
	plt.show()
	plt.savefig('SNR_Thorlabs_feb28.png',dpi=300)



	#### FEMTO

	plt.close()
	plot_scale_V = 1e3
	plot_scale_W = 1e9
	plt.plot(plot_scale_W*sig_F_W[:], plot_scale_V*noise_F[:],'o',label="total noise")
	plt.plot(plot_scale_W*sig_F_W[:], plot_scale_V*sim_noise_F[:],label="simulated noise")
	plt.plot(plot_scale_W*sig_F_W[:], plot_scale_V*sim_noise_photon_F[:],label="photon noise")
	plt.plot([np.min(plot_scale_W*sig_F_W[:]),np.max(plot_scale_W*sig_F_W[:])], [plot_scale_V*sim_noise_dark_F,plot_scale_V*sim_noise_dark_F],label="dark noise")
	plt.plot([np.min(plot_scale_W*sig_F_W[:]),np.max(plot_scale_W*sig_F_W[:])], [plot_scale_V*sim_noise_readout_F,plot_scale_V*sim_noise_readout_F],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [mV]')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Femto_feb28.png',dpi=300)

	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	plt.plot(sig_F_dBm[:], plot_scale_V*noise_F[:],'o',label="total noise")
	plt.plot(sig_F_dBm[:], plot_scale_V*sim_noise_F[:],label="simulated noise")
	plt.plot(sig_F_dBm[:], plot_scale_V*sim_noise_photon_F[:],label="photon noise")
	plt.plot([np.min(sig_F_dBm[:]),np.max(sig_F_dBm[:])], [plot_scale_V*sim_noise_dark_F,plot_scale_V*sim_noise_dark_F],label="dark noise")
	plt.plot([np.min(sig_F_dBm[:]),np.max(sig_F_dBm[:])], [plot_scale_V*sim_noise_readout_F,plot_scale_V*sim_noise_readout_F],label="read-out noise")
	#plt.xscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Noise [mV]')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Femto_dBm_feb28.png',dpi=300)


	plt.close()
	plot_scale = 1e9
	plt.plot(plot_scale*sig_F_W[:], plot_scale*noise_F_W[:],'o',label="total noise")
	plt.plot(plot_scale*sig_F_W[:], plot_scale*sim_noise_F_W[:],label="simulated noise")
	plt.plot(plot_scale*sig_F_W[:], plot_scale*sim_noise_photon_F_W[:],label="photon noise")
	plt.plot([np.min(plot_scale*sig_F_W[:]),np.max(plot_scale*sig_F_W[:])], [plot_scale*sim_noise_dark_F_W,plot_scale*sim_noise_dark_F_W],label="dark noise")
	plt.plot([np.min(plot_scale*sig_F_W[:]),np.max(plot_scale*sig_F_W[:])], [plot_scale*sim_noise_readout_F_W,plot_scale*sim_noise_readout_F_W],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [nW]')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Femto_W_feb28.png',dpi=300)


	plt.close()
	plot_scale_W = 1e9
	plt.plot(plot_scale_W*sig_F_W[:], snr_F[:],'o')
	#plt.plot(plot_scale_W*sig_T_W[1:], snr_T[1:],'o')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('SNR')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	#plt.legend()
	plt.show()
	plt.savefig('SNR_Femto_feb28.png',dpi=300)


def plot_snr_22feb_DELTA():
	# Read data
	# Thorlabs is Ch1
	# Femto is Ch2




	# Read data
	t_T1, s_T1 = mytools.load_data('22feb_ch1',70,79)
	t_F1, s_F1 = mytools.load_data('22feb_ch2',70,79)
	t_T2, s_T2 = mytools.load_data('22feb_ch1',81,91)
	t_F2, s_F2 = mytools.load_data('22feb_ch2',81,91)
	t_T = np.concatenate((t_T1, t_T2))
	s_T = np.concatenate((s_T1, s_T2))
	t_F = np.concatenate((t_F1, t_F2))
	s_F = np.concatenate((s_F1, s_F2))


	F_gain = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, \
			  1e8, 1e8, 1e8, 1e8, 1e8, \
			  1e7, 1e7, 1e7, 1e7, 1e7, \
			  1e6, 1e6, 1e6]

	sig_T = []
	sig_F = []
	noise_T = []
	noise_F = []
	for i in range(1,20):
		# Thorlabs
		mean = np.mean(s_T[i][1000:])
		noise = np.std(s_T[i][1000:])
		sig_T.append(mean)
		noise_T.append(noise)
		# Femto
		mean = np.mean(s_F[i][1000:])
		noise = np.std(s_F[i][1000:])
		sig_F.append(mean)
		noise_F.append(noise)

	print noise_T
	sys.exit()


	dark_T = np.mean(s_T[0])
	dark_F = np.mean(s_F[20])

	sig_T = np.asarray(sig_T)
	sig_F = np.asarray(sig_F)
	noise_T = np.asarray(noise_T)
	noise_F = np.asarray(noise_F)
	dark_T = np.asarray(dark_T)
	dark_F = np.asarray(dark_F)

	# Convert to dBm
	#F_mean_H_dBm = 10.0 * np.log10(1e3*F_mean_H)
	#F_mean_dBm = 10.0 * np.log10(1e3*F_mean)


	# Signals in Volts
	# Let's assume that the Femto signal is correct and the V/W conversion is too.
	# So, there is no need to use the absolute Thorlabs signal.
	# The Thorlabs signal is only used for calcualting the noise.
	sig_T_raw = np.asarray(sig_T)
	sig_T_dark = dark_T
	sig_T = np.asarray(sig_T) - np.asarray(dark_T)
	noise_T = np.asarray(noise_T)
	snr_T = sig_T / noise_T

	sig_F_raw = np.asarray(sig_F)
	sig_F_dark = dark_F
	sig_F = np.asarray(sig_F) - np.asarray(dark_F)
	noise_F = np.asarray(noise_F)
	snr_F = sig_F / noise_F

	# Signals in Watts
	Gain_F = 1E6								# V/W
	Gain_T_VA = 0.75E5							# V/A
	Gain_T_AW =  0.85							# A/W
	Amp_VV = 10.0								# Voltage amplification
	Gain_T = Gain_T_VA * Gain_T_AW * Amp_VV 	# V/W
	sig_T_W = sig_T / Gain_T
	noise_T_W = np.asarray(noise_T) / Gain_T
	sig_F_W = sig_F / Gain_F
	noise_F_W = noise_F / Gain_F

	# Convert to dBm
	sig_T_dBm = 10.0 * np.log10(1e3*sig_T_W)
	sig_F_dBm = 10.0 * np.log10(1e3*sig_F_W)


	#Now, let's model the noise
	# noise = sqrt(signal + readoutnoise + darknoise)
	# From doing measurements at different Ranges we an estimate - very roughly -
	# that the dark noise of the Thorlabs detector is 1.24 mV and of the Femto detector 0.15 mV
	# And that the dark signal of the Thorlabs is 139 mV and of the Femto less than 0.1 mV
	### Note that these measurements were done with a +/- 4V range.

	#Thorlabs
	dark_noise_T = 1.26e-3 								# V
	dark_noise_T_W = dark_noise_T / Gain_T 				# W
	read_out_noise_T = 1.28e-3							# V
	read_out_noise_T_W = read_out_noise_T / Gain_T	# W
	sig_CF_T = 2.4e-7									# Scaling factor ; Amp to electrons TBD.
	sim_noise_T = np.sqrt(sig_T*sig_CF_T + dark_noise_T**2 + read_out_noise_T**2 )	# V
	sim_noise_T_W = np.sqrt(sig_T*sig_CF_T + dark_noise_T**2 + read_out_noise_T**2) / Gain_T	# W
	
	
	# different factors
	sim_noise_photon_T    = np.sqrt(sig_T*sig_CF_T)		# V
	sim_noise_photon_T_W  = sim_noise_photon_T / Gain_T	# W
	sim_noise_dark_T      = dark_noise_T				# V
	sim_noise_dark_T_W    = dark_noise_T / Gain_T		# W
	sim_noise_readout_T   = read_out_noise_T			# V
	sim_noise_readout_T_W = read_out_noise_T / Gain_T	# W


	# Femto
	dark_noise_F = 1.28e-4 								# V
	dark_noise_F_W = dark_noise_F / Gain_F 				# W
	read_out_noise_F = 8.7e-4							# V
	read_out_noise_F_W = read_out_noise_F / Gain_F	# W
	sig_CF_F = 2.4e-7									# Scaling factor ; Amp to electrons TBD.
	sim_noise_F = np.sqrt(sig_F*sig_CF_F + dark_noise_F**2 + read_out_noise_F**2 )	# V
	sim_noise_F_W = np.sqrt(sig_F*sig_CF_F + dark_noise_F**2 + read_out_noise_F**2) / Gain_F	# W
	# different factors
	sim_noise_photon_F    = np.sqrt(sig_F*sig_CF_F)		# V
	sim_noise_photon_F_W  = sim_noise_photon_F / Gain_F	# W
	sim_noise_dark_F      = dark_noise_F				# V
	sim_noise_dark_F_W    = dark_noise_F / Gain_F		# W
	sim_noise_readout_F   = read_out_noise_F			# V
	sim_noise_readout_F_W = read_out_noise_F / Gain_F	# W


	# Thorlabs
	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	#plt.plot(sig_T_dBm[1:], photon_signal_T[1:], 'o-',label="photon signal")
	#plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm)], [dark_signal_T,dark_signal_T], 'o-',label="dark signal")
	plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm)], [sig_T_dark,sig_T_dark], 'o-',label="dark signal")
	plt.plot(sig_T_dBm[1:], sig_T[1:], 'o-',label="photon signal")
	plt.plot(sig_T_dBm[1:], sig_T_raw[1:], 'd-', label='total_signal')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal at detetor [V]')
	plt.yscale('log')
	plt.ylim([1e-2,1e1])
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Sig_Thorlabs_dBm_feb22_V2.png',dpi=300)



	# Femto
	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	#plt.plot(sig_T_dBm[1:], photon_signal_T[1:], 'o-',label="photon signal")
	#plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm)], [dark_signal_T,dark_signal_T], 'o-',label="dark signal")
	plt.plot([np.min(sig_F_dBm[1:]),np.max(sig_F_dBm)], [sig_F_dark,sig_F_dark], 'o-',label="dark signal")
	plt.plot(sig_F_dBm[1:], sig_F[1:], 'o-',label="photon signal")
	plt.plot(sig_F_dBm[1:], sig_F_raw[1:], 'd-', label='total_signal')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Signal at detetor [V]')
	plt.yscale('log')
	#plt.ylimit([1e-2,1e1])
	plt.title('Femto')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Sig_Femto_dBm_feb22_V2.png',dpi=300)



#	sys.exit()


	plt.close()
	plot_scale_V = 1e3
	plot_scale_W = 1e9
	plt.plot(plot_scale_W*sig_T_W[1:], plot_scale_V*noise_T[1:],'o',label="total noise")
	plt.plot(plot_scale_W*sig_T_W[1:], plot_scale_V*sim_noise_T[1:],label="simulated noise")
	plt.plot(plot_scale_W*sig_T_W[1:], plot_scale_V*sim_noise_photon_T[1:],label="photon noise")
	plt.plot([np.min(plot_scale_W*sig_T_W[1:]),np.max(plot_scale_W*sig_T_W[1:])], [plot_scale_V*sim_noise_dark_T,plot_scale_V*sim_noise_dark_T],label="dark noise")
	plt.plot([np.min(plot_scale_W*sig_T_W[1:]),np.max(plot_scale_W*sig_T_W[1:])], [plot_scale_V*sim_noise_readout_T,plot_scale_V*sim_noise_readout_T],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Thorlabs_feb22_V2.png',dpi=300)


	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	plt.plot(sig_T_dBm[1:], plot_scale_V*noise_T[1:],'o',label="total noise")
	plt.plot(sig_T_dBm[1:], plot_scale_V*sim_noise_T[1:],label="simulated noise")
	plt.plot(sig_T_dBm[1:], plot_scale_V*sim_noise_photon_T[1:],label="photon noise")
	plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm[1:])], [plot_scale_V*sim_noise_dark_T,plot_scale_V*sim_noise_dark_T],label="dark noise")
	plt.plot([np.min(sig_T_dBm[1:]),np.max(sig_T_dBm[1:])], [plot_scale_V*sim_noise_readout_T,plot_scale_V*sim_noise_readout_T],label="read-out noise")
	#plt.xscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Thorlabs_dBm_feb22_V2.png',dpi=300)


	plt.close()
	plot_scale = 1e9
	plt.plot(plot_scale*sig_T_W[1:], plot_scale*noise_T_W[1:],'o',label="total noise")
	plt.plot(plot_scale*sig_T_W[1:], plot_scale*sim_noise_T_W[1:],label="simulated noise")
	plt.plot(plot_scale*sig_T_W[1:], plot_scale*sim_noise_photon_T_W[1:],label="photon noise")
	plt.plot([np.min(plot_scale*sig_T_W[1:]),np.max(plot_scale*sig_T_W[1:])], [plot_scale*sim_noise_dark_T_W,plot_scale*sim_noise_dark_T_W],label="dark noise")
	plt.plot([np.min(plot_scale*sig_T_W[1:]),np.max(plot_scale*sig_T_W[1:])], [plot_scale*sim_noise_readout_T_W,plot_scale*sim_noise_readout_T_W],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [nW]')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Thorlabs_W_feb22_V2.png',dpi=300)

	
	plt.close()
	plot_scale_W = 1e9
	#plt.plot(plot_scale_W*sig_F_W[1:], snr_T[1:],'o')
	plt.plot(plot_scale_W*sig_T_W[1:], snr_T[1:],'o')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('SNR')
	plt.title('Thorlabs PDA30B2 - gain 40dB  - AMP 10V/V')
	#plt.legend()
	plt.show()
	plt.savefig('SNR_Thorlabs_feb22_V2.png',dpi=300)



	#### FEMTO

	plt.close()
	plot_scale_V = 1e3
	plot_scale_W = 1e9
	plt.plot(plot_scale_W*sig_F_W[:], plot_scale_V*noise_F[:],'o',label="total noise")
	plt.plot(plot_scale_W*sig_F_W[:], plot_scale_V*sim_noise_F[:],label="simulated noise")
	plt.plot(plot_scale_W*sig_F_W[:], plot_scale_V*sim_noise_photon_F[:],label="photon noise")
	plt.plot([np.min(plot_scale_W*sig_F_W[:]),np.max(plot_scale_W*sig_F_W[:])], [plot_scale_V*sim_noise_dark_F,plot_scale_V*sim_noise_dark_F],label="dark noise")
	plt.plot([np.min(plot_scale_W*sig_F_W[:]),np.max(plot_scale_W*sig_F_W[:])], [plot_scale_V*sim_noise_readout_F,plot_scale_V*sim_noise_readout_F],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [mV]')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Femto_feb22_V2.png',dpi=300)

	plt.close()
	plot_scale_V = 1e3
	#plot_scale_W = 1e9
	plt.plot(sig_F_dBm[:], plot_scale_V*noise_F[:],'o',label="total noise")
	plt.plot(sig_F_dBm[:], plot_scale_V*sim_noise_F[:],label="simulated noise")
	plt.plot(sig_F_dBm[:], plot_scale_V*sim_noise_photon_F[:],label="photon noise")
	plt.plot([np.min(sig_F_dBm[:]),np.max(sig_F_dBm[:])], [plot_scale_V*sim_noise_dark_F,plot_scale_V*sim_noise_dark_F],label="dark noise")
	plt.plot([np.min(sig_F_dBm[:]),np.max(sig_F_dBm[:])], [plot_scale_V*sim_noise_readout_F,plot_scale_V*sim_noise_readout_F],label="read-out noise")
	#plt.xscale('log')
	plt.xlabel('Signal at detector [dBm]')
	plt.ylabel('Noise [mV]')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Femto_dBm_feb22_V2.png',dpi=300)


	plt.close()
	plot_scale = 1e9
	plt.plot(plot_scale*sig_F_W[:], plot_scale*noise_F_W[:],'o',label="total noise")
	plt.plot(plot_scale*sig_F_W[:], plot_scale*sim_noise_F_W[:],label="simulated noise")
	plt.plot(plot_scale*sig_F_W[:], plot_scale*sim_noise_photon_F_W[:],label="photon noise")
	plt.plot([np.min(plot_scale*sig_F_W[:]),np.max(plot_scale*sig_F_W[:])], [plot_scale*sim_noise_dark_F_W,plot_scale*sim_noise_dark_F_W],label="dark noise")
	plt.plot([np.min(plot_scale*sig_F_W[:]),np.max(plot_scale*sig_F_W[:])], [plot_scale*sim_noise_readout_F_W,plot_scale*sim_noise_readout_F_W],label="read-out noise")
	plt.xscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('Noise [nW]')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	plt.legend()
	plt.show()
	plt.savefig('Sig_vs_Noise_Femto_W_feb22_V2.png',dpi=300)


	plt.close()
	plot_scale_W = 1e9
	plt.plot(plot_scale_W*sig_F_W[:], snr_F[:],'o')
	#plt.plot(plot_scale_W*sig_T_W[1:], snr_T[1:],'o')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$\mathrm{Signal\ at\ detector}\ [nW]$')
	plt.ylabel('SNR')
	plt.title('Femto OE-200-IN2  LowNoise gain 1E6')
	#plt.legend()
	plt.show()
	plt.savefig('SNR_Femto_feb22_V2.png',dpi=300)



def plot_low_sig():

	t11, s11 = mytools.load_data('27feb_PM_ch1',103,107)
	t21, s21 = mytools.load_data('27feb_PM_ch2',103,107)

	fs = 20000


	# Femto
	ts0 = t11[0][:30000]	# 2 second
	yraw0 = s11[0][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter0 = scipy.signal.lfilter(b, a, yraw0)

	ts1 = t11[1][:30000]	# 2 second
	yraw1 = s11[1][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter1 = scipy.signal.lfilter(b, a, yraw1)

	ts2 = t11[2][:30000]	# 2 second
	yraw2 = s11[2][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter2 = scipy.signal.lfilter(b, a, yraw2)

	ts3 = t11[3][:30000]	# 2 second
	yraw3 = s11[3][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter3 = scipy.signal.lfilter(b, a, yraw3)

	ts4 = t11[4][:30000]	# 2 second
	yraw4 = s11[4][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter4 = scipy.signal.lfilter(b, a, yraw4)

	m0 = np.mean(yraw0)
	m1 = np.mean(yraw1)
	m2 = np.mean(yraw2)
	m3 = np.mean(yraw3)
	m4 = np.mean(yraw4)

	print 'Thorlabs'
	print (m0-m4)*1e9/1e6, ' nW' 
	print (m1-m4)*1e9/1e6, ' nW' 
	print (m2-m4)*1e9/1e6, ' nW' 
	print (m3-m4)*1e9/1e6, ' nW' 

	plt.close()
	#plt.plot(ts0,yraw0)
	plt.plot(ts0[2000:],y_lfilter0[2000:],label='0')
	#plt.plot(ts1,yraw1)
	plt.plot(ts1[2000:],y_lfilter1[2000:],label='1')
	#plt.plot(ts2,yraw2)
	plt.plot(ts2[2000:],y_lfilter2[2000:],label='2')
	#plt.plot(ts3,yraw3)
	plt.plot(ts3[2000:],y_lfilter3[2000:],label='3')
	#plt.plot(ts4,yraw4)
	plt.plot(ts4[2000:],y_lfilter4[2000:],label='4')
	plt.legend()
	plt.show()


	'''

	# Femto
	ts0 = t21[0][:30000]	# 2 second
	yraw0 = s21[0][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter0 = scipy.signal.lfilter(b, a, yraw0)

	ts1 = t21[1][:30000]	# 2 second
	yraw1 = s21[1][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter1 = scipy.signal.lfilter(b, a, yraw1)

	ts2 = t21[2][:30000]	# 2 second
	yraw2 = s21[2][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter2 = scipy.signal.lfilter(b, a, yraw2)

	ts3 = t21[3][:30000]	# 2 second
	yraw3 = s21[3][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter3 = scipy.signal.lfilter(b, a, yraw3)

	ts4 = t21[4][:30000]	# 2 second
	yraw4 = s21[4][:30000]
	b, a = scipy.signal.iirfilter(2, Wn=12.0, fs=fs, btype="low", ftype="butter")
	y_lfilter4 = scipy.signal.lfilter(b, a, yraw4)

	m0 = np.mean(yraw0)
	m1 = np.mean(yraw1)
	m2 = np.mean(yraw2)
	m3 = np.mean(yraw3)
	m4 = np.mean(yraw4)

	print 'Femto'
	print (m0-m4)*1e9/1e6, ' nW' 
	print (m1-m4)*1e9/1e6, ' nW' 
	print (m2-m4)*1e9/1e6, ' nW' 
	print (m3-m4)*1e9/1e6, ' nW' 



	plt.close()
	#plt.plot(ts0,yraw0)
	plt.plot(ts0,y_lfilter0,label='0')
	#plt.plot(ts1,yraw1)
	plt.plot(ts1,y_lfilter1,label='1')
	#plt.plot(ts2,yraw2)
	plt.plot(ts2,y_lfilter2,label='2')
	#plt.plot(ts3,yraw3)
	plt.plot(ts3,y_lfilter3,label='3')
	#plt.plot(ts4,yraw4)
	plt.plot(ts4,y_lfilter4,label='4')
	plt.legend()
	plt.show()

	'''

def noise_1_mrt():

	# FEMTO
	Vrange = np.asarray([0.2, 0.4, 0.8, 2, 4, 8])
	gain_F = np.asarray([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
	gain_T = np.asarray([0, 10, 20, 30, 40, 50, 60, 70])
	noise_F = np.asarray([  \
		[0.07,	0.10,	0.16,	0.6,	0.9,	1.6], \
		[0.07,	0.10,	0.16,	0.6,	0.9,	1.6], \
		[0.09,	0.11,	0.17,	0.6,	0.9,	1.6], \
		[0.16,	0.17,	0.22,	0.7,	0.9,	1.6], \
		[0.41,	0.43,	0.46,	0.8,	1.0,	1.6], \
		[0.61,	0.61,	0.63,	0.9,	1.1,	1.7], \
		[0.93,	0.93,	0.96,	1.1,	1.3,	1.9] ])
	noise_T = np.asarray([  \
		[0.49,	0.50,	0.53,	1.0,	1.4,	2.4], \
		[1.05,	1.05,	1.07,	1.4,	1.7,	2.5], \
		[1.03,	1.03,	1.05,	1.3,	1.7,	2.5], \
		[1.13,	1.14,	1.15,	1.4,	1.7,	2.6], \
		[1.23,	1.24,	1.24,	1.5,	1.8,	2.6], \
		[1.47,	1.49,	1.49,	1.7,	2.0,	2.7], \
		[2.3,	2.3,	2.3,	2.6,	2.7,	3.4], \
		[6,   	6,		6,		6,		6,		6] ])
	noise_T_noAMP = np.asarray([  \
		[0.20,	0.23,	0.30,	0.9,	1.3], \
		[0.24,	0.26,	0.32,	0.9,	1.3], \
		[0.21,	0.23,	0.30,	0.9,	1.3], \
		[0.21,	0.23,	0.30,	0.9,	1.3], \
		[0.22,	0.24,	0.30,	0.9,	1.3], \
		[0.23,	0.25,	0.31,	0.9,	1.3], \
		[0.29,	0.29,	0.38,	0.9,	1.4], \
		[0.56, 	0.54,	0.64,	1.1,	1.5] ])

	# This is a set of measurments of the noise of the two channels, nothing atttached, as function of readout range.
	# Range is from 0.2V to 20V
	noise_no_cables = np.asarray([ 	[0.10, 0.13, 0.23, 0.9, 1.4, 2.4, 8], \
									[0.17, 0.18, 0.23, 0.7, 0.9, 1.6, 6.0]])
	# Note: this is a measurment where the detectors ar turned off (switch in Thorlabs, unplugged power cable for Femto)
	# Note: the firt two measurements (range 0.2V and 0.4V) cause saturation. Clearly this is caused by the Amplifier, as it was stil attached.
	# Range is from 0.2V to 20V
	# Note: interestingly: the noise is lower when the detectors are switched on than when nothing is attached.
	# Note, the first two ZERO readings are due to saturation - no change in signal means sigma=0
	noise_det_off = np.asarray([[0,0,1.46, 1.6, 1.9, 2.6, 8], \
								[0.07, 0.09, 0.16, 0.7, 0.9, 1.6, 6.0]])

	Vrange_20 = np.asarray([0.2, 0.4, 0.8, 2, 4, 8, 20])

	
	femto_noise_spec_W = np.asarray([23E-9, 2.8E-9, 650E-12, 180E-12, 51E-12, 7.5E-12, 1.1E-12])
	femto_noise_V = femto_noise_spec_W * gain_F
	femto_noise_mV = femto_noise_V * 1e3

	print femto_noise_mV
	
	#channel difference
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange_20, noise_no_cables[0,:],'o-',label='no cables: CH1')
	plt.plot(Vrange_20, noise_no_cables[1,:],'d-',label='no cables: CH2')
	plt.plot(Vrange_20[2:], noise_det_off[0,2:],'<-',label='detector OFF: CH1')
	plt.plot(Vrange_20, noise_det_off[1,:],'s-',label='detector OFF: CH2')
	#plt.plot(Vrange, noise_F[0,:],'^-', label='Femto; gain 1E3')
	#plt.plot(Vrange, noise_T[0,:],'p-', label='Thorlabs; gain 0dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Meas. range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Noise Measurements TiePie')
	plt.show()
	plt.savefig('Channel_noise.png',dpi=300)
	
	#channel difference
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange_20, noise_no_cables[0,:]/Vrange_20,'o-',label='no cables: CH1')
	plt.plot(Vrange_20, noise_no_cables[1,:]/Vrange_20,'d-',label='no cables: CH2')
	plt.plot(Vrange_20[2:], noise_det_off[0,2:]/Vrange_20[2:],'<-',label='detector OFF: CH1')
	plt.plot(Vrange_20, noise_det_off[1,:]/Vrange_20,'s-',label='detector OFF: CH2')
	#plt.plot(Vrange, noise_F[0,:],'^-', label='Femto; gain 1E3')
	#plt.plot(Vrange, noise_T[0,:],'p-', label='Thorlabs; gain 0dB')
	plt.xscale('log')
	#plt.yscale('log')
	plt.legend()
	plt.xlabel('Meas. range [V]')
	plt.ylabel('Noise / Meas. range [mV / V]')
	plt.title('Noise Measurements TiePie')
	plt.show()
	plt.savefig('Channel_noise_relative.png',dpi=300)
	
	# Femto
	plt.close()
	plt.figure(constrained_layout=True)
	#plt.plot(Vrange_20, noise_no_cables[0,:],'o-',label='no cables: CH1')
	plt.plot(Vrange_20, noise_no_cables[1,:],'d-',label='no cables: CH2')
	#plt.plot(range_20[2:], noise_det_off[0,2:],'<-',label='detector OFF: CH1')
	plt.plot(Vrange_20, noise_det_off[1,:],'s-',label='detector OFF: CH2')
	plt.plot(Vrange, noise_F[0,:],'^-', label='Femto dark; gain 1E3')
	plt.plot(Vrange, noise_F[3,:],'^-', label='Femto dark; gain 1E6')
	plt.plot(Vrange, noise_F[6,:],'^-', label='Femto dark; gain 1E9')
	#plt.plot(Vrange, noise_T[0,:],'p-', label='Thorlabs; gain 0dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Noise Measurements Femto')
	plt.show()
	plt.savefig('Femto_noise.png',dpi=300)
	
	plt.close()
	plt.figure(constrained_layout=True)
	#plt.plot(Vrange_20, noise_no_cables[0,:],'o-',label='no cables: CH1')
	plt.plot(Vrange_20, noise_no_cables[1,:]/Vrange_20,'d-',label='no cables: CH2')
	#plt.plot(range_20[2:], noise_det_off[0,2:],'<-',label='detector OFF: CH1')
	plt.plot(Vrange_20, noise_det_off[1,:]/Vrange_20,'s-',label='detector OFF: CH2')
	plt.plot(Vrange, noise_F[0,:]/Vrange,'^-', label='Femto dark; gain 1E3')
	plt.plot(Vrange, noise_F[3,:]/Vrange,'^-', label='Femto dark; gain 1E6')
	plt.plot(Vrange, noise_F[6,:]/Vrange,'^-', label='Femto dark; gain 1E9')
	#plt.plot(Vrange, noise_T[0,:],'p-', label='Thorlabs; gain 0dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Readout range [V]')
	plt.ylabel('Noise  / Meas. range [mV/V]')
	plt.title('Noise Measurements Femto')
	plt.show()
	plt.savefig('Femto_noise_relative.png',dpi=300)
	
	# thorlabs
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange_20, noise_no_cables[0,:],'o-',label='no cables: CH1')
	#plt.plot(Vrange_20, noise_no_cables[1,:],'d-',label='no cables: CH2')
	plt.plot(Vrange_20[2:], noise_det_off[0,2:],'<-',label='detector OFF: CH1')
	#plt.plot(Vrange_20, noise_det_off[1,:],'s-',label='detector OFF: CH2')
	#plt.plot(Vrange, noise_F[0,:],'^-', label='Femto; gain 1E3')
	plt.plot(Vrange, noise_T[0,:],'p-', label='Thorlabs dark; gain 0dB')
	plt.plot(Vrange, noise_T[2,:],'p-', label='Thorlabs dark; gain 20dB')
	plt.plot(Vrange, noise_T[4,:],'p-', label='Thorlabs dark; gain 40dB')
	plt.plot(Vrange, noise_T[6,:],'p-', label='Thorlabs dark; gain 60dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Noise Measurements Thorlabs + AMP 10 V/V')
	plt.show()
	plt.savefig('Thorlabs_noise_amp10.png',dpi=300)
	
	# thorlabs no AMP
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange_20, noise_no_cables[0,:],'o-',label='no cables: CH1')
	#plt.plot(Vrange_20, noise_no_cables[1,:],'d-',label='no cables: CH2')
	plt.plot(Vrange_20[2:], noise_det_off[0,2:],'<-',label='detector OFF: CH1')
	#plt.plot(Vrange_20, noise_det_off[1,:],'s-',label='detector OFF: CH2')
	#plt.plot(Vrange, noise_F[0,:],'^-', label='Femto; gain 1E3')
	plt.plot(Vrange[:-1], noise_T_noAMP[0,:],'p-', label='Thorlabs dark; gain 0dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[2,:],'p-', label='Thorlabs dark; gain 20dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[4,:],'p-', label='Thorlabs dark; gain 40dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[6,:],'p-', label='Thorlabs dark; gain 60dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Noise Measurements Thorlabs + NO AMP')
	plt.show()
	plt.savefig('Thorlabs_noise_noAMP.png',dpi=300)
	

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange_20, noise_no_cables[0,:]/Vrange_20,'o-',label='no cables: CH1')
	#plt.plot(Vrange_20, noise_no_cables[1,:],'d-',label='no cables: CH2')
	plt.plot(Vrange_20[2:], noise_det_off[0,2:]/Vrange_20[2:],'<-',label='detector OFF: CH1')
	#plt.plot(Vrange_20, noise_det_off[1,:],'s-',label='detector OFF: CH2')
	#plt.plot(Vrange, noise_F[0,:],'^-', label='Femto; gain 1E3')
	plt.plot(Vrange, noise_T[0,:]/Vrange,'p-', label='Thorlabs dark; gain 0dB')
	plt.plot(Vrange, noise_T[2,:]/Vrange,'p-', label='Thorlabs dark; gain 20dB')
	plt.plot(Vrange, noise_T[4,:]/Vrange,'p-', label='Thorlabs dark; gain 40dB')
	plt.plot(Vrange, noise_T[6,:]/Vrange,'p-', label='Thorlabs dark; gain 60dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Readout range [V]')
	plt.ylabel('Noise per readout range [mV/V]')
	plt.title('Noise Measurements Thorlabs + AMP 10 V/V')
	plt.show()
	plt.savefig('Thorlabs_noise_amp10_relative.png',dpi=300)
	
	# thorlabs no AMP
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange_20, noise_no_cables[0,:]/Vrange_20,'o-',label='no cables: CH1')
	#plt.plot(Vrange_20, noise_no_cables[1,:],'d-',label='no cables: CH2')
	plt.plot(Vrange_20[2:], noise_det_off[0,2:]/Vrange_20[2:],'<-',label='detector OFF: CH1')
	#plt.plot(Vrange_20, noise_det_off[1,:],'s-',label='detector OFF: CH2')
	#plt.plot(Vrange, noise_F[0,:],'^-', label='Femto; gain 1E3')
	plt.plot(Vrange[:-1], noise_T_noAMP[0,:]/Vrange[:-1],'p-', label='Thorlabs dark; gain 0dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[2,:]/Vrange[:-1],'p-', label='Thorlabs dark; gain 20dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[4,:]/Vrange[:-1],'p-', label='Thorlabs dark; gain 40dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[6,:]/Vrange[:-1],'p-', label='Thorlabs dark; gain 60dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Readout range [V]')
	plt.ylabel('Noise per readout range [mV/V]')
	plt.title('Noise Measurements Thorlabs + NO AMP')
	plt.show()
	plt.savefig('Thorlabs_noise_noAMP_relative.png',dpi=300)
	
	#sys.exit()

	# Femto
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(gain_F, noise_F[:,0],'o-', label='0.2V')
	plt.plot(gain_F, noise_F[:,1],'d-', label='0.4V')
	plt.plot(gain_F, noise_F[:,2],'^-', label='0.8V')
	plt.plot(gain_F, noise_F[:,3],'s-', label='2V')
	plt.plot(gain_F, noise_F[:,4],'p-', label='4V')
	plt.plot(gain_F, noise_F[:,5],'v-', label='8V')
	plt.plot(gain_F, femto_noise_mV,  '--', label='Noise Spec.')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('detector gain')
	plt.ylabel('Noise [mV]')
	plt.title('Femto - dark noise')
	plt.show()
	plt.savefig('Femto_darknoise_perGain.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange, noise_F[0,:],'o-', label='gain 1E3')
	plt.plot(Vrange, noise_F[1,:],'d-', label='gain 1E4')
	plt.plot(Vrange, noise_F[2,:],'^-', label='gain 1E5')
	plt.plot(Vrange, noise_F[3,:],'s-', label='gain 1E6')
	plt.plot(Vrange, noise_F[4,:],'p-', label='gain 1E7')
	plt.plot(Vrange, noise_F[5,:],'v-', label='gain 1E8')
	plt.plot(Vrange, noise_F[6,:],'>-', label='gain 1E9')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Femto - dark noise')
	plt.show()
	plt.savefig('Femto_darknoise_perRange.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange, noise_F[0,:]/Vrange,'o-', label='gain 1E3')
	plt.plot(Vrange, noise_F[1,:]/Vrange,'d-', label='gain 1E4')
	plt.plot(Vrange, noise_F[2,:]/Vrange,'^-', label='gain 1E5')
	plt.plot(Vrange, noise_F[3,:]/Vrange,'s-', label='gain 1E6')
	plt.plot(Vrange, noise_F[4,:]/Vrange,'p-', label='gain 1E7')
	plt.plot(Vrange, noise_F[5,:]/Vrange,'v-', label='gain 1E8')
	plt.plot(Vrange, noise_F[6,:]/Vrange,'>-', label='gain 1E9')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise / readout range [mV/V]')
	plt.title('Femto - dark noise')
	plt.show()
	plt.savefig('Femto_darknoise_perRange_relative.png',dpi=300)

	# Thorlabs
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(gain_T, noise_T[:,0],'o-', label='0.2V')
	plt.plot(gain_T, noise_T[:,1],'d-', label='0.4V')
	plt.plot(gain_T, noise_T[:,2],'^-', label='0.8V')
	plt.plot(gain_T, noise_T[:,3],'s-', label='2V')
	plt.plot(gain_T, noise_T[:,4],'p-', label='4V')
	plt.plot(gain_T, noise_T[:,5],'v-', label='8V')
	#plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('detector gain [dB]')
	plt.ylabel('Noise [mV]')
	plt.ylim([.04,7])
	plt.title('Thorlabs - dark noise  + AMP 10 V/V')
	plt.show()
	plt.savefig('Thorlabs_darknoise_perGain_amp10.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange, noise_T[0,:],'o-', label='gain 0dB')
	plt.plot(Vrange, noise_T[1,:],'d-', label='gain 10dB')
	plt.plot(Vrange, noise_T[2,:],'^-', label='gain 20dB')
	plt.plot(Vrange, noise_T[3,:],'s-', label='gain 30dB')
	plt.plot(Vrange, noise_T[4,:],'p-', label='gain 40dB')
	plt.plot(Vrange, noise_T[5,:],'v-', label='gain 50dB')
	plt.plot(Vrange, noise_T[6,:],'>-', label='gain 60dB')
	plt.plot(Vrange, noise_T[7,:],'<-', label='gain 70dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs - dark noise + AMP 10 V/V')
	plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perRange_amp10.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange, noise_T[0,:]/Vrange,'o-', label='gain 0dB')
	plt.plot(Vrange, noise_T[1,:]/Vrange,'d-', label='gain 10dB')
	plt.plot(Vrange, noise_T[2,:]/Vrange,'^-', label='gain 20dB')
	plt.plot(Vrange, noise_T[3,:]/Vrange,'s-', label='gain 30dB')
	plt.plot(Vrange, noise_T[4,:]/Vrange,'p-', label='gain 40dB')
	plt.plot(Vrange, noise_T[5,:]/Vrange,'v-', label='gain 50dB')
	plt.plot(Vrange, noise_T[6,:]/Vrange,'>-', label='gain 60dB')
	plt.plot(Vrange, noise_T[7,:]/Vrange,'<-', label='gain 70dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs - dark noise + AMP 10 V/V')
	#plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perRange_amp10_relative.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange, noise_T[0,:]/Vrange/10**(0./20.),'o-', label='gain 0dB')
	plt.plot(Vrange, noise_T[1,:]/Vrange/10**(10./20.),'d-', label='gain 10dB')
	plt.plot(Vrange, noise_T[2,:]/Vrange/10**(20./20.),'^-', label='gain 20dB')
	plt.plot(Vrange, noise_T[3,:]/Vrange/10**(30./20.),'s-', label='gain 30dB')
	plt.plot(Vrange, noise_T[4,:]/Vrange/10**(40./20.),'p-', label='gain 40dB')
	plt.plot(Vrange, noise_T[5,:]/Vrange/10**(50./20.),'v-', label='gain 50dB')
	plt.plot(Vrange, noise_T[6,:]/Vrange/10**(60./20.),'>-', label='gain 60dB')
	plt.plot(Vrange, noise_T[7,:]/Vrange/10**(70./20.),'<-', label='gain 70dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs - dark noise + AMP 10 V/V')
	#plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perRange_amp10_relative_divgain.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(gain_T, noise_T[:,0]/10,'o-', label='0.2V')
	plt.plot(gain_T, noise_T[:,1]/10,'d-', label='0.4V')
	plt.plot(gain_T, noise_T[:,2]/10,'^-', label='0.8V')
	plt.plot(gain_T, noise_T[:,3]/10,'s-', label='2V')
	plt.plot(gain_T, noise_T[:,4]/10,'p-', label='4V')
	plt.plot(gain_T, noise_T[:,5]/10,'v-', label='8V')
	#plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('detector gain [dB]')
	plt.ylabel('Noise [mV]')
	plt.ylim([.04,7])
	plt.title('Thorlabs - dark noise  + AMP 10 V/V / SCALED')
	plt.show()
	plt.savefig('Thorlabs_darknoise_perGain_amp10_scaled.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(Vrange, noise_T[0,:]/10,'o-', label='gain 0dB')
	plt.plot(Vrange, noise_T[1,:]/10,'d-', label='gain 10dB')
	plt.plot(Vrange, noise_T[2,:]/10,'^-', label='gain 20dB')
	plt.plot(Vrange, noise_T[3,:]/10,'s-', label='gain 30dB')
	plt.plot(Vrange, noise_T[4,:]/10,'p-', label='gain 40dB')
	plt.plot(Vrange, noise_T[5,:]/10,'v-', label='gain 50dB')
	plt.plot(Vrange, noise_T[6,:]/10,'>-', label='gain 60dB')
	plt.plot(Vrange, noise_T[7,:]/10,'<-', label='gain 70dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs - dark noise + AMP 10 V/V  /  SCALED')
	plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perRange_amp10_scaled.png',dpi=300)

	# Thorlabs NO AMP
	plt.close()
	plt.figure(constrained_layout=True)
	plt.plot(gain_T, noise_T_noAMP[:,0],'o-', label='0.2V')
	plt.plot(gain_T, noise_T_noAMP[:,1],'d-', label='0.4V')
	plt.plot(gain_T, noise_T_noAMP[:,2],'^-', label='0.8V')
	plt.plot(gain_T, noise_T_noAMP[:,3],'s-', label='2V')
	plt.plot(gain_T, noise_T_noAMP[:,4],'p-', label='4V')
	plt.plot(gain_T, specnoise/1e3, '--', label='Noise Spec.')
	#plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('detector gain [dB]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs - dark noise - no AMP')
	plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perGain_noAMP.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	#my_ticks=['0.2','0.4','0.8','2','4']
	#plt.xticks((Vrange[:-1]),(my_ticks))
	plt.plot(Vrange[:-1], noise_T_noAMP[0,:],'o-', label='gain 0dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[1,:],'d-', label='gain 10dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[2,:],'^-', label='gain 20dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[3,:],'s-', label='gain 30dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[4,:],'p-', label='gain 40dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[5,:],'v-', label='gain 50dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[6,:],'>-', label='gain 60dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[7,:],'<-', label='gain 70dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise [mV]')
	plt.title('Thorlabs - dark noise + NO AMP')
	#plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perRange_noAMP.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	#my_ticks=['0.2','0.4','0.8','2','4']
	#plt.xticks((Vrange[:-1]),(my_ticks))
	plt.plot(Vrange[:-1], noise_T_noAMP[0,:]/Vrange[:-1],'o-', label='gain 0dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[1,:]/Vrange[:-1],'d-', label='gain 10dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[2,:]/Vrange[:-1],'^-', label='gain 20dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[3,:]/Vrange[:-1],'s-', label='gain 30dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[4,:]/Vrange[:-1],'p-', label='gain 40dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[5,:]/Vrange[:-1],'v-', label='gain 50dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[6,:]/Vrange[:-1],'>-', label='gain 60dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[7,:]/Vrange[:-1],'<-', label='gain 70dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise  / readout range [mV/V]')
	plt.title('Thorlabs - dark noise + NO AMP')
	#plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perRange_noAMP_relative.png',dpi=300)

	plt.close()
	plt.figure(constrained_layout=True)
	#my_ticks=['0.2','0.4','0.8','2','4']
	#plt.xticks((Vrange[:-1]),(my_ticks))
	plt.plot(Vrange[:-1], noise_T_noAMP[0,:]/Vrange[:-1]/10.**(0./20),'o-', label='gain 0dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[1,:]/Vrange[:-1]/10.**(10./20),'d-', label='gain 10dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[2,:]/Vrange[:-1]/10.**(20./20),'^-', label='gain 20dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[3,:]/Vrange[:-1]/10.**(30./20),'s-', label='gain 30dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[4,:]/Vrange[:-1]/10.**(40./20),'p-', label='gain 40dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[5,:]/Vrange[:-1]/10.**(50./20),'v-', label='gain 50dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[6,:]/Vrange[:-1]/10.**(60./20),'>-', label='gain 60dB')
	plt.plot(Vrange[:-1], noise_T_noAMP[7,:]/Vrange[:-1]/10.**(70./20),'<-', label='gain 70dB')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('readout range [V]')
	plt.ylabel('Noise  / readout range [mV/V]')
	plt.title('Thorlabs - dark noise + NO AMP')
	#plt.ylim([.04,7])
	plt.show()
	plt.savefig('Thorlabs_darknoise_perRange_noAMP_relative_divgain.png',dpi=300)

		
#### Here is a list of 'callable' routines
#### These routines (should) have no return.
#### Mostly they will generate a plot, and in some cases, they will print some values at the prompt.

noise_1_mrt()

#t1, s1 = mytools.load_data('27feb_PM_ch1',108,116)
#t2, s2 = mytools.load_data('30jan_ch1',92,92)

#plot_low_sig()

#plot_snr_27feb()
#plot_snr_22feb_DELTA()

#snr_thorlabs_22feb()

#snr_femto_22feb()

#t_T1, s_T1 = mytools.load_data('22feb_ch1',83,91)
#t_F1, s_F1 = mytools.load_data('22feb_ch2',83,91)


#t12, s12 = mytools.load_data('27feb_PM_ch1',72,75)
#t22, s22 = mytools.load_data('27feb_PM_ch2',72,75)


#plot_raw_sinewave_21feb()

#Tall_data_40dB, Tall_data_50dB, Tall_data_60dB = analyse_sine_wave_21feb_Thorlabs()

#Fall_data_40dB, Fall_data_50dB, Fall_data_60dB = analyse_sine_wave_21feb_Femto()

#time_40db_1VV_0Hz_1,    signal_40db_1VV_0Hz_1     = mytools.load_data('21feb_ch2',58,58)
#time_40db_1VV_0Hz_2,    signal_40db_1VV_0Hz_2     = mytools.load_data('21feb_ch2',60,60)
#time_40db_1VV_0Hz_3,    signal_40db_1VV_0Hz_3     = mytools.load_data('21feb_ch2',62,62)

#time_40db_1VV_0Hz_1T,    signal_40db_1VV_0Hz_1T     = mytools.load_data('21feb_ch1',58,58)
#time_40db_1VV_10Hz_1T,   signal_40db_1VV_10Hz_1T    = mytools.load_data('21feb_ch1',49,49)
#time_40db_1VV_120Hz_1T,  signal_40db_1VV_120Hz_1T   = mytools.load_data('21feb_ch1',52,52)

#time_50db_1VV_0Hz_2T,    signal_50db_1VV_0Hz_2T     = mytools.load_data('21feb_ch1',60,60)
#time_60db_1VV_0Hz_3T,    signal_60db_1VV_0Hz_3T     = mytools.load_data('21feb_ch1',62,62)

#Make_plot_for_Thomas()

#Wavelength_variability_14feb()

#Plot_test()

#Compare_Femto_and_Thorlabs_sinewave_response()

#Test_wavelength_dependence_13feb()

#Test_wavelength_dependence_6feb()

#Plot_sinewaves_and_fits()

#Make_sinewave_response_plots_6feb()
	
#Write_sinewave_measurements_toFile_6feb()

#Write_RAW_sinewave_measurements_toFile_6feb()

#Plot_sinewave_fits_30jan()

#Plot_percentile_data_30jan()

#Plot_bandwidth_test_1feb()

#dark_signal, dark_noise = Get_noise_data_1feb()

#Write_noise_measurements_toFile_1feb()

#Plot_noise_30jan()

#Plot_noise_1feb()

#Plot_EVOA_attenuation([0.87, 2.35,3.34])

#Compare_EVOA_calibrations()

#Make_EVOA_attentuation_plots()

#Plot_SNR_6feb()