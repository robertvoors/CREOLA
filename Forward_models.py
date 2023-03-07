'''
Forward models
'''

import numpy as np

def fibre_connector_loss(sig_in, unit_in, loss=5):
   if (loss == 5): print " ** Default loss of 5 percent is assumed"
   sig_out = sig_in * (100-loss)/100
   unit_out = unit_in
   return sig_out, unit_out


def get_laser_sig(lambda_set=1550, power=8.8):
   # Note: two inputs are required: wavelength, in nm and laser power, in dBm
   # this will be converted to output in Watts
   if (lambda_set == 1550): print " ** Default wavelength of 1550 nm is assumed"
   if (power == 8): print " ** Default signal strength of 8 dBm is assumed"
   sig_out = 10.**((power - 30.) / 10.)
   unit_out = 'W'
   return sig_out, unit_out


def get_EVOA_loss(sig_in, unit_in, loss=1.5):
   # Note input loss is in dB; this is converted in linear loss
   if (loss == 1.5): print " ** Default loss of 1.5 dB is assumed"
   if (unit_in == 'W'): 
      max_sig = 200e-3
      saturation_check(sig_in, max_sig, unit_in, "EVOA_loss")
   else:
      print "Saturation check cannot be done; unexpected units"
   frac_loss = 10.**(loss/10.)
   sig_out = sig_in / frac_loss
   unit_out = unit_in
   return sig_out, unit_out


def attenuator_loss(sig_in, unit_in, loss=0):
   # Loss in dB; default is no loss
   if (loss == 0): print " ** Default loss of 0 dB is assumed"
   frac_loss = 10.**(loss/10.)
   sig_out = sig_in / frac_loss
   unit_out = unit_in
   return sig_out, unit_out

def detector_conversion_signal(sig_in, unit_in, conv_rate=1):
   # convert incoming signal in Watts to a current in Ampere
   if (conv_rate == 1): print " ** Default conversion rate (A/W) of 1 assumed"
   sig_out = sig_in * conv_rate
   if (unit_in == 'W'): 
      unit_out = 'A'
   else:
      exit('Wrong units at input of algorithm detector_conversion_signal')
   return sig_out, unit_out

def get_conversion_rate(gain):
   conv_rate = 1510. * 10.**(gain/20.)
   return conv_rate

def multiply_by_detector_gain(sig_in, unit_in, gain=0, det_name=""):
   # convert detector current to Voltage
   if (gain == 0): print " ** Default detector gain of 0 dB assumed"
   if (det_name == 'Thorlabs'):
      amp_to_volt = get_conversion_rate(gain)
   else:
      amp_to_volt = 10.**(gain/10.)
   print "amp_to_volt ", amp_to_volt
   
   sig_out = sig_in * amp_to_volt
   
   if (unit_in == 'A'): 
      unit_out = 'V'
   else:
      exit('Wrong units at input of algorithm detector_gain')
   if (unit_out == 'V'): 
      max_sig = 10 
      saturation_check(sig_out, max_sig, unit_out, "detector_gain")
   else:
      print "Saturation check cannot be done; unexpected units"

   return sig_out, unit_out

def voltage_amplification(sig_in, unit_in, amp=1):
   if (amp == 1): print " ** Default voltage amplification factor of 1 assumed"
   unit_out = unit_in
   sig_out = sig_in * amp
   if (unit_in == 'V'): 
      max_sig = 2 
      saturation_check(np.abs(sig_out), max_sig, unit_in, "voltage_amplification")
   else:
      print "Saturation check cannot be done; unexpected units"

   return sig_out, unit_out

def get_dark_current(sig_in, unit_in, gain=0, amp=1, det_name=""):
   if (det_name == 'thorlabs'):
      import pickle
      with open('dark_noise_signal.pkl','rb') as f:
         dark_noise_sig = pickle.load(f)
      dark_signal = dark_noise_sig['signal']
      #dark_noise = dark_noise_sig['noise']
      gain_array = np.asarray([0,10,20,30,40,50,60,70])
      gain_idx = np.where(gain_array==gain)[0][0]
      amp_str = 'AMP'+str(amp)
      dc = dark_signal[amp_str][gain_idx]
      sig_out = dc
   else:
      sig_out = 0.033
   unit_out = unit_in
   return sig_out, unit_out

def add_dark_currrent(sig_in, unit_in, gain=0, amp=1):
   if (amp == 1): print " ** Default voltage amplification factor of 1 assumed"
   if (gain == 0): print " ** Default detector gain of 0 dB is assumed"

   dark_current, unit_out = get_dark_current(sig_in, unit_in, gain, amp) 
   sig_out = sig_in + dark_current
   return sig_out, unit_out

def add_offset(sig_in, unit_in, offset=0.0):
   unit_out = unit_in
   sig_out = sig_in + offset
   return sig_out, unit_out

def saturation_check(sig_in, max_sig, algorithm_name, unit):
   if (sig_in > max_sig):
      print " ** Possible saturation at ", algorithm_name, " !!!"
      print " - max signal: ", str(max_sig), unit
      print " - measured signal: ", sig_in, " ", unit

def print_line(value_in, units_in, num_decimals=3, text=''):
   if (len(text)>0):
      print text
   if (num_decimals==1):
      print   '{:.1e}'.format(value_in), '[',units_in,']'
   elif (num_decimals==2):
      print   '{:.2e}'.format(value_in), '[',units_in,']'
   elif (num_decimals==3):
      print   '{:.3e}'.format(value_in), '[',units_in,']'
   elif (num_decimals==4):
      print   '{:.3e}'.format(value_in), '[',units_in,']'
   else: 
      print   '{:.3e}'.format(value_in), '[',units_in,']'
