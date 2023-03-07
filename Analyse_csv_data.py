import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def calc_stats(time, signal):
   std = np.std(signal)
   mean = np.mean(signal)
   #print "MEAN: ", mean 
   #print "STD: ", std
   return mean, std

def make_plot(time, signal):
   plt.close()
   plt.plot(time,signal)
   plt.show()


def read_csv(filename):

   with open(filename) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=';')
      header = []
      header = next(csv_reader)
      date_recorded_string = next(csv_reader)
      print date_recorded_string
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


#filename = "../CSV_DATA/ch2_000.csv" 
filename = "../CSV_DATA/myfilename_038.csv" 
time, signal = read_csv(filename)


make_plot(time, signal)

mean, std = calc_stats(time, signal)

print 'mean: ', mean
print 'std:  ', std
