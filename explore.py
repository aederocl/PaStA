import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import os
import sys
from astropy.io import fits

# https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

filtersDictionary = {
'FUV':'fuv_mag',
'NUV':'nuv_mag',
'Bp':'phot_bp_mean_mag',
'G':'phot_g_mean_mag',
'Rp':'phot_rp_mean_mag',
'J':'Jmag',
'H':'Hmag',
'K':'Kmag',
'W1':'W1mag',
'W2':'W2mag',
'W3':'W3mag',
'W4':'W4mag'
}

numberOfFilters = len(filtersDictionary)

print(len(filtersDictionary))

pasta = fits.open('pasta.fits')
#pasta.info()
pasta_data = pasta[1].data
pasta.close()

#print(pasta_data)

fig = plt.figure()

counter = 1

for index1,eachFilter1 in enumerate(filtersDictionary) :
 for index2,eachFilter2 in enumerate(filtersDictionary) :
  if index1 == index2 :
   print('histogram',eachFilter1)
   print(index1)
   print(counter)
   #plt.subplot(numberOfFilters,numberOfFilters,index1 + index2 + 1)
   plt.subplot(numberOfFilters,numberOfFilters,counter)
   plt.hist(pasta_data[filtersDictionary[eachFilter1]])
   plt.xlabel(eachFilter1)
  elif index1 > index2 :
   print(counter)
   print(eachFilter1,eachFilter2)
   #plt.subplot(numberOfFilters,numberOfFilters,index1 + index2 + 1)
   plt.subplot(numberOfFilters,numberOfFilters,counter)
   plt.scatter(pasta_data[filtersDictionary[eachFilter1]],pasta_data[filtersDictionary[eachFilter2]])
   plt.xlabel(eachFilter1)
   plt.ylabel(eachFilter2)
  counter = counter + 1

plt.savefig('explore.png')
#plt.show()