import os
import sys
from astropy.io import fits
import glob

if os.path.exists('pasta.fits'):
 print('removing old pasta.fits')
 os.system('rm pasta.fits')
 print('old pasta.fits removed')

myFiles = glob.glob('pasta_tmp*fits')

for eachfile in myFiles :
 print(eachfile)
 if not os.path.exists('pasta.fits'):
  os.system('mv ' + eachfile + ' pasta.fits')
 else:
  stiltsCommand = 'stilts tcat ifmt=fits in=pasta.fits in=' + eachfile + ' out=tmp.fits'
  os.system(stiltsCommand)
  os.system('rm pasta.fits')
  os.system('mv tmp.fits pasta.fits')
