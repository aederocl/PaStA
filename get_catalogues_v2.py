import os
import sys
from astropy.io import fits

def countSources(myInputFitsFile):
 myNumberOfSources = 0
 if os.path.exists(myInputFitsFile):
  hdul = fits.open(myInputFitsFile)
  if len(hdul) > 1 :
   myData = hdul[1].data
   myNumberOfSources = myData.shape[0]  
 return myNumberOfSources


if os.path.exists('pasta.fits'):
 print('removing old pasta.fits')
 os.system('rm pasta.fits')
 print('old pasta.fits removed')


SurveyMinDec = -10
SurveyMaxDec = 10
SurveyMinRA = 0
SurveyMaxRA = 360

dec_step = 5
ra_step = 5

myLog = open('myLog.txt','w')

for declination in range(SurveyMinDec,SurveyMaxDec,dec_step):
 for rightAscension in range(SurveyMinRA,SurveyMaxRA,ra_step):
  print(declination,rightAscension)
  minDec = declination
  maxDec = declination + dec_step
  minRA = rightAscension
  maxRA = rightAscension + ra_step
  #stiltsCommand = 'stilts tapquery tapurl=https://gea.esac.esa.int/tap-server/tap  \
  #adql="select * from gaiaedr3.gaia_source \
  #where parallax_over_error > 5 and ruwe < 1.4 and \
  #astrometric_excess_noise < 1 and phot_bp_rp_excess_factor > (1 + 0.015 * bp_rp * bp_rp) and \
  #phot_bp_rp_excess_factor < (1.3 + 0.06 * bp_rp * bp_rp ) and \
  #ra between  ' + str(minRA) + \
  #' and ' + str(maxRA)+ ' and dec between ' + str(minDec) + ' and ' + str(maxDec) +  ' "  out=tmp.fits'
  
  # make a TAP query and put the output in tmp.fits
  stiltsCommand = 'stilts tapquery tapurl=https://gea.esac.esa.int/tap-server/tap  \
  adql="select * from gaiaedr3.gaia_source \
  where parallax_over_error > 5 and ruwe < 1.4 and \
  ra between  ' + str(minRA) + \
  ' and ' + str(maxRA)+ ' and dec between ' + str(minDec) + ' and ' + str(maxDec) +  ' "  out=tmp.fits'
  print(stiltsCommand)
  os.system(stiltsCommand)
  NumberGaia = countSources('tmp.fits')
  
  if NumberGaia > 0 :
   # now I match the Gaia sources with GALEX and put the result in a galex.fits file
   stiltsCommand = 'stilts cdsskymatch in=tmp.fits ifmt=fits  ra=ra dec=dec cdstable=II/312/ais radius=1.0 find=best out=galex.fits'
   print(stiltsCommand)
   os.system(stiltsCommand)
   NumberGalex = countSources('galex.fits')
   if NumberGalex == 0: # if there is nothing in Galex, then move to the next Gaia field
    #os.system('mv tmp.fits galex.fits')
    NumberAllWise = 0 # since I don't cross-match with AllWISE, I assign 0 by default
   else: # if there are Gaia-GALEX sources, we go on to AllWISE
    os.system('rm tmp.fits')
    stiltsCommand = 'stilts cdsskymatch in=galex.fits ifmt=fits  ra=ra_in dec=dec_in cdstable=II/328/allwise radius=1.0 find=best out=allwise.fits'
    print(stiltsCommand)
    os.system(stiltsCommand)
    NumberAllWise = countSources('allwise.fits')
    if NumberAllWise == 0:
     os.system('mv galex.fits allwise.fits')
     os.system("rm allwise.fits")
    else:
     os.system('rm galex.fits')
     outpasta = 'pasta_tmp_' + str(rightAscension) + '_' +  str(declination).replace('-','m') + '.fits'
     os.system('cp allwise.fits ' + outpasta )

     if not os.path.exists('pasta.fits'):
      os.system('mv allwise.fits pasta.fits')
     else:
      stiltsCommand = 'stilts tcat ifmt=fits in=pasta.fits in=allwise.fits out=tmp.fits'
      os.system(stiltsCommand)
      os.system('rm pasta.fits')
      os.system('rm allwise.fits')
      os.system('mv tmp.fits pasta.fits')

  NumberPasta = countSources('pasta.fits')
  myString = str(declination) + ' ' + str(rightAscension) + ' ' + str(NumberGaia) + ' ' + \
   str(NumberGalex) + ' ' + str(NumberAllWise) + ' ' + str(NumberPasta)
  print(myString)
  myLog.write(myString + '\n')

myLog.close()