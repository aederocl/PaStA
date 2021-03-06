import os
import sys
from astropy.io import fits
import sqlite3

def checkDB(myMinRA,myMaxRA,myMinDec,myMaxDec):
 myFieldAlreadyQueried = True
 if not os.path.exists('pasta.db'):
  sys.exit('pasta.db does not exists')
 else:
  conn = sqlite3.connect('pasta.db')
 
 cur = conn.cursor()
 cur.execute('select RA_MIN,RA_MAX,DEC_MIN,DEC_MAX from PASTA')
 rows = cur.fetchall()
 #for eachRow in rows:
 # print(eachRow)
 if (float(myMinRA),float(myMaxRA),float(myMinDec),float(myMaxDec)) not in rows:
  myFieldAlreadyQueried = False
  #print('gotcha') 
 return myFieldAlreadyQueried
 
 
def addFieldToDB(myMinRA,myMaxRA,myMinDec,myMaxDec, myNGaia, myNGalex, myNAllWise):
 if not os.path.exists('pasta.db'):
  sys.exit('pasta.db does not exists')
 else:
  conn = sqlite3.connect('pasta.db')
 
 cur = conn.cursor()
 SQLCommand = 'INSERT INTO PASTA (RA_MIN,RA_MAX,DEC_MIN,DEC_MAX,N_OBJ_GAIA,N_OBJ_GALEX,N_OBJ_ALLWISE) VALUES (' + \
 str(float(myMinRA)) + ',' + str(float(myMaxRA)) +  ',' + str(float(myMinDec)) + \
 ',' +  str(float(myMaxDec)) + ',' + str(float(myNGaia)) + ',' + str(float(myNGalex)) + \
 ',' + str(float(myNAllWise)) +   ')'
 print(SQLCommand)
 cur.execute(SQLCommand)
 conn.commit()


 
def countSources(myInputFitsFile):
 myNumberOfSources = 0
 if os.path.exists(myInputFitsFile):
  hdul = fits.open(myInputFitsFile)
  if len(hdul) > 1 :
   myData = hdul[1].data
   myNumberOfSources = myData.shape[0]  
 return myNumberOfSources

#if not os.path.exists(outputFitsFileName) :
minRA = sys.argv[1]
maxRA = sys.argv[2]
minDec = sys.argv[3]
maxDec = sys.argv[4]

outputFitsFileName = 'pasta_tmp_' + str(minRA) + '_' +  str(minDec).replace('-','m') + '.fits'

FieldAlreadyQueried = checkDB(minRA,maxRA,minDec,maxDec)

if FieldAlreadyQueried == False:

 # make a TAP query and put the output in tmp.fits
 stiltsCommand = 'stilts tapquery tapurl=https://gea.esac.esa.int/tap-server/tap  \
    adql="select * from gaiaedr3.gaia_source \
    where parallax_over_error > 5 and ruwe < 1.4 and \
    ra between  ' + str(minRA) + \
    ' and ' + str(maxRA)+ ' and dec between ' + str(minDec) + ' and ' + str(maxDec) +  ' "  out=tmp.fits'
 print(stiltsCommand)
 os.system(stiltsCommand)
 NumberGaia = countSources('tmp.fits')
 NumberGalex = 0
 NumberAllWise = 0
 if NumberGaia > 0 :
    # now I match the Gaia sources with GALEX and put the result in a galex.fits file
    stiltsCommand = 'stilts cdsskymatch in=tmp.fits ifmt=fits  ra=ra dec=dec cdstable=II/312/ais radius=1.0 find=best out=galex.fits'
    print(stiltsCommand)
    os.system(stiltsCommand)
    NumberGalex = countSources('galex.fits')
    print(NumberGalex)
    if NumberGalex == 0: # if there is nothing in Galex, then move to the next Gaia field
     #os.system('mv tmp.fits galex.fits')
     NumberAllWise = 0 # since I don't cross-match with AllWISE, I assign 0 by default
    else: # if there are Gaia-GALEX sources, we go on to AllWISE
     os.system('rm tmp.fits')
     stiltsCommand = 'stilts cdsskymatch in=galex.fits ifmt=fits  ra=ra_in dec=dec_in cdstable=II/328/allwise radius=1.0 find=best out=allwise.fits'
     print(stiltsCommand)
     os.system(stiltsCommand)
     NumberAllWise = countSources('allwise.fits')
     print(NumberAllWise)
     if NumberAllWise == 0:
      os.system('mv galex.fits allwise.fits')
      os.system("rm allwise.fits")
     else:
      os.system('rm galex.fits')
      #outpasta = 'pasta_tmp_' + str(rightAscension) + '_' +  str(declination).replace('-','m') + '.fits'
      os.system('cp allwise.fits ' + outputFitsFileName )

 addFieldToDB(minRA,maxRA,minDec,maxDec,NumberGaia,NumberGalex,NumberAllWise)
else:
 print('sorry, this field has already been queried')
  
'''
if NumberGaia > 0 :
    # now I match the Gaia sources with GALEX and put the result in a galex.fits file
    stiltsCommand = 'stilts cdsskymatch in=tmp.fits ifmt=fits  ra=ra dec=dec cdstable=II/312/ais radius=1.0 find=best out=galex.fits'
    print(stiltsCommand)
    os.system(stiltsCommand)
    NumberGalex = countSources('galex.fits')
    print(NumberGalex)
    if NumberGalex == 0: # if there is nothing in Galex, then move to the next Gaia field
     #os.system('mv tmp.fits galex.fits')
     NumberAllWise = 0 # since I don't cross-match with AllWISE, I assign 0 by default
    else: # if there are Gaia-GALEX sources, we go on to AllWISE
     os.system('rm tmp.fits')
     stiltsCommand = 'stilts cdsskymatch in=galex.fits ifmt=fits  ra=ra_in dec=dec_in cdstable=II/328/allwise radius=1.0 find=best out=allwise.fits'
     print(stiltsCommand)
     os.system(stiltsCommand)
     NumberAllWise = countSources('allwise.fits')
     print(NumberAllWise)
     if NumberAllWise == 0:
      os.system('mv galex.fits allwise.fits')
      os.system("rm allwise.fits")
     else:
      os.system('rm galex.fits')
      #outpasta = 'pasta_tmp_' + str(rightAscension) + '_' +  str(declination).replace('-','m') + '.fits'
      os.system('cp allwise.fits ' + outputFitsFileName )

     #NumberPasta = countSources('pasta.fits')
     myString = str(declination) + ' ' + str(rightAscension) + ' ' + str(NumberGaia) + ' ' + \
      str(NumberGalex) + ' ' + str(NumberAllWise) #+ ' ' + str(NumberPasta)
     print(myString)
     myLog.write(myString + '\n')

'''