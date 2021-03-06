import os
import sys
from astropy.io import fits

SurveyMinDec = -10
SurveyMaxDec = 10
SurveyMinRA = 40
SurveyMaxRA = 90

dec_step = 5
ra_step = 5

myLog = open('runPaStA.sh','w')

for declination in range(SurveyMinDec,SurveyMaxDec,dec_step):
 for rightAscension in range(SurveyMinRA,SurveyMaxRA,ra_step):
  #print(declination,rightAscension)
  minDec = declination
  maxDec = declination + dec_step
  minRA = rightAscension
  maxRA = rightAscension + ra_step
  myString = 'python get_catalogues_v3.py ' + str(minRA) + ' ' + str(maxRA) + ' ' + str(minDec) + ' ' + str(maxDec)
  myLog.write(myString + '\n')

myLog.close()