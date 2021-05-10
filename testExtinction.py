import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import astropy.units as units
import os
import sys
from astropy.io import fits
from astropy.coordinates import SkyCoord
from dustmaps.planck import PlanckQuery
from dustmaps.sfd import SFDQuery
from dustmaps.bayestar import BayestarQuery


pasta = fits.open('pasta_tmp_0_5.fits')
#pasta.info()
pasta_data = pasta[1].data
pasta.close()

#print(pasta_data)

pasta_distances = 1000. / pasta_data['parallax']

coords = SkyCoord(pasta_data['ra_in']*units.deg, pasta_data['dec_in']*units.deg, distance=pasta_distances*units.pc , frame='icrs')

planck = PlanckQuery()
plankExtinctions = planck(coords)

sfd = SFDQuery()
sftExtinctions = sfd(coords)

bayestar = BayestarQuery(max_samples=2, version='bayestar2019')
ebv = bayestar(coords, mode='median')

plt.scatter(pasta_data['E_bv'],ebv)
#plt.scatter(sftExtinctions,plankExtinctions)
#plt.scatter( pasta_data['E_bv'] , sftExtinctions)
plt.show()