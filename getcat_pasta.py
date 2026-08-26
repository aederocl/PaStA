"""
Build one PaStA sky tile: Gaia DR3 x GALEX AIS x AllWISE(+2MASS), with
Gaia-ESO/APOGEE/LAMOST/SEGUE atmospheric parameters and CSFD/Edenhofer2023
extinction chained on.

This is a fixed version of getcat_20260720.py (Morgan Camargo's tile
builder). Usage is unchanged:

    python getcat_pasta.py <minRA> <maxRA> <minDec> <maxDec>

Fixes relative to getcat_20260720.py
-------------------------------------
1. Duplicate sources at tile edges (found: 377 sources, 754 rows, in
   pasta.fits -- confined to one RA~4-6 deg / Dec~-2-0 deg patch, not
   spread across the sky). Root cause: the Gaia ADQL query used
   "ra between minRA and maxRA" (inclusive on both ends), so a source
   sitting exactly on a tile boundary is captured by both adjacent
   tiles. Fixed by querying a half-open interval [minRA, maxRA) x
   [minDec, maxDec) instead. NOTE: whatever driver script generates the
   grid of (minRA, maxRA, minDec, maxDec) tiles must give the very last
   tile in RA (approaching 360) and in Dec (approaching +90) an
   inclusive upper bound, or those two edges of the survey will be
   silently dropped -- that grid-generation script wasn't available to
   fix here.

2. checkDB() crash on an already-queried tile. The original had a typo:
   it initialised `myFieldAreladyQueried` (note the misspelling) but the
   branch that marks a field as new set a *different*, correctly-spelled
   variable `myFieldAlreadyQueried`, which was then the one returned. So
   whenever a tile *was* already in pasta.db, the function hit
   `myFieldAlreadyQueried` unassigned and raised UnboundLocalError instead
   of returning True. If a crashed run was ever manually restarted for a
   tile without this check working, that tile could be processed a second
   time -- this is one plausible mechanism for the duplicates in (1).
   Rewritten below as a plain, always-well-defined boolean check.

3. `sys,exit(...)` (comma instead of a dot) in both checkDB() and
   addFieldToDB(): this silently does NOT call sys.exit -- it evaluates
   the tuple `(sys, exit('...'))`, using the builtin `exit()` rather than
   `sys.exit()`. Fixed to `sys.exit(...)`.

4. Columns that should have been deleted but weren't. The original
   script asked STILTS to `delcols RAJ2000_galex/DEJ2000_galex` and
   `delcols RAJ2000_allwise/DEJ2000_allwise` inside the same ocmd chain
   that ran the sky match -- but pasta.fits still has all four columns,
   meaning that delete silently didn't happen (most likely because
   STILTS only auto-suffixes a remote column when its name clashes with
   an existing local one, so the exact suffixed name to delete isn't
   guaranteed). Fixed by dropping those columns afterwards in Python,
   where the column names actually present can be checked directly
   rather than assumed.

Not changed (flagged, not fixed, since these are science-methodology
calls, not bugs)
-----------------------------------------------------------------------
- AllWISE/GALEX matching here is a positional crossmatch to the AllWISE/
  GALEX AIS *point-source catalogues* (STILTS tapskymatch, find=best,
  sr=0.00027 deg ~ 0.97"), not forced photometry from the WISE images.
  Worth checking this is what the paper text actually claims.
- The Gaia J2000 propagation (`raj2000 = ra - PMRA/(3.6e6)*16`) does not
  divide the RA term by cos(dec), even though Gaia's PMRA is
  mu_alpha* = mu_alpha*cos(dec). This under- or over-corrects RA at high
  declination / high proper motion. Left as-is here since correcting it
  changes which GALEX/AllWISE sources get matched, not just bookkeeping --
  that's a decision for the group, not a silent fix.
"""
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as units
import os
import sys
import sqlite3
from astropy.io import fits
from dustmaps.config import config
_dustmaps_dir = os.environ.get('DUSTMAPS_DATA_DIR')
if _dustmaps_dir:
    config['data_dir'] = _dustmaps_dir
from dustmaps.csfd import CSFDQuery
from dustmaps.edenhofer2023 import Edenhofer2023Query


def checkFieldAlreadyQueried(min_ra, max_ra, min_dec, max_dec):
    if not os.path.exists('pasta.db'):
        sys.exit('pasta.db does not exist')
    conn = sqlite3.connect('pasta.db')
    try:
        cur = conn.cursor()
        cur.execute('select RA_MIN,RA_MAX,DEC_MIN,DEC_MAX from PASTA')
        rows = cur.fetchall()
    finally:
        conn.close()
    tile = (float(min_ra), float(max_ra), float(min_dec), float(max_dec))
    return tile in rows


def addFieldToDB(myMinRA, myMaxRA, myMinDec, myMaxDec, myNGaia, myNGalex, myNAllWise):
    if not os.path.exists('pasta.db'):
        sys.exit('pasta.db does not exist')
    conn = sqlite3.connect('pasta.db')
    try:
        cur = conn.cursor()
        SQLCommand = 'INSERT INTO PASTA (RA_MIN,RA_MAX,DEC_MIN,DEC_MAX,N_OBJ_GAIA,N_OBJ_GALEX,N_OBJ_ALLWISE) VALUES (' + \
            str(float(myMinRA)) + ',' + str(float(myMaxRA)) + ',' + str(float(myMinDec)) + \
            ',' + str(float(myMaxDec)) + ',' + str(float(myNGaia)) + ',' + str(float(myNGalex)) + \
            ',' + str(float(myNAllWise)) + ')'
        print(SQLCommand)
        cur.execute(SQLCommand)
        conn.commit()
    finally:
        conn.close()


def countSources(myInputFitsFile):
    myNumberOfSources = 0
    if os.path.exists(myInputFitsFile):
        with fits.open(myInputFitsFile) as hdul:
            if len(hdul) > 1:
                myNumberOfSources = hdul[1].data.shape[0]
    return myNumberOfSources


def dropColumnsIfPresent(fits_path, columns_to_drop):
    """Remove columns by name if they exist; no-op (and no error) if not.
    Replaces relying on STILTS ocmd delcols guessing the right suffixed
    column name (see fix 4 in the module docstring)."""
    t = Table.read(fits_path)
    existing = [c for c in columns_to_drop if c in t.colnames]
    if existing:
        t.remove_columns(existing)
        t.write(fits_path, overwrite=True)
    return existing


def insertAPs():
    stiltsCommand = """
    ~/stilts tmatch2 in1=allwise.fits ifmt1=fits in2=gaiaeso.fits ifmt2=fits out="geso.fits" matcher=sky values1="ra dec" values2="RA DECLINATION" params="1" find=best join=all1 fixcols=all suffix1="" suffix2="_geso" ocmd="delcols 'Separation RA_geso DECLINATION_geso'"
    """
    print(stiltsCommand)
    os.system(stiltsCommand)
    os.system('rm allwise.fits')
    stiltsCommand = """
    ~/stilts tmatch2 in1=geso.fits ifmt1=fits in2=apogee.fit ifmt2=fits out="apog.fits" matcher=sky values1="ra dec" values2="RAJ2000 DEJ2000" params="1" find=best join=all1 fixcols=all suffix1="" suffix2="_apog" ocmd="delcols 'Separation RAJ2000_apog DEJ2000_apog'"
    """
    print(stiltsCommand)
    os.system(stiltsCommand)
    os.system('rm geso.fits')
    stiltsCommand = """
    ~/stilts tmatch2 in1=apog.fits ifmt1=fits in2=lamostafgk.fit ifmt2=fits out="lafgk.fits" matcher=sky values1="ra dec" values2="RAJ2000 DEJ2000" params="1" find=best join=all1 fixcols=all suffix1="" suffix2="_lafgk" ocmd="delcols 'Separation RAJ2000_lafgk DEJ2000_lafgk'"
    """
    print(stiltsCommand)
    os.system(stiltsCommand)
    os.system('rm apog.fits')
    stiltsCommand = """
    ~/stilts tmatch2 in1=lafgk.fits ifmt1=fits in2=lamostm.fit ifmt2=fits out="lm.fits" matcher=sky values1="ra dec" values2="RAJ2000 DEJ2000" params="1" find=best join=all1 fixcols=all suffix1="" suffix2="_lm" ocmd="delcols 'Separation RAJ2000_lm DEJ2000_lm'"
    """
    print(stiltsCommand)
    os.system(stiltsCommand)
    os.system('rm lafgk.fits')
    stiltsCommand = """
    ~/stilts tmatch2 in1=lm.fits ifmt1=fits in2=segue.fits ifmt2=fits out="final.fits" matcher=sky values1="ra dec" values2="ra dec" params="1" find=best join=all1 fixcols=all suffix1="" suffix2="_seg" ocmd="delcols 'Separation objid_seg ra_seg dec_seg specobjid_seg'"
    """
    print(stiltsCommand)
    os.system(stiltsCommand)
    os.system('rm lm.fits')

    datatable = Table.read('final.fits')
    df = datatable.to_pandas()

    coords = SkyCoord(df['ra'] * units.deg, df['dec'] * units.deg, distance=df['distance'] * units.pc,
                       unit=("deg", "deg", "pc"), frame='icrs')

    edenhofer = Edenhofer2023Query(integrated=True)
    ebv3d = (edenhofer(coords) * 2.8) / 3.1

    csfd = CSFDQuery()
    ebv2d = csfd(coords)

    df['extinction'] = np.where(df['distance'] > 1250, ebv2d, ebv3d)

    geso_condition = df['TEFF_geso'].notnull()
    apogee_condition = df['Teff_apog'].notnull() & ((df['phot_g_mean_mag'].le(10) | df['phot_g_mean_mag'].between(12, 16, inclusive='right')) | (df['phot_g_mean_mag'].between(10, 12, inclusive='right') & df['Teff_lafgk'].empty == True))
    segue_condition = df['teffadop_seg'].ge(0) & (((df['phot_g_mean_mag'].le(16)) | (df['phot_g_mean_mag'].between(14, 16, inclusive='right')) & (df['Teff_apog'].empty == True & df['Teff_lafgk'].empty == True)))
    lafgk_condition = df['Teff_lafgk'].notnull() & (df['phot_g_mean_mag'].between(10, 18, inclusive='right'))
    lm_condition = df['Teff_lm'].notnull() & (df['phot_g_mean_mag'].between(12, 18, inclusive='right'))

    condition = [geso_condition, apogee_condition, segue_condition, lafgk_condition, lm_condition]
    choice = [1, 2, 3, 4, 5]
    df['AP_origin'] = np.select(condition, choice, default=0).astype(int)

    choice = [df['TEFF_geso'], df['Teff_apog'], df['teffadop_seg'], df['Teff_lafgk'], df['Teff_lm']]
    df['Teff'] = np.select(condition, choice, default=np.nan)

    choice = [df['E_TEFF_geso'], df['e_Teff_apog'], df['teffadopunc_seg'], df['e_Teff_lafgk'], df['e_Teff_lm']]
    df['e_Teff'] = np.select(condition, choice, default=np.nan)

    choice = [df['LOGG_geso'], df['logg_apog'], df['loggadop_seg'], df['logg_lafgk'], df['logg_lm']]
    df['logg'] = np.select(condition, choice, default=np.nan)
    choice = [df['E_LOGG_geso'], df['e_logg_apog'], df['loggadopunc_seg'], df['e_logg_lafgk'], df['e_logg_lm']]
    df['e_logg'] = np.select(condition, choice, default=np.nan)

    condition = [geso_condition, apogee_condition, segue_condition, lafgk_condition]
    choice = [df['FEH_geso'], df['__Fe_H__apog'], df['fehadop_seg'], df['__Fe_H__lafgk']]
    df['feh'] = np.select(condition, choice, default=np.nan)
    choice = [df['E_FEH_geso'], df['e__Fe_H__apog'], df['fehadopunc_seg'], df['e__Fe_H__lafgk']]
    df['e_feh'] = np.select(condition, choice, default=np.nan)

    df = df.drop(labels=['TEFF_geso', 'E_TEFF_geso', 'LOGG_geso',
                          'E_LOGG_geso', 'FEH_geso', 'E_FEH_geso',
                          'Teff_apog', 'e_Teff_apog', 'logg_apog', 'e_logg_apog', '__Fe_H__apog',
                          'e__Fe_H__apog', 'Teff_lafgk',
                          'e_Teff_lafgk', 'logg_lafgk', 'e_logg_lafgk', '__Fe_H__lafgk',
                          'e__Fe_H__lafgk', 'Teff_lm', 'e_Teff_lm',
                          'logg_lm', 'e_logg_lm', 'fehadop_seg', 'fehadopunc_seg', 'teffadop_seg',
                          'teffadopunc_seg', 'loggadop_seg', 'loggadopunc_seg'], axis=1)

    t = Table.from_pandas(df)
    t.write('full.fits', overwrite=True)


def main():
    minRA = sys.argv[1]
    maxRA = sys.argv[2]
    minDec = sys.argv[3]
    maxDec = sys.argv[4]

    outputFitsFileName = 'pasta_tmp_' + str(minRA) + '_' + str(minDec).replace('-', 'm') + '.fits'

    if checkFieldAlreadyQueried(minRA, maxRA, minDec, maxDec):
        print('sorry, this field has already been queried')
        return

    # Fix 1: half-open interval [minRA, maxRA) x [minDec, maxDec) so a
    # source sitting exactly on a shared tile edge is captured by exactly
    # one of the two adjacent tiles, not both.
    stiltsCommand = '~/stilts tapquery tapurl=https://gea.esac.esa.int/tap-server/tap  \
    adql="select source_id, ref_epoch, ra, ra_error, dec, dec_error, parallax, parallax_error, PMRA, PMRA_ERROR, PMDEC, PMDEC_ERROR, ruwe, phot_g_mean_flux, phot_g_mean_flux_error, phot_g_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_mean_mag, radial_velocity, radial_velocity_error, phot_variable_flag, non_single_star, has_xp_sampled, has_epoch_photometry from gaiadr3.gaia_source \
    where parallax_over_error > 5 and ruwe < 1.4 and \
    ra >= ' + str(minRA) + \
        ' and ra < ' + str(maxRA) + ' and dec >= ' + str(minDec) + ' and dec < ' + str(maxDec) + ' " ocmd="addcol distance 1000/parallax" ocmd="addcol -after ra raj2000 ra-PMRA/(3600*1000)*16 " ocmd="addcol dej2000 -after dec dec-PMDEC/(3600*1000)*16" out=gaia.fits'
    print(stiltsCommand)
    os.system(stiltsCommand)
    NumberGaia = countSources('gaia.fits')
    NumberGalex = 0
    NumberAllWise = 0
    if NumberGaia > 0:
        stiltsCommand = """
        ~/stilts tapskymatch tapurl=http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap taptable='"II/335/galex_ais"' taplon=RAJ2000 taplat=DEJ2000 in=gaia.fits inlon=RAJ2000 inlat=DEJ2000 tapcols=RAJ2000,DEJ2000,FUVmag,e_FUVmag,NUVmag,e_NUVmag suffixin= suffixremote=_galex sr=0.00027 find=best ocmd="addcol -before FUVmag FUV FUVmag-2.223" ocmd="addcol -before NUVmag NUV NUVmag-1.699" ocmd="delcols SEP_ARCSEC" ocmd="delcols FUVmag" ocmd="delcols NUVmag" out=galex.fits
        """
        print(stiltsCommand)
        os.system(stiltsCommand)
        # Fix 4: drop the GALEX RAJ2000/DEJ2000 duplicate coordinate columns
        # explicitly in Python, rather than trusting STILTS' ocmd delcols to
        # have suffixed and then deleted the right name.
        dropped = dropColumnsIfPresent('galex.fits', ['RAJ2000_galex', 'DEJ2000_galex'])
        print('dropped from galex.fits:', dropped)
        NumberGalex = countSources('galex.fits')
        print('GALEX matches: ', NumberGalex)
        os.system('rm gaia.fits')
        if NumberGalex > 0:  # if there are Gaia-GALEX sources, we go on to AllWISE
            stiltsCommand = """
            ~/stilts tapskymatch in=galex.fits inlon=RAJ2000 inlat=DEJ2000 tapurl=http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap taptable='"II/328/allwise"' taplon=RAJ2000 taplat=DEJ2000 sr=0.00027 tapcols=RAJ2000,DEJ2000,W1mag,e_W1mag,W2mag,e_W2mag,W3mag,e_W3mag,W4mag,e_W4mag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag,var,qpm,qph,fdet suffixin= suffixremote=_allwise find=best ocmd="delcols SEP_ARCSEC" out=allwise.fits
            """
            print(stiltsCommand)
            os.system(stiltsCommand)
            dropped = dropColumnsIfPresent('allwise.fits', ['RAJ2000_allwise', 'DEJ2000_allwise'])
            print('dropped from allwise.fits:', dropped)
            NumberAllWise = countSources('allwise.fits')
            print('AllWISE matches: ', NumberAllWise)
            os.system('rm galex.fits')
            if NumberAllWise > 0:
                insertAPs()
                os.system('cp full.fits ' + outputFitsFileName)
                os.system('rm full.fits')
    addFieldToDB(minRA, maxRA, minDec, maxDec, NumberGaia, NumberGalex, NumberAllWise)


if __name__ == "__main__":
    main()
