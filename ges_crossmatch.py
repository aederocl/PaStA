#!/usr/bin/env python3
"""
ges_crossmatch.py
Cross-match Gaia ESO Survey DR5 against PaStA.

For each GES source, find the nearest PaStA source within MATCH_RADIUS
arcsec (Gaia J2000 position).  Produces:
  ges_pasta_matched.fits   — matched table with GES params + PaStA photometry
  ges_pasta_matched.csv    — same, human-readable

Also prints match statistics to stdout.

Run from the paper root directory:
    python ges_crossmatch.py
"""

import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

PASTA_FILE  = 'pasta.fits'
GES_FILE    = os.environ.get('GES_DR5_FITS', 'Gaia_ESO_DR5.fits')
MATCH_RADIUS = 1.0   # arcsec
OUT_FITS    = 'ges_pasta_matched.fits'
OUT_CSV     = 'ges_pasta_matched.csv'

# ---------------------------------------------------------------------------
# Load PaStA coordinates and photometry
# ---------------------------------------------------------------------------
print('Loading PaStA ...')
with fits.open(PASTA_FILE, memmap=True) as h:
    p = h[1].data
    p_sid  = np.array(p['source_id'],          dtype=np.int64)
    p_ra   = np.array(p['raj2000'],             dtype=np.float64)
    p_dec  = np.array(p['dej2000'],             dtype=np.float64)
    p_dist = np.array(p['distance'],            dtype=np.float32)
    p_plx  = np.array(p['parallax'],            dtype=np.float32)
    # Gaia magnitudes and flux-derived errors
    p_g    = np.array(p['phot_g_mean_mag'],     dtype=np.float32)
    p_gf   = np.array(p['phot_g_mean_flux'],    dtype=np.float64)
    p_gfe  = np.array(p['phot_g_mean_flux_error'], dtype=np.float64)
    p_bp   = np.array(p['phot_bp_mean_mag'],    dtype=np.float32)
    p_bpf  = np.array(p['phot_bp_mean_flux'],   dtype=np.float64)
    p_bpfe = np.array(p['phot_bp_mean_flux_error'], dtype=np.float64)
    p_rp   = np.array(p['phot_rp_mean_mag'],    dtype=np.float32)
    p_rpf  = np.array(p['phot_rp_mean_flux'],   dtype=np.float64)
    p_rpfe = np.array(p['phot_rp_mean_flux_error'], dtype=np.float64)
    # UV
    p_fuv  = np.array(p['fuv'],      dtype=np.float32)
    p_efuv = np.array(p['e_FUVmag'], dtype=np.float32)
    p_nuv  = np.array(p['nuv'],      dtype=np.float32)
    p_enuv = np.array(p['e_NUVmag'], dtype=np.float32)
    # 2MASS
    p_j    = np.array(p['Jmag'],     dtype=np.float32)
    p_ej   = np.array(p['e_Jmag'],   dtype=np.float32)
    p_h    = np.array(p['Hmag'],     dtype=np.float32)
    p_eh   = np.array(p['e_Hmag'],   dtype=np.float32)
    p_k    = np.array(p['Kmag'],     dtype=np.float32)
    p_ek   = np.array(p['e_Kmag'],   dtype=np.float32)
    # AllWISE
    p_w1   = np.array(p['W1mag'],    dtype=np.float32)
    p_ew1  = np.array(p['e_W1mag'],  dtype=np.float32)
    p_w2   = np.array(p['W2mag'],    dtype=np.float32)
    p_ew2  = np.array(p['e_W2mag'],  dtype=np.float32)
    p_w3   = np.array(p['W3mag'],    dtype=np.float32)
    p_ew3  = np.array(p['e_W3mag'],  dtype=np.float32)
    p_w4   = np.array(p['W4mag'],    dtype=np.float32)
    p_ew4  = np.array(p['e_W4mag'],  dtype=np.float32)

print(f'  {len(p_ra):,} PaStA sources.')

# Gaia magnitude errors from flux SNR: σ_mag = 1.0857 × (flux_err / flux)
def flux_to_mag_err(flux, flux_err):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(
            (flux > 0) & np.isfinite(flux) & np.isfinite(flux_err),
            1.0857 * flux_err / flux,
            np.nan
        ).astype(np.float32)

p_eg  = flux_to_mag_err(p_gf,  p_gfe)
p_ebp = flux_to_mag_err(p_bpf, p_bpfe)
p_erp = flux_to_mag_err(p_rpf, p_rpfe)

# ---------------------------------------------------------------------------
# Load GES DR5
# ---------------------------------------------------------------------------
print('Loading GES DR5 ...')
with fits.open(GES_FILE) as h:
    g = h[1].data
    g_obj  = np.array([s.strip() for s in g['OBJECT']])
    g_ra   = np.array(g['RA'],          dtype=np.float64)
    g_dec  = np.array(g['DECLINATION'], dtype=np.float64)
    g_teff = np.array(g['TEFF'],        dtype=np.float32)
    g_eteff= np.array(g['E_TEFF'],      dtype=np.float32)
    g_logg = np.array(g['LOGG'],        dtype=np.float32)
    g_elogg= np.array(g['E_LOGG'],      dtype=np.float32)
    g_feh  = np.array(g['FEH'],         dtype=np.float32)
    g_efeh = np.array(g['E_FEH'],       dtype=np.float32)
    g_vrad = np.array(g['VRAD'],        dtype=np.float32)

print(f'  {len(g_ra):,} GES sources.')

# Only keep GES sources with valid coordinates
ges_ok = np.isfinite(g_ra) & np.isfinite(g_dec)
print(f'  {ges_ok.sum():,} with valid coordinates.')

# ---------------------------------------------------------------------------
# Sky-coordinate cross-match
# ---------------------------------------------------------------------------
print(f'Cross-matching (radius = {MATCH_RADIUS} arcsec) ...')

pasta_sky = SkyCoord(ra=p_ra * u.deg, dec=p_dec * u.deg)
ges_sky   = SkyCoord(ra=g_ra[ges_ok] * u.deg, dec=g_dec[ges_ok] * u.deg)

idx, sep, _ = ges_sky.match_to_catalog_sky(pasta_sky)
matched     = sep.arcsec <= MATCH_RADIUS

print(f'  GES sources with a PaStA match: {matched.sum():,} / {ges_ok.sum():,}  '
      f'({100*matched.mean():.1f}%)')
print(f'  Separation (matched): median {np.median(sep[matched].arcsec):.3f}"  '
      f'max {sep[matched].arcsec.max():.3f}"')

# Indices into original (full-length) GES array
ges_full_idx   = np.where(ges_ok)[0][matched]
pasta_idx      = idx[matched]

n_match = matched.sum()

# GES parameter completeness in the matched sample
ok_t = np.isfinite(g_teff[ges_full_idx]) & (g_teff[ges_full_idx] > 0)
ok_g = np.isfinite(g_logg[ges_full_idx])
ok_f = np.isfinite(g_feh[ges_full_idx])
print(f'  Teff valid in matched: {ok_t.sum():,}')
print(f'  logg valid in matched: {ok_g.sum():,}')
print(f'  [Fe/H] valid in matched: {ok_f.sum():,}')
print(f'  All three valid: {(ok_t & ok_g & ok_f).sum():,}')

# PaStA brightness check: saturation in AllWISE (<8 mag) and GALEX absence
w1_m  = p_w1[pasta_idx]
nuv_m = p_nuv[pasta_idx]
g_m   = p_g[pasta_idx]
sat_wise  = np.isfinite(w1_m) & (w1_m < 8.0)
miss_nuv  = ~(np.isfinite(nuv_m) & (nuv_m > 0))
print(f'\n  Bright-star caveats in matched sample:')
print(f'    W1 < 8 mag (AllWISE saturation risk): {sat_wise.sum():,}')
print(f'    NUV missing or zero:                  {miss_nuv.sum():,}')

# ---------------------------------------------------------------------------
# Build output table
# ---------------------------------------------------------------------------
print('\nBuilding output table ...')

from astropy.table import Table

t_out = Table()

# GES columns
t_out['ges_object'] = g_obj[ges_full_idx]
t_out['ges_ra']     = g_ra [ges_full_idx]
t_out['ges_dec']    = g_dec[ges_full_idx]
t_out['ges_teff']   = g_teff [ges_full_idx]
t_out['ges_e_teff'] = g_eteff[ges_full_idx]
t_out['ges_logg']   = g_logg [ges_full_idx]
t_out['ges_e_logg'] = g_elogg[ges_full_idx]
t_out['ges_feh']    = g_feh  [ges_full_idx]
t_out['ges_e_feh']  = g_efeh [ges_full_idx]
t_out['ges_vrad']   = g_vrad [ges_full_idx]

# Match quality
t_out['sep_arcsec'] = sep[matched].arcsec.astype(np.float32)

# PaStA columns
t_out['pasta_source_id'] = p_sid [pasta_idx]
t_out['pasta_ra']        = p_ra  [pasta_idx]
t_out['pasta_dec']       = p_dec [pasta_idx]
t_out['pasta_dist_pc']   = p_dist[pasta_idx]
t_out['pasta_parallax']  = p_plx [pasta_idx]

# Photometry
for col, arr, earr in [
    ('FUV', p_fuv, p_efuv),
    ('NUV', p_nuv, p_enuv),
    ('BP',  p_bp,  p_ebp),
    ('G',   p_g,   p_eg),
    ('RP',  p_rp,  p_erp),
    ('J',   p_j,   p_ej),
    ('H',   p_h,   p_eh),
    ('K',   p_k,   p_ek),
    ('W1',  p_w1,  p_ew1),
    ('W2',  p_w2,  p_ew2),
    ('W3',  p_w3,  p_ew3),
    ('W4',  p_w4,  p_ew4),
]:
    t_out[f'pasta_{col}']     = arr[pasta_idx]
    t_out[f'pasta_e_{col}']   = earr[pasta_idx]

# Save
t_out.write(OUT_FITS, overwrite=True)
t_out.write(OUT_CSV,  overwrite=True, format='csv')
print(f'Saved {OUT_FITS}  ({n_match:,} rows)')
print(f'Saved {OUT_CSV}')
print('Done.')
