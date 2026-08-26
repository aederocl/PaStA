#!/usr/bin/env python3
"""
crossmatch_stats.py
Computes cross-match separation statistics and per-band detection fractions
from pasta.fits for use in §3.3 of PaStA Paper I.

Run from the paper root directory:
    python crossmatch_stats.py

Reads only the coordinate and photometry columns needed (memmap'd — does not
load the full 3.4 GB file into RAM).
"""

import numpy as np
from astropy.io import fits

PASTA = 'pasta.fits'

print('Opening pasta.fits (memmap) ...')
with fits.open(PASTA, memmap=True) as hdul:
    t = hdul[1].data
    ra_g   = np.array(t['raj2000'],         dtype=float)
    dec_g  = np.array(t['dej2000'],         dtype=float)
    ra_gx  = np.array(t['RAJ2000_galex'],   dtype=float)
    dec_gx = np.array(t['DEJ2000_galex'],   dtype=float)
    ra_ws  = np.array(t['RAJ2000_allwise'], dtype=float)
    dec_ws = np.array(t['DEJ2000_allwise'], dtype=float)
    gmag   = np.array(t['phot_g_mean_mag'], dtype=float)
    fuv    = np.array(t['fuv'],   dtype=float)
    nuv    = np.array(t['nuv'],   dtype=float)
    w1     = np.array(t['W1mag'], dtype=float)
    w2     = np.array(t['W2mag'], dtype=float)
    w3     = np.array(t['W3mag'], dtype=float)
    w4     = np.array(t['W4mag'], dtype=float)

N = len(ra_g)
print(f'\n=== Catalogue ===')
print(f'Total sources : {N:,}')
print(f'Gaia G range  : {np.nanmin(gmag):.2f} – {np.nanmax(gmag):.2f}')
print(f'G percentiles (1 / 50 / 99): '
      f'{np.nanpercentile(gmag,1):.2f} / '
      f'{np.nanpercentile(gmag,50):.2f} / '
      f'{np.nanpercentile(gmag,99):.2f}')

# ---------------------------------------------------------------------------
# GALEX
# ---------------------------------------------------------------------------
galex_mask = np.isfinite(ra_gx) & np.isfinite(dec_gx)
fuv_mask   = np.isfinite(fuv)   & (fuv > 0)
nuv_mask   = np.isfinite(nuv)   & (nuv > 0)

print(f'\n=== GALEX ===')
print(f'Sources with GALEX position : {galex_mask.sum():,}  ({100*galex_mask.mean():.1f} %)')
print(f'FUV detections              : {fuv_mask.sum():,}  ({100*fuv_mask.mean():.2f} %)')
print(f'NUV detections              : {nuv_mask.sum():,}  ({100*nuv_mask.mean():.1f} %)')

dra_gx  = (ra_gx[galex_mask] - ra_g[galex_mask]) * \
           np.cos(np.radians(dec_g[galex_mask]))
ddec_gx = dec_gx[galex_mask] - dec_g[galex_mask]
sep_gx  = np.sqrt(dra_gx**2 + ddec_gx**2) * 3600   # arcsec

print(f'Gaia – GALEX separation     : '
      f'median={np.median(sep_gx):.3f}"  '
      f'95th={np.percentile(sep_gx, 95):.3f}"  '
      f'99th={np.percentile(sep_gx, 99):.3f}"')

# ---------------------------------------------------------------------------
# AllWISE
# ---------------------------------------------------------------------------
wise_mask = np.isfinite(ra_ws) & np.isfinite(dec_ws)

print(f'\n=== AllWISE ===')
print(f'Sources with AllWISE position : {wise_mask.sum():,}  ({100*wise_mask.mean():.1f} %)')
for band, arr in [('W1', w1), ('W2', w2), ('W3', w3), ('W4', w4)]:
    m = np.isfinite(arr) & (arr > 0)
    print(f'{band} detections                : {m.sum():,}  ({100*m.mean():.1f} %)')

dra_ws  = (ra_ws[wise_mask] - ra_g[wise_mask]) * \
           np.cos(np.radians(dec_g[wise_mask]))
ddec_ws = dec_ws[wise_mask] - dec_g[wise_mask]
sep_ws  = np.sqrt(dra_ws**2 + ddec_ws**2) * 3600

print(f'Gaia – AllWISE separation     : '
      f'median={np.median(sep_ws):.3f}"  '
      f'95th={np.percentile(sep_ws, 95):.3f}"  '
      f'99th={np.percentile(sep_ws, 99):.3f}"')
