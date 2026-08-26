#!/usr/bin/env python3
"""
make_moc_sky.py
Build a Multi-Order Coverage (MOC) map of the PaStA catalogue and plot it
as a Mollweide sky projection in equatorial coordinates.

The MOC captures the actual sky footprint; the figure replaces the two
previous sky maps (survey-footprint figure and the aitoff density plot).

Output
------
  fig/pasta_moc.{pdf,png}

Run from the paper root directory:
    python make_moc_sky.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astropy.io import fits
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u
from mocpy import MOC, WCS as MocWCS

PASTA_FILE = 'pasta.fits'
OUTDIR     = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
MOC_ORDER  = 8          # 0.052 deg² per pixel — good footprint resolution
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load coordinates
# ---------------------------------------------------------------------------
print('Loading PaStA coordinates ...', flush=True)
with fits.open(PASTA_FILE, memmap=True) as hdul:
    ra  = np.array(hdul[1].data['raj2000'], dtype=np.float64)
    dec = np.array(hdul[1].data['dej2000'], dtype=np.float64)

valid = np.isfinite(ra) & np.isfinite(dec)
ra, dec = ra[valid], dec[valid]
print(f'  {len(ra):,} sources with valid coordinates.', flush=True)

# ---------------------------------------------------------------------------
# Build MOC
# ---------------------------------------------------------------------------
print(f'Building MOC at order {MOC_ORDER} ...', flush=True)
moc = MOC.from_lonlat(ra * u.deg, dec * u.deg, max_norder=MOC_ORDER)
sky_fraction = moc.sky_fraction
print(f'  Sky fraction covered: {100*sky_fraction:.2f} per cent '
      f'({sky_fraction*41253:.0f} deg²)', flush=True)

# ---------------------------------------------------------------------------
# Galactic plane and Galactic centre marker
# ---------------------------------------------------------------------------
# Galactic plane: l = 0..360, b = 0
l_vals  = np.linspace(0, 360, 2000) * u.deg
b_zeros = np.zeros(2000) * u.deg
gal_plane = SkyCoord(l=l_vals, b=b_zeros, frame=Galactic())

# Galactic centre
gal_centre = SkyCoord(l=0*u.deg, b=0*u.deg, frame=Galactic())

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print('Plotting ...', flush=True)
fig = plt.figure(figsize=(10, 5.2))

with MocWCS(
    fig,
    fov=360 * u.deg,
    center=SkyCoord(0, 0, unit='deg', frame='icrs'),
    coordsys='icrs',
    projection='MOL',
) as wcs:
    ax = fig.add_subplot(projection=wcs)

    # PaStA footprint
    moc.fill(ax, wcs,
             alpha=0.75, fill=True,
             color='#2171b5', linewidth=0)
    moc.border(ax, wcs,
               alpha=0.6,
               color='#08306b', linewidth=0.4)

    # Galactic plane — split at RA wrap to avoid spurious horizontal lines
    ra_gp  = gal_plane.icrs.ra.deg
    dec_gp = gal_plane.icrs.dec.deg
    # Detect discontinuities (>180° jump in RA)
    breaks = np.where(np.abs(np.diff(ra_gp)) > 180)[0] + 1
    segs   = np.split(np.column_stack([ra_gp, dec_gp]), breaks)
    for seg in segs:
        coords = SkyCoord(ra=seg[:, 0]*u.deg, dec=seg[:, 1]*u.deg)
        ax.plot_coord(coords, color='white', lw=1.2, ls='--',
                      path_effects=[pe.Stroke(linewidth=2.4,
                                              foreground='#444444'),
                                    pe.Normal()])

    # Galactic centre marker
    ax.plot_coord(gal_centre.icrs, 'x', color='white', ms=8, mew=1.5,
                  path_effects=[pe.Stroke(linewidth=3, foreground='#444444'),
                                pe.Normal()])

    # Axis labels
    ax.set_xlabel('Right Ascension (J2000)', fontsize=11)
    ax.set_ylabel('Declination (J2000)', fontsize=11)
    ax.coords[0].set_ticklabel(size=9)
    ax.coords[1].set_ticklabel(size=9)

    # Annotation
    ax.text(0.98, 0.04,
            f'PaStA: {len(ra)/1e6:.2f}M sources\n'
            f'Sky coverage: {100*sky_fraction:.1f}\\,per cent '
            f'({sky_fraction*41253:.0f}\\,deg$^2$)',
            transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.5,
            color='white',
            path_effects=[pe.Stroke(linewidth=2, foreground='#222222'),
                          pe.Normal()])

fig.tight_layout(pad=0.3)

for ext in ('pdf', 'png'):
    path = os.path.join(OUTDIR, f'pasta_moc.{ext}')
    fig.savefig(path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved {path}')

plt.close(fig)
print('Done.')
