#!/usr/bin/env python3
"""
make_sky_density.py
Surface density of PaStA sources on the sky, in equatorial coordinates,
Mollweide projection.  Replaces the binary MOC footprint figure produced
by make_moc_sky.py.

The footprint is still readable -- pixels containing no source are left
blank -- but every covered pixel now carries a number rather than a
single bit, so the figure shows the selection function of the catalogue
on the sky instead of only its outline.

Binning uses HEALPix at nside=64, that is order 6, giving 0.84 deg^2 per
pixel.  That is deliberately close to the 1.2 deg diameter field of view
of the GALEX AIS (about 1.1 deg^2): one pixel is roughly one AIS
pointing, so each pixel asks whether that field was observed and how
rich it is, which is the selection function at its natural scale.

Finer binning resolves structure within the tiling -- the gaps along and
between individual AIS scan legs, typically 0.5-1.5 deg thick and 8-18
deg long.  Those gaps are real, and because a great circle runs nearly
east-west at mid declination they read as horizontal stripes, which is
distracting and is a property of how GALEX tiled the sky rather than of
PaStA.  Coarser binning (nside=32) is worse: a pixel straddling the edge
of the footprint is only fractionally filled and reports a genuinely
false low density, appearing as dark speckle.

HEALPix pixels are also close to isotropic in shape.  An earlier version
of this figure binned uniformly in right ascension and in sin(Dec),
which is equal-area but produces cells 1.6 times wider than tall at the
equator and progressively more elongated towards the poles.

Note on the long horizontal stripes that earlier versions of this figure
showed, and that the text above once defended as scan legs: they were
not.  They came from the way the x edges were built (see the comment
below), which left the first quad spanning 359.8 deg of sky.  pcolormesh
drew that quad first, so every masked pixel showed it instead of the
empty-sky background: the Galactic plane was painted over, and the
declination rows where the RA ~ 360 deg column happens to be empty
appeared as full-width grey stripes.  Fixed 2026-08-24.  The figure now
shows the GALEX AIS plane avoidance, in agreement with the 6 per cent
order-6 occupancy at |b| < 5 quoted in Sect. 3.1.  Fine structure along
individual scan legs survives the fix and is real.

The sky coverage quoted in the text (26 785 deg^2, 64.9 per cent) comes
from the MOC at order 8 and is NOT re-derived here.  The covered
fraction printed below is a diagnostic of the binning, not a property of
the catalogue: a pixel counts as covered if it holds even one source, so
the estimate falls with pixel size -- 81.9 per cent at order 5, 76.6 at
order 6, 70.0 at order 7, and 64.9 at order 8, where it reproduces the
MOC value exactly.  Below order 8 it falls again (57.3 per cent at order
9) because the pixels become smaller than the source spacing in sparse
regions, so it starts measuring density rather than footprint.

Input
-----
  pasta1_public.fits  (released catalogue, J2016.0, AB)

Output
------
  fig/pasta_sky_density.{pdf,png}

Requires astropy-healpix.

Run from the paper root directory:
    python make_sky_density.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import astropy.units as u

PASTA_FILE = 'pasta1_public.fits'
OUTDIR = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
NSIDE = 64                 # order 6, 0.84 deg^2 per pixel ~ one AIS field
GRID_RA, GRID_DEC = 2160, 1080     # rendering grid, finer than a pixel

os.makedirs(OUTDIR, exist_ok=True)

mpl.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
})


def ra_to_x(ra_deg):
    """Equatorial RA in degrees to Mollweide x, RA increasing to the left."""
    ra = np.asarray(ra_deg, dtype=float)
    ra_shifted = (ra - 180.0 + 180.0) % 360.0 - 180.0
    return -ra_shifted * np.pi / 180.0


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print('Loading PaStA ...', flush=True)
with fits.open(PASTA_FILE, memmap=True) as hdul:
    ra = np.array(hdul[1].data['ra_2016'], dtype=np.float64)
    dec = np.array(hdul[1].data['dec_2016'], dtype=np.float64)

good = np.isfinite(ra) & np.isfinite(dec)
ra, dec = ra[good], dec[good]
print(f'  {len(ra):,} sources with valid positions.', flush=True)

# ---------------------------------------------------------------------------
# HEALPix binning
# ---------------------------------------------------------------------------
hp = HEALPix(nside=NSIDE, order='ring')
pix_area = hp.pixel_area.to(u.deg ** 2).value
print(f'HEALPix nside={NSIDE}: {hp.npix:,} pixels of {pix_area:.4f} deg^2',
      flush=True)

pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
counts = np.bincount(pix, minlength=hp.npix)

covered = counts > 0
print(f'Covered pixels: {covered.sum():,} of {hp.npix:,} '
      f'({100 * covered.mean():.1f} per cent, '
      f'{covered.sum() * pix_area:.0f} deg^2 -- diagnostic only, '
      f'see docstring)', flush=True)

density = np.where(covered, counts / pix_area, np.nan)
print(f'Density: p1 {np.nanpercentile(density, 1):.0f}, '
      f'median {np.nanmedian(density):.0f}, '
      f'p99 {np.nanpercentile(density, 99):.0f} sources/deg^2', flush=True)

# ---------------------------------------------------------------------------
# Render the HEALPix map onto a regular grid, finer than a pixel, so that
# the pixels keep their true shape in the projection
# ---------------------------------------------------------------------------
ra_edges = np.linspace(0.0, 360.0, GRID_RA + 1)
dec_edges = np.linspace(-90.0, 90.0, GRID_DEC + 1)
ra_cen = 0.5 * (ra_edges[:-1] + ra_edges[1:])
dec_cen = 0.5 * (dec_edges[:-1] + dec_edges[1:])

RA_G, DEC_G = np.meshgrid(ra_cen, dec_cen)
grid_pix = hp.lonlat_to_healpix(RA_G.ravel() * u.deg, DEC_G.ravel() * u.deg)
img = density[grid_pix].reshape(DEC_G.shape)

# RA increases to the left, so x runs from +pi at RA = 0 down to -pi at
# RA = 360.  Do NOT build these edges by calling ra_to_x on ra_edges: that
# function wraps, so the final edge at RA = 360 comes back as +pi instead of
# -pi.  Reversing such an array leaves its first cell spanning 359.8 deg of
# sky, which pcolormesh draws as a full-width band underneath the map; every
# masked pixel then shows that band instead of the empty-sky background, and
# the Galactic plane is painted over.
x_edges = np.radians(180.0 - ra_edges)
y_edges = np.radians(dec_edges)
assert np.all(np.diff(x_edges) < 0), 'x_edges must be monotone'

X, Y = np.meshgrid(x_edges, y_edges)

# ---------------------------------------------------------------------------
# Galactic plane in equatorial coordinates
# ---------------------------------------------------------------------------
l_line = np.linspace(0, 360, 3000)


def split_wrap(ra_deg, dec_deg):
    """Split a track into segments that do not cross the RA wrap."""
    x = ra_to_x(ra_deg)
    y = np.radians(dec_deg)
    brk = np.nonzero(np.abs(np.diff(x)) > np.pi)[0] + 1
    return list(zip(np.split(x, brk), np.split(y, brk)))


gp = SkyCoord(l=l_line * u.deg, b=np.zeros_like(l_line) * u.deg,
              frame='galactic').icrs
gp_segs = split_wrap(gp.ra.deg, gp.dec.deg)

gb_segs = []
for b in (-5.0, 5.0):
    c = SkyCoord(l=l_line * u.deg, b=np.full_like(l_line, b) * u.deg,
                 frame='galactic').icrs
    gb_segs += split_wrap(c.ra.deg, c.dec.deg)

gc = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame='galactic').icrs

# ---------------------------------------------------------------------------
# Figure.  The Mollweide ellipse leaves the corners of the axes box empty;
# the colour bar goes into the lower right one rather than below the map,
# which would make the figure a third taller for no extra information.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(3.5, 1.95))
ax = fig.add_axes([0.005, 0.02, 0.99, 0.96], projection='mollweide')
ax.set_facecolor('#e9e9e9')

vmin = max(1.0, np.nanpercentile(density, 1))
vmax = np.nanpercentile(density, 99.5)
mesh = ax.pcolormesh(X, Y, np.ma.masked_invalid(img),
                     cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
                     shading='flat', rasterized=True)

dark = [pe.Stroke(linewidth=2.4, foreground='#333333'), pe.Normal()]
for xs, ys in gp_segs:
    ax.plot(xs, ys, color='white', lw=1.4, zorder=5, path_effects=dark)
for xs, ys in gb_segs:
    ax.plot(xs, ys, color='white', lw=0.6, ls='--', alpha=0.9, zorder=5,
            path_effects=[pe.Stroke(linewidth=1.5, foreground='#333333'),
                          pe.Normal()])
ax.plot(ra_to_x(gc.ra.deg), np.radians(gc.dec.deg), 'x',
        color='white', ms=4.5, mew=1.1, zorder=6,
        path_effects=[pe.Stroke(linewidth=2.2, foreground='#333333'),
                      pe.Normal()])

ra_ticks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
ax.set_xticks([ra_to_x(h * 15) for h in ra_ticks])
ax.set_xticklabels([f'{h}h' for h in ra_ticks])
ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
ax.set_yticklabels([r'$-60\degree$', r'$-30\degree$', r'$0\degree$',
                    r'$+30\degree$', r'$+60\degree$'])
ax.grid(True, color='gray', ls=':', lw=0.4, alpha=0.35)
ax.tick_params(colors='#333333')

# RA labels sit on the equator, over the data: give them a white halo
for lbl in ax.get_xticklabels():
    lbl.set_fontsize(5.5)
    lbl.set_color('#111111')
    lbl.set_path_effects([pe.Stroke(linewidth=1.8, foreground='white'),
                          pe.Normal()])

cax = fig.add_axes([0.735, 0.135, 0.225, 0.035])
cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend='both')
cb.set_label(r'sources per deg$^{2}$', fontsize=5.5, labelpad=1.5)
cb.ax.tick_params(labelsize=5.0, length=1.8, pad=1.2)
cb.outline.set_linewidth(0.5)

for ext in ('pdf', 'png'):
    path = os.path.join(OUTDIR, f'pasta_sky_density.{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved {path}')

plt.close(fig)
print('Done.')
