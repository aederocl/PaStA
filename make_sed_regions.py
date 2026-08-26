#!/usr/bin/env python3
"""
make_sed_regions.py
Average SEDs for nine CMD regions in PaStA Paper I (Sect. 5).

Produces two figures saved to fig/:
  cmd_regions.{pdf,png}   — CMD density map with region boxes overlaid
  sed_regions.{pdf,png}   — 3x3 grid of average SEDs (median + 16/84 percentile)

SED units: lambda*F_lambda normalised to the G-band value of each individual
source, then median and 16/84 percentile taken across all sources in the region.

Photometric system
------------------
Input is pasta1_public.fits, in which every band is already on the AB system
(Sect. 3.1 of the paper), so the single zero point 3631 Jy applies to all
twelve bands:  F_nu = 3631 Jy * 10^(-0.4 m_AB).

This replaces the earlier version of this script, which read the native-system
pasta.fits and assigned 3631 Jy to the Gaia bands as well -- but those
magnitudes were VEGAMAG, and the RP VEGAMAG-to-AB offset is 0.356 mag, not
negligible. Because the SEDs are normalised to G (offset 0.114 mag) the error
did not cancel: RP came out ~25 per cent too high, BP ~9 per cent too low, and
every non-Gaia band ~10 per cent too low. Reading the AB catalogue removes the
problem by construction.

CMD regions
-----------
Region boundaries are quoted in the AB system, obtained from the original
Vega-system boxes by the exact shifts implied by Table 4 of the paper --
(BP-RP) by -0.3406 mag and M_G by +0.1136 mag -- and rounded to two decimals.
The boxes therefore keep the spectral-type correspondence they were given by
Pecaut & Mamajek (2013); they are not re-cut on round AB numbers, which would
have silently changed which stars belong to each class.

Run from the paper root directory:
    python make_sed_regions.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from astropy.io import fits

# ---------------------------------------------------------------------------
# Band definitions: (label, FITS column, lambda_pivot [micron])
# All bands are AB in pasta1_public.fits, so the zero point is 3631 Jy
# throughout and drops out of the G-normalised ratio entirely.
#
# Wavelengths are the PIVOT wavelengths from the SVO Filter Profile Service,
# taken from one consistent revision per instrument (GAIA3 for Gaia). The
# pivot wavelength is defined by
#     lambda_pivot^2 = int(lambda T dlambda) / int(T dlambda / lambda)
# so that F_lambda = c F_nu / lambda_pivot^2 holds exactly for ANY source
# spectrum. That is what makes the lambda*F_lambda = F_nu c / lambda step
# below exact rather than approximate, and it is the natural partner of the
# AB system, which is defined on F_nu.
#
# Deliberately NOT the effective wavelength: SVO computes lambda_eff against
# Vega, so it is spectrum-dependent (for the very broad Gaia G band it lands
# at 5822 A versus a mean of 6720 A). PaStA spans OB stars to M dwarfs to
# white dwarfs, so a Vega-weighted wavelength would bias exactly the SED
# shapes this script measures. SVO also reports lambda_eff = lambda_mean for
# 2MASS and WISE, i.e. no spectrum weighting was applied there at all.
# ---------------------------------------------------------------------------
BANDS = [
    ('FUV', 'FUV', 0.1535079),
    ('NUV', 'NUV', 0.2300785),
    ('BP',  'BP',  0.5109712),
    ('G',   'G',   0.6217590),
    ('RP',  'RP',  0.7769023),
    ('J',   'J',   1.2393089),
    ('H',   'H',   1.6494947),
    ('K',   'Ks',  2.1638606),
    ('W1',  'W1',  3.3682213),
    ('W2',  'W2',  4.6179057),
    ('W3',  'W3',  12.0718118),
    ('W4',  'W4',  22.1944039),
]

BAND_NAMES  = [b[0] for b in BANDS]
BAND_COLS   = [b[1] for b in BANDS]
BAND_LAMBDA = np.array([b[2] for b in BANDS])   # micron
G_IDX       = BAND_NAMES.index('G')

# ---------------------------------------------------------------------------
# CMD region definitions, AB system.
# Each entry: label, (BP-RP) range, M_G range, colour for plot.
# ---------------------------------------------------------------------------
REGIONS = [
    {'label': 'OB',  'tex': 'OB',        'bprp': (-0.94, -0.24), 'mg': (-4.89,  2.11), 'color': '#4C72B0'},
    {'label': 'A',   'tex': 'A',         'bprp': (-0.24,  0.11), 'mg': ( 0.11,  4.11), 'color': '#64B5CD'},
    {'label': 'F',   'tex': 'F',         'bprp': ( 0.11,  0.31), 'mg': ( 2.61,  5.11), 'color': '#CCBB44'},
    {'label': 'G',   'tex': 'G',         'bprp': ( 0.31,  0.51), 'mg': ( 4.11,  6.11), 'color': '#EE8833'},
    {'label': 'K',   'tex': 'K',         'bprp': ( 0.51,  1.16), 'mg': ( 5.61,  8.11), 'color': '#CC3311'},
    {'label': 'M',   'tex': 'M',         'bprp': ( 1.16,  3.66), 'mg': ( 8.11, 17.11), 'color': '#882255'},
    {'label': 'RGB', 'tex': 'RGB',       'bprp': ( 0.66,  1.46), 'mg': (-2.89,  0.41), 'color': '#AA4499'},
    {'label': 'RC',  'tex': 'Red Clump', 'bprp': ( 0.66,  0.96), 'mg': ( 0.41,  1.41), 'color': '#8B4513'},
    {'label': 'WD',  'tex': 'WD',        'bprp': (-0.74,  0.06), 'mg': (11.11, 16.11), 'color': '#117733'},
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
PASTA = 'pasta1_public.fits'
print(f'Reading {PASTA} ...')
with fits.open(PASTA, memmap=True) as hdul:
    t    = hdul[1].data
    dist = np.array(t['distance'], dtype=np.float32)
    mags = np.column_stack([
        np.array(t[col], dtype=np.float32) for col in BAND_COLS
    ])   # shape (N, 12)
    errs = np.column_stack([
        np.array(t['e_' + col], dtype=np.float32) for col in BAND_COLS
    ])   # shape (N, 12)

g_  = mags[:, G_IDX]
bp_ = mags[:, BAND_NAMES.index('BP')]
rp_ = mags[:, BAND_NAMES.index('RP')]

print(f'  {len(dist):,} sources loaded.')

# ---------------------------------------------------------------------------
# Compute CMD quantities (AB)
# ---------------------------------------------------------------------------
ok = (np.isfinite(dist) & (dist > 0) &
      np.isfinite(g_) & np.isfinite(bp_) & np.isfinite(rp_))
mg_all   = g_[ok] - 5.0 * np.log10(dist[ok]) + 5.0
bprp_all = bp_[ok] - rp_[ok]
mags_ok  = mags[ok]
errs_ok  = errs[ok]

print(f'  {ok.sum():,} sources with valid distance and optical photometry.')

# ---------------------------------------------------------------------------
# Detection criterion.
#
# AllWISE reports a magnitude in W3 and W4 for almost every source, but sets
# the uncertainty to null when the source falls below the detection threshold;
# that magnitude is then a 95 per cent confidence UPPER LIMIT, not a
# measurement. Testing only isfinite(mag) therefore counts upper limits as
# faint detections: 94.7 per cent of W4 values and 59.8 per cent of W3 values
# in PaStA carry no uncertainty at all. Because those limits cluster at the
# survey sensitivity floor (W4 median 15.55 AB, 16/84 spread 0.65 mag, against
# 1.62 mag for real detections), dividing them by a varying G flux manufactures
# an apparent mid-infrared excess in every CMD region -- the spurious W3/W4
# upturn seen in earlier versions of this figure.
#
# A band therefore contributes to a source only if it has a finite magnitude
# AND a finite uncertainty. The presence of an uncertainty is exactly the flag
# that separates a measurement from a censored upper limit, and censoring is
# confined to W3 (59.8 per cent), W4 (94.7 per cent) and ~1 per cent of Ks.
#
# No further signal-to-noise threshold is imposed, deliberately. GALEX and Gaia
# never report a magnitude without an uncertainty, so for those bands a low SNR
# means a noisy measurement, not a limit; discarding such values preferentially
# removes the faint ones and biases the median upward. An earlier version of
# this script applied a uniform SNR > 3 cut and inflated the M-dwarf NUV median
# by a factor of 1.69 for exactly that reason (the G and K regions were
# unaffected, at 1.01 and 0.99). A median is robust to noise but not to
# censoring, so the right criterion is censoring alone.
# ---------------------------------------------------------------------------
detected = np.isfinite(mags_ok) & np.isfinite(errs_ok)

# Fraction of a region's sources for which the band median is a real estimate
# rather than an upper limit. Below 50 per cent more than half the region is
# undetected, so the median of the detections is strictly an upper bound on the
# median of the population and is drawn as such.
DETFRAC_MIN = 0.50

# ---------------------------------------------------------------------------
# Magnitudes -> F_nu [Jy] -> lambda*F_lambda, normalised to each star's G band.
#
#   F_nu = 3631 Jy * 10^(-0.4 m_AB)                      (all twelve bands)
#   (lambda F_lambda)_norm = (F_nu_band / F_nu_G) * (lambda_G / lambda_band)
#
# The 3631 Jy cancels in the ratio, which is precisely the simplification the
# AB harmonisation buys us.
# ---------------------------------------------------------------------------
print('Converting magnitudes to normalised lambda*F_lambda ...')

fnu = np.where(detected,
               10.0 ** (-0.4 * mags_ok),
               np.nan)

fnu_G = fnu[:, G_IDX, np.newaxis]
with np.errstate(invalid='ignore', divide='ignore'):
    lflf_norm = (fnu / fnu_G) * (BAND_LAMBDA[G_IDX] / BAND_LAMBDA[np.newaxis, :])

bad_G = ~np.isfinite(fnu_G[:, 0])
lflf_norm[bad_G, :] = np.nan

print(f'  {(~bad_G).sum():,} sources with valid G-band normalisation.')

# ---------------------------------------------------------------------------
# Per-region statistics
# ---------------------------------------------------------------------------
print('Computing per-region statistics ...')
region_stats = []
for reg in REGIONS:
    blo, bhi = reg['bprp']
    mlo, mhi = reg['mg']
    mask = ((bprp_all >= blo) & (bprp_all < bhi) &
            (mg_all   >= mlo) & (mg_all   < mhi))
    data    = lflf_norm[mask]
    n_total = int(mask.sum())

    med    = np.full(len(BANDS), np.nan)
    p16    = np.full(len(BANDS), np.nan)
    p84    = np.full(len(BANDS), np.nan)
    n_band = np.zeros(len(BANDS), dtype=int)

    for j in range(len(BANDS)):
        finite = data[:, j][np.isfinite(data[:, j])]
        if len(finite) >= 10:
            med[j]    = np.median(finite)
            p16[j]    = np.percentile(finite, 16)
            p84[j]    = np.percentile(finite, 84)
            n_band[j] = len(finite)

    detfrac  = n_band / n_total if n_total else np.zeros(len(BANDS))
    is_limit = detfrac < DETFRAC_MIN

    region_stats.append({'n': n_total, 'med': med, 'p16': p16, 'p84': p84,
                         'n_band': n_band, 'detfrac': detfrac,
                         'is_limit': is_limit})
    print(f'  {reg["label"]:4s}: {n_total:>9,} sources   '
          f'upper limits in: '
          f'{", ".join(BAND_NAMES[j] for j in range(len(BANDS)) if is_limit[j]) or "none"}')

# ---------------------------------------------------------------------------
# LaTeX snippet for Table 3 of the paper
# ---------------------------------------------------------------------------
print('\n--- Table 3 body (AB), paste into pasta_paper1.tex ---')
_TEX = {'OB': 'OB MS', 'A': 'A MS', 'F': 'F MS', 'G': 'G MS', 'K': 'K MS',
        'M': 'M MS', 'RGB': 'RGB', 'RC': 'Red Clump', 'WD': 'White Dwarfs'}
for reg, st in zip(REGIONS, region_stats):
    blo, bhi = reg['bprp']
    mlo, mhi = reg['mg']
    n = f'{st["n"]:,}'.replace(',', '\\,')
    print(f'        {_TEX[reg["label"]]:<12s} & ${blo:+.2f}$--${bhi:+.2f}$ '
          f'& ${mlo:+.2f}$--${mhi:+.2f}$ & {n} \\\\')
print('--- end snippet ---\n')

print('--- measured (uncensored) fraction per region and band (per cent) ---')
print(f'{"region":6s}' + ''.join(f'{b:>7s}' for b in BAND_NAMES))
for reg, st in zip(REGIONS, region_stats):
    print(f'{reg["label"]:6s}' + ''.join(f'{100*f:7.1f}' for f in st['detfrac']))
print('--- end ---\n')

# ---------------------------------------------------------------------------
# Figure 1: CMD density map with region boxes
# ---------------------------------------------------------------------------
print('Making CMD figure ...')
BPRP_MIN, BPRP_MAX = -1.0, 3.7
MG_MIN,   MG_MAX   = -5.0, 17.2

counts, xedges, yedges = np.histogram2d(
    bprp_all, mg_all, bins=[350, 450],
    range=[[BPRP_MIN, BPRP_MAX], [MG_MIN, MG_MAX]]
)

fig_cmd, ax_cmd = plt.subplots(figsize=(6, 8))
ax_cmd.pcolormesh(xedges, yedges, counts.T,
                  norm=mcolors.LogNorm(vmin=1, vmax=counts.max()),
                  cmap='magma_r', rasterized=True)

for reg in REGIONS:
    blo, bhi = reg['bprp']
    mlo, mhi = reg['mg']
    ax_cmd.add_patch(mpatches.Rectangle(
        (blo, mlo), bhi - blo, mhi - mlo,
        linewidth=1.5, edgecolor=reg['color'], facecolor='none', zorder=3))
    ax_cmd.text((blo + bhi) / 2, mlo + 0.15 * (mhi - mlo),
                reg['label'], color=reg['color'], fontsize=7,
                ha='center', va='bottom', fontweight='bold', zorder=4)

ax_cmd.set_xlim(BPRP_MIN, BPRP_MAX)
ax_cmd.set_ylim(MG_MAX, MG_MIN)
ax_cmd.set_xlabel(r'$(G_{\rm BP} - G_{\rm RP})_{\rm AB}$  [mag]', fontsize=12)
ax_cmd.set_ylabel(r'$M_{G,\,\rm AB}$  [mag]', fontsize=12)
ax_cmd.grid(True, color='grey', lw=0.3, alpha=0.4)
fig_cmd.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2: 3x3 SED panels
# ---------------------------------------------------------------------------
print('Making SED figure ...')
fig_sed, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=False)
axes = axes.flatten()

for i, (reg, stats) in enumerate(zip(REGIONS, region_stats)):
    ax    = axes[i]
    med, p16, p84 = stats['med'], stats['p16'], stats['p84']
    is_limit = stats['is_limit']
    color = reg['color']

    for j in range(len(BANDS)):
        if not np.isfinite(med[j]):
            continue
        if is_limit[j]:
            # Fewer than half the region is detected in this band, so the
            # median of the detections bounds the median of the population
            # from above: draw it as an upper limit, with no error bar.
            ax.errorbar(BAND_LAMBDA[j], med[j],
                        yerr=[[med[j] * 0.55], [0.0]], uplims=True,
                        fmt='_', color=color, ms=6, lw=1.0,
                        elinewidth=1.0, alpha=0.75, zorder=3)
        else:
            ax.errorbar(BAND_LAMBDA[j], med[j],
                        yerr=[[med[j] - p16[j]], [p84[j] - med[j]]],
                        fmt='o', color=color, ms=5, lw=1.2,
                        elinewidth=1.0, capsize=2, zorder=3)

    # Connect only the genuine measurements; upper limits are left unjoined so
    # the eye does not read a sequence through them.
    solid = [j for j in range(len(BANDS))
             if np.isfinite(med[j]) and not is_limit[j]]
    if solid:
        ax.plot(BAND_LAMBDA[solid], med[solid],
                '-', color=color, lw=1.0, alpha=0.6, zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.10, 30)
    ax.set_title(f'{reg["tex"]}   ($N={stats["n"]:,}$)',
                 fontsize=9, color=color, fontweight='bold')
    ax.grid(True, which='both', lw=0.3, alpha=0.4)
    ax.axhline(1.0, color='grey', lw=0.6, ls='--', alpha=0.5)

    if i >= 6:
        ax.set_xlabel(r'$\lambda$  [$\mu$m]', fontsize=9)
    if i % 3 == 0:
        ax.set_ylabel(r'$(\lambda F_\lambda) / (\lambda F_\lambda)_G$', fontsize=8)

    if i == 0:
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(BAND_LAMBDA)
        ax2.set_xticklabels(BAND_NAMES, fontsize=6, rotation=45)

fig_sed.suptitle(
    r'Average SEDs per CMD region  '
    r'(median $\pm$ 16/84 percentile, normalised to $G$-band, AB system; '
    r'arrows are upper limits)',
    fontsize=11)
fig_sed.tight_layout(rect=[0, 0, 1, 0.97])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
outdir = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
os.makedirs(outdir, exist_ok=True)

for name, fig in [('cmd_regions', fig_cmd), ('sed_regions', fig_sed)]:
    for ext in ('pdf', 'png'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')

# ---------------------------------------------------------------------------
# Persist the region medians so the outlier-flag script can reuse them
# ---------------------------------------------------------------------------
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'sed_region_stats_ab.npz'),
         band_names=np.array(BAND_NAMES),
         band_lambda=BAND_LAMBDA,
         labels=np.array([r['label'] for r in REGIONS]),
         med=np.array([s['med'] for s in region_stats]),
         p16=np.array([s['p16'] for s in region_stats]),
         p84=np.array([s['p84'] for s in region_stats]),
         n=np.array([s['n'] for s in region_stats]))
print('Saved sed_region_stats_ab.npz')
