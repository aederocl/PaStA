#!/usr/bin/env python3
"""
make_eda.py
Exploratory Data Analysis figures for PaStA Paper I (Sect. 4).

Outputs
-------
  fig/cc_diagrams.{pdf,png}    — two-panel colour-colour: (NUV-G) vs (BP-RP)
                                  and (BP-RP) vs (J-W1)
  fig/mag_histograms.{pdf,png} — 3x4 grid of per-band magnitude distributions

Also prints the measured-fraction table as ready-to-paste LaTeX rows.

Photometric system
------------------
Input is pasta1_public.fits, so every magnitude is AB (Sect. 3.1 of the paper)
and every source count is post-deduplication. Colours are therefore offset from
their Vega-system equivalents: (BP-RP) by -0.3406 mag, (NUV-G) by -0.1136,
(J-W1) by -1.7795.

What counts as a measurement
----------------------------
AllWISE reports a magnitude in W3 and W4 for almost every source but sets the
uncertainty to null when the source is below the detection threshold; that
value is then a 95 per cent confidence upper limit, not a measurement. The
earlier version of this script counted a band as detected whenever the
magnitude was finite and positive, which reported W1-W4 at 100 per cent in
every magnitude bin and disagreed with the average SEDs of Sect. 5.

A band is counted here only when it has both a finite magnitude and a finite
uncertainty. No signal-to-noise threshold is applied on top: GALEX and Gaia
never report a magnitude without an uncertainty, so for those bands a low SNR
is a noisy measurement rather than a limit, and cutting on it would bias the
statistics toward the bright end. See make_sed_regions.py for the same
criterion applied to the average SEDs.

Run from the paper root directory:
    python make_eda.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits

PASTA  = 'pasta1_public.fits'
OUTDIR = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
os.makedirs(OUTDIR, exist_ok=True)

# (name, FITS column, plot colour)
BANDS = [
    ('FUV', 'FUV', '#9B59B6'),
    ('NUV', 'NUV', '#3498DB'),
    ('BP',  'BP',  '#1ABC9C'),
    ('G',   'G',   '#2ECC71'),
    ('RP',  'RP',  '#F39C12'),
    ('J',   'J',   '#E67E22'),
    ('H',   'H',   '#E74C3C'),
    ('K',   'Ks',  '#C0392B'),
    ('W1',  'W1',  '#8E44AD'),
    ('W2',  'W2',  '#2980B9'),
    ('W3',  'W3',  '#16A085'),
    ('W4',  'W4',  '#27AE60'),
]

# ---------------------------------------------------------------------------
# Load catalogue
# ---------------------------------------------------------------------------
print(f'Reading {PASTA} ...')
with fits.open(PASTA, memmap=True) as hdul:
    t     = hdul[1].data
    n_all = len(t)
    mags = {name: np.array(t[col], dtype=np.float32)
            for name, col, _ in BANDS}
    errs = {name: np.array(t['e_' + col], dtype=np.float32)
            for name, col, _ in BANDS}

print(f'  {n_all:,} sources loaded.')


def measured(name):
    """True where the band is a real measurement rather than a censored
    upper limit: AllWISE nulls the uncertainty for undetected W3/W4."""
    return np.isfinite(mags[name]) & np.isfinite(errs[name])


g = mags['G']
ok_g = measured('G')

# ---------------------------------------------------------------------------
# Figure 1: two-panel colour-colour diagrams (AB)
# ---------------------------------------------------------------------------
print('Building colour-colour diagrams ...')

ok_a = measured('NUV') & ok_g & measured('BP') & measured('RP')
x_a  = (mags['BP'] - mags['RP'])[ok_a]
y_a  = (mags['NUV'] - g)[ok_a]
print(f'  (NUV-G) vs (BP-RP): {ok_a.sum():,} sources')
cnt_a, xe_a, ye_a = np.histogram2d(x_a, y_a, bins=400,
                                   range=[[-1.0, 3.2], [-3.0, 12.0]])

ok_b = measured('BP') & measured('RP') & measured('J') & measured('W1')
x_b  = (mags['BP'] - mags['RP'])[ok_b]
y_b  = (mags['J'] - mags['W1'])[ok_b]
print(f'  (BP-RP) vs (J-W1):  {ok_b.sum():,} sources')
cnt_b, xe_b, ye_b = np.histogram2d(x_b, y_b, bins=400,
                                   range=[[-1.0, 3.2], [-3.0, 1.5]])

fig_cc, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

for ax, cnt, xe, ye, xlabel, ylabel in [
    (ax_a, cnt_a, xe_a, ye_a,
     r'$(G_{\rm BP} - G_{\rm RP})_{\rm AB}$  [mag]',
     r'$(\mathrm{NUV} - G)_{\rm AB}$  [mag]'),
    (ax_b, cnt_b, xe_b, ye_b,
     r'$(G_{\rm BP} - G_{\rm RP})_{\rm AB}$  [mag]',
     r'$(J - W1)_{\rm AB}$  [mag]'),
]:
    im = ax.pcolormesh(xe, ye, cnt.T,
                       norm=mcolors.LogNorm(vmin=1, vmax=cnt.max()),
                       cmap='magma_r', rasterized=True)
    fig_cc.colorbar(im, ax=ax, label='Sources per bin', pad=0.02, fraction=0.046)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color='grey', lw=0.3, alpha=0.4)

fig_cc.tight_layout()
for ext in ('pdf', 'png'):
    path = os.path.join(OUTDIR, f'cc_diagrams.{ext}')
    fig_cc.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved {path}')

# ---------------------------------------------------------------------------
# Figure 2: per-band magnitude histograms (3 x 4 grid)
# ---------------------------------------------------------------------------
print('Building magnitude histograms ...')

fig_hist, axes_hist = plt.subplots(3, 4, figsize=(14, 9))
axes_hist = axes_hist.flatten()

for i, (name, col, color) in enumerate(BANDS):
    vals = mags[name][measured(name)]
    ax   = axes_hist[i]
    lo, hi = np.percentile(vals, [0.1, 99.9])
    ax.hist(vals, bins=80, range=(lo, hi),
            color=color, alpha=0.85, histtype='stepfilled', density=True)
    ax.set_title(name, fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel('AB mag', fontsize=9)
    ax.set_ylabel('density', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.97, 0.96, f'$N={len(vals):,}$',
            transform=ax.transAxes, fontsize=8, ha='right', va='top')
    ax.grid(True, lw=0.3, alpha=0.4)

fig_hist.suptitle('Per-band magnitude distributions in PaStA (AB, measurements only)',
                  fontsize=13)
fig_hist.tight_layout()

for ext in ('pdf', 'png'):
    path = os.path.join(OUTDIR, f'mag_histograms.{ext}')
    fig_hist.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved {path}')

# ---------------------------------------------------------------------------
# Measured-fraction table, printed as LaTeX rows
# ---------------------------------------------------------------------------
print('\n--- Table body (measured fractions), paste into pasta_paper1.tex ---')

G_EDGES  = [0, 10, 12, 14, 16, 18, 20, 99]
G_LABELS = ['$<10$', '$10$--$12$', '$12$--$14$',
            '$14$--$16$', '$16$--$18$', '$18$--$20$', '$>20$']

names = [b[0] for b in BANDS]
msk_b = {n: measured(n) for n in names}

rows = list(zip(G_EDGES[:-1], G_EDGES[1:], G_LABELS)) + [(None, None, 'Total')]
for lo, hi, label in rows:
    mask = ok_g if lo is None else (ok_g & (g >= lo) & (g < hi))
    n_g  = int(mask.sum())
    cells = ''.join(f' & {100.0*int((mask & msk_b[n]).sum())/n_g:.1f}' if n_g else ' & --'
                    for n in names)
    n_fmt = f'{n_g:,}'.replace(',', '\\,')
    sep = '\\midrule\n' if label == 'Total' else ''
    print(f'{sep}    {label:<12} & {n_fmt:>12}{cells} \\\\')
print('--- end ---')
print('\nDone.')
