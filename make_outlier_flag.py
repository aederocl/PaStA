#!/usr/bin/env python3
"""
make_outlier_flag.py
Compute the outlier_flag column for PaStA Paper I (Sect. 6).

For each source assigned to a CMD region (Sect. 5), compute the normalised
residual against the region's median lambda*F_lambda SED and encode deviations
as a 9-bit integer flag.

Bit encoding
------------
  Bit 0  (  1): FUV excess   (FUV residual > +N_SIGMA)
  Bit 1  (  2): NUV excess
  Bit 2  (  4): FUV deficit  (FUV residual < -N_SIGMA)
  Bit 3  (  8): NUV deficit
  Bit 4  ( 16): W1/W2 near-IR excess  (BOTH W1 AND W2 > +N_SIGMA)
  Bit 5  ( 32): W3/W4 mid-IR excess   (EITHER W3 OR W4 > +N_SIGMA)
  Bit 6  ( 64): W1/W2 near-IR deficit (BOTH W1 AND W2 < -N_SIGMA)
  Bit 7  (128): W3/W4 mid-IR deficit  (EITHER W3 OR W4 < -N_SIGMA)
  Bit 8  (256): shape outlier (>= K_SHAPE bands deviant in any direction)

The scatter reference is sigma = (p84 - p16) / 2 per band per region
(half the 68-per-cent interval; equals 1 sigma for a Gaussian).
Sources outside all CMD regions receive flag = 0.
A bit is only ever set from a band the source actually has.

A bit additionally requires the deviation to exceed SIG_SOURCE times the
source's OWN photometric uncertainty, so that a flag is significant against
the measurement as well as against the population. Region medians are tight
enough that in the mid-infrared, where most detections are marginal, the
population threshold alone was being reached by ordinary noise.

Photometric system
------------------
Input is pasta1_public.fits: one epoch (J2016.0), all twelve bands AB, and
already deduplicated (9,705,879 rows). The single zero point 3631 Jy therefore
applies throughout and cancels in the G-normalised ratio. The earlier version
of this script read the native-system pasta.fits with the pre-AB Vega CMD
boxes, so both the region membership and the SED normalisation were wrong in
the same way as the first version of make_sed_regions.py.

Note that the normalised residual (x - median)/sigma is exactly invariant
under a constant per-band rescaling of the fluxes, so the flag is insensitive
to the choice of zero points; what it is NOT insensitive to is which sources
fall in which CMD box, and which values are real measurements.

Censoring
---------
AllWISE tabulates a magnitude for undetected sources but nulls the
uncertainty, in which case the magnitude is a 95-per-cent upper limit rather
than a measurement (94.7 per cent of W4 and 59.8 per cent of W3 in PaStA). The
previous version tested only isfinite(mag), so those limits entered the SEDs
as faint detections clustered at the AllWISE sensitivity floor; divided by a
varying G flux they manufactured an apparent mid-infrared excess, and bit 5
was largely a measurement of that floor rather than of infrared excess.

A band therefore contributes only if it has a finite magnitude AND a finite
uncertainty -- the same criterion as make_sed_regions.py, with no
signal-to-noise cut on top (GALEX and Gaia never null an uncertainty, so an
SNR cut there would preferentially discard faint sources and bias the medians).

One asymmetry follows from this and is handled explicitly. Where a band is
detected in fewer than DETFRAC_MIN of a region's sources, the median of the
detections is an upper bound on the median of the population -- exactly the
case Sect. 5 draws as an upper limit. A source detected in that band is still
compared against a reference that is biased bright, which makes an EXCESS
harder to reach (conservative, so it is kept) but makes a DEFICIT too easy to
reach (so it is suppressed for that band in that region).

Outputs
-------
  outlier_flag.npz                                 source_id + flag (uint16)
  fig/outlier_examples.{pdf,png}

Run from the paper root directory:
    python3 make_outlier_flag.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ---------------------------------------------------------------------------
# Band definitions: (label, FITS column, lambda_pivot [micron])
# Identical to make_sed_regions.py: AB throughout, SVO pivot wavelengths,
# one revision per instrument (GAIA3 for Gaia).
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
BAND_LAMBDA = np.array([b[2] for b in BANDS])
NB          = len(BANDS)
G_IDX   = BAND_NAMES.index('G')
BP_IDX  = BAND_NAMES.index('BP')
RP_IDX  = BAND_NAMES.index('RP')
FUV_IDX = BAND_NAMES.index('FUV')
NUV_IDX = BAND_NAMES.index('NUV')
W1_IDX  = BAND_NAMES.index('W1')
W2_IDX  = BAND_NAMES.index('W2')
W3_IDX  = BAND_NAMES.index('W3')
W4_IDX  = BAND_NAMES.index('W4')

# ---------------------------------------------------------------------------
# CMD region definitions, AB system. Identical to make_sed_regions.py.
# The boxes do not overlap, so first-match assignment here and the independent
# box tests there select the same sources; this is asserted below.
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
NR = len(REGIONS)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SIGMA     = 3.0    # detection threshold in units of region scatter
K_SHAPE     = 4      # minimum number of deviant bands for the shape-outlier flag
DETFRAC_MIN = 0.50   # below this the region median is an upper limit
SIG_SOURCE  = 3.0    # deviation must also exceed this x the source's own error
PASTA       = 'pasta1_public.fits'
STATS_NPZ   = 'sed_region_stats_ab.npz'   # written by make_sed_regions.py
OUT_NPZ     = 'outlier_flag.npz'

# ---------------------------------------------------------------------------
# Load catalogue
# ---------------------------------------------------------------------------
print(f'Reading {PASTA} ...')
with fits.open(PASTA, memmap=True) as hdul:
    t         = hdul[1].data
    n_all     = len(t)
    source_id = np.array(t['source_id'], dtype=np.int64)
    dist      = np.array(t['distance'], dtype=np.float32)
    mags = np.column_stack([
        np.array(t[col], dtype=np.float32) for col in BAND_COLS
    ])   # (n_all, 12)
    errs = np.column_stack([
        np.array(t['e_' + col], dtype=np.float32) for col in BAND_COLS
    ])   # (n_all, 12)

print(f'  {n_all:,} sources loaded.')

# ---------------------------------------------------------------------------
# CMD position
# ---------------------------------------------------------------------------
g_, bp_, rp_ = mags[:, G_IDX], mags[:, BP_IDX], mags[:, RP_IDX]
ok = (np.isfinite(dist) & (dist > 0) &
      np.isfinite(g_) & np.isfinite(bp_) & np.isfinite(rp_))
ok_idx  = np.where(ok)[0]
N_ok    = int(ok.sum())
mg_ok   = g_[ok] - 5.0 * np.log10(dist[ok]) + 5.0
bprp_ok = bp_[ok] - rp_[ok]
print(f'  {N_ok:,} sources with a valid CMD position.')

# ---------------------------------------------------------------------------
# Assign each source to exactly one CMD region (first match wins)
# ---------------------------------------------------------------------------
print('Assigning sources to CMD regions ...')
region_id = np.full(N_ok, -1, dtype=np.int8)
for i, reg in enumerate(REGIONS):
    blo, bhi = reg['bprp']
    mlo, mhi = reg['mg']
    box = ((bprp_ok >= blo) & (bprp_ok < bhi) &
           (mg_ok   >= mlo) & (mg_ok   < mhi))
    mask = (region_id == -1) & box
    # The boxes are disjoint, so first-match assignment must lose nothing.
    assert int(box.sum()) == int(mask.sum()), \
        f'region {reg["label"]} overlaps an earlier region'
    region_id[mask] = i
    print(f'  {reg["label"]:4s}: {mask.sum():>10,} sources')

in_reg = region_id >= 0
n_reg  = int(in_reg.sum())
print(f'  in a region: {n_reg:,}   (outside all boxes: {N_ok - n_reg:,})')

# ---------------------------------------------------------------------------
# Restrict all further work to sources inside a box: everything else keeps
# flag = 0 by construction, and this keeps the residual arrays small.
# ---------------------------------------------------------------------------
sel      = np.where(in_reg)[0]
reg_sel  = region_id[sel]
mags_sel = mags[ok_idx[sel]]
errs_sel = errs[ok_idx[sel]]

# ---------------------------------------------------------------------------
# Detection criterion: finite magnitude AND finite uncertainty (see docstring).
# ---------------------------------------------------------------------------
detected = np.isfinite(mags_sel) & np.isfinite(errs_sel)

# ---------------------------------------------------------------------------
# Magnitudes -> F_nu -> lambda*F_lambda normalised to each star's own G band.
#   (lambda F_lambda)_norm = (F_nu_band / F_nu_G) * (lambda_G / lambda_band)
# The 3631 Jy cancels in the ratio.
# ---------------------------------------------------------------------------
print('Converting to normalised lambda*F_lambda ...')
fnu = np.where(detected, 10.0 ** (-0.4 * mags_sel), np.nan).astype(np.float32)
fnu_G = fnu[:, G_IDX, np.newaxis]
with np.errstate(invalid='ignore', divide='ignore'):
    lflf = ((fnu / fnu_G) *
            (BAND_LAMBDA[G_IDX] / BAND_LAMBDA[np.newaxis, :])).astype(np.float32)
lflf[~np.isfinite(fnu_G[:, 0]), :] = np.nan

# ---------------------------------------------------------------------------
# Per-region reference SED: median, sigma = (p84 - p16)/2, and the fraction of
# the region actually measured in each band.
# ---------------------------------------------------------------------------
print('Computing per-region reference SEDs ...')
med     = np.full((NR, NB), np.nan)
sigma   = np.full((NR, NB), np.nan)
p16     = np.full((NR, NB), np.nan)
p84     = np.full((NR, NB), np.nan)
detfrac = np.zeros((NR, NB))
for i in range(NR):
    data  = lflf[reg_sel == i]
    n_tot = len(data)
    for j in range(NB):
        finite = data[:, j][np.isfinite(data[:, j])]
        detfrac[i, j] = len(finite) / n_tot if n_tot else 0.0
        if len(finite) >= 10:
            med[i, j]   = np.median(finite)
            p16[i, j]   = np.percentile(finite, 16)
            p84[i, j]   = np.percentile(finite, 84)
            sigma[i, j] = max((p84[i, j] - p16[i, j]) / 2.0, 1e-6)

# Cross-check against the reference SEDs Sect. 5 published, so that Sect. 6
# provably measures residuals against the medians drawn in Fig. 5.
if os.path.exists(STATS_NPZ):
    ref = np.load(STATS_NPZ, allow_pickle=True)
    assert list(ref['labels']) == [r['label'] for r in REGIONS], \
        f'{STATS_NPZ} has different regions'
    both = np.isfinite(med) & np.isfinite(ref['med'])
    rel  = np.abs(med[both] - ref['med'][both]) / np.abs(ref['med'][both])
    print(f'  cross-check vs {STATS_NPZ}: max relative difference in the '
          f'medians {rel.max():.2e}')
    assert rel.max() < 1e-5, f'medians disagree with {STATS_NPZ}'
else:
    print(f'  ({STATS_NPZ} absent -- skipping the Sect. 5 cross-check)')

# Bands whose region median is really an upper limit: a deficit measured
# against a reference biased bright is not meaningful, so it is suppressed.
is_limit = detfrac < DETFRAC_MIN
for i, reg in enumerate(REGIONS):
    lim = [BAND_NAMES[j] for j in range(NB) if is_limit[i, j]]
    print(f'  {reg["label"]:4s}: deficit bits suppressed in '
          f'{", ".join(lim) if lim else "no band"}')

# ---------------------------------------------------------------------------
# Normalised residuals r = (lambda F_lambda - median) / sigma.
# NaN where the band is missing or censored; NaN comparisons are False, so a
# band a source does not have can never raise a flag.
# ---------------------------------------------------------------------------
print('Computing normalised residuals ...')
with np.errstate(invalid='ignore', divide='ignore'):
    res = ((lflf - med[reg_sel]) / sigma[reg_sel]).astype(np.float32)

# Per-source mask of bands whose region reference is an upper limit
lim_sel = is_limit[reg_sel]

# Significance of the deviation against the SOURCE's own photometric error.
#
# A residual is measured against the region scatter, which for a homogeneous
# population is tight: in the red clump a 3 sigma deviation in W3 is a colour
# offset of only 0.55 mag. A source whose W3 is a marginal detection carries
# ~0.4 mag of photometric error, so ordinary noise reaches that threshold
# unaided. Requiring the deviation to be significant against the measurement
# as well as against the population removes exactly those cases: before this
# condition, 89.5 per cent of the W3 excesses and 95.1 per cent of the W4
# excesses had SNR < 5, against 0.3 and 0.2 per cent for W1 and W2.
#
# This is NOT the population-level SNR cut rejected in make_sed_regions.py,
# and the two do not conflict. There the quantity was a MEDIAN over a region,
# which an SNR cut biases by discarding the faint members. Here it is a
# PER-SOURCE deviation, where the measurement error is precisely what decides
# whether a deviation is measurable at all. A flat SNR cut would also be the
# wrong instrument: it would discard a fifth of the NUV excesses, which are
# genuine because the NUV region scatter is more than a magnitude wide and
# noise of a few tenths cannot manufacture a deviation that large.
#
# The G-normalised ratio carries the errors of both bands:
#   d(ratio)/ratio = 0.4 ln(10) sqrt(sigma_band^2 + sigma_G^2)
with np.errstate(invalid='ignore', divide='ignore'):
    dev_frac = 0.4 * np.log(10.0) * np.sqrt(
        errs_sel ** 2 + errs_sel[:, G_IDX, np.newaxis] ** 2)
    sig_src = np.abs(lflf - med[reg_sel]) / (np.abs(lflf) * dev_frac)

with np.errstate(invalid='ignore'):
    significant = sig_src >= SIG_SOURCE
    hi = (res >  N_SIGMA) & significant
    lo = (res < -N_SIGMA) & significant & ~lim_sel

# What the source-significance condition removes, per band and direction.
with np.errstate(invalid='ignore'):
    print('  effect of the source-significance condition:')
    for name, j in (('FUV', FUV_IDX), ('NUV', NUV_IDX), ('W1', W1_IDX),
                    ('W2', W2_IDX), ('W3', W3_IDX), ('W4', W4_IDX)):
        raw_hi = res[:, j] > N_SIGMA
        raw_lo = (res[:, j] < -N_SIGMA) & ~lim_sel[:, j]
        for lbl, raw, kept in (('excess', raw_hi, hi[:, j]),
                               ('deficit', raw_lo, lo[:, j])):
            n = int(raw.sum())
            if not n:
                continue
            k = int(kept.sum())
            print(f'    {name:>3s} {lbl:<8s} {n:>8,} -> {k:>8,}  '
                  f'({100 * k / n:5.1f} per cent kept)')

# ---------------------------------------------------------------------------
# Set flag bits
# ---------------------------------------------------------------------------
print('Setting flag bits ...')
flags_sel = np.zeros(len(sel), dtype=np.uint16)

def _set(cond, bit):
    flags_sel[cond] |= np.uint16(bit)

_set(hi[:, FUV_IDX], 1)     # Bit 0: FUV excess
_set(hi[:, NUV_IDX], 2)     # Bit 1: NUV excess
_set(lo[:, FUV_IDX], 4)     # Bit 2: FUV deficit
_set(lo[:, NUV_IDX], 8)     # Bit 3: NUV deficit

# Near-IR: both W1 AND W2 must deviate (suppresses single-band noise)
_set(hi[:, W1_IDX] & hi[:, W2_IDX],  16)   # Bit 4
_set(lo[:, W1_IDX] & lo[:, W2_IDX],  64)   # Bit 6

# Mid-IR: either W3 OR W4 suffices (W4 is a real detection for few sources)
_set(hi[:, W3_IDX] | hi[:, W4_IDX],  32)   # Bit 5
_set(lo[:, W3_IDX] | lo[:, W4_IDX], 128)   # Bit 7

# Shape outlier: number of bands deviant in either direction
n_deviant = np.sum(hi | lo, axis=1)
_set(n_deviant >= K_SHAPE, 256)            # Bit 8

# ---------------------------------------------------------------------------
# Write the flag for all n_all sources, keyed on source_id
# ---------------------------------------------------------------------------
flags_all = np.zeros(n_all, dtype=np.uint16)
flags_all[ok_idx[sel]] = flags_sel
np.savez(OUT_NPZ, source_id=source_id, flag=flags_all)
print(f'\nSaved {OUT_NPZ}  ({n_all:,} sources, uint16)')

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
BIT_TABLE = [
    (  1, 'FUV excess'),
    (  2, 'NUV excess'),
    (  4, 'FUV deficit'),
    (  8, 'NUV deficit'),
    ( 16, 'W1/W2 excess'),
    ( 32, 'W3/W4 excess'),
    ( 64, 'W1/W2 deficit'),
    (128, 'W3/W4 deficit'),
    (256, 'Shape outlier'),
]

print('\n--- Global summary (sources in a CMD region) ---')
print(f'  In a CMD region     : {n_reg:>10,}')
print(f'  Any flag set        : {(flags_sel > 0).sum():>10,}  '
      f'({100 * (flags_sel > 0).mean():.2f} per cent)')
print()
print(f'  {"Bit":>4}  {"Label":<20}  {"N":>10}  {"per cent":>8}')
print('  ' + '-' * 48)
for val, label in BIT_TABLE:
    n = int((flags_sel & np.uint16(val)).astype(bool).sum())
    print(f'  {val:>4}  {label:<20}  {n:>10,}  {100 * n / n_reg:>8.2f}')

print('\n--- Per-region breakdown ---')
header = f'{"Region":<6}  {"N_in":>10}  {"any":>10}'
for val, label in BIT_TABLE:
    header += f'  {label[:10]:>10}'
print(header)
for i, reg in enumerate(REGIONS):
    m    = (reg_sel == i)
    n_in = int(m.sum())
    row  = f'{reg["label"]:<6}  {n_in:>10,}  {int((flags_sel[m] > 0).sum()):>10,}'
    for val, label in BIT_TABLE:
        row += f'  {int((flags_sel[m] & np.uint16(val)).astype(bool).sum()):>10,}'
    print(row)

print('\n--- measured (uncensored) fraction per region and band (per cent) ---')
print(f'{"region":6s}' + ''.join(f'{b:>7s}' for b in BAND_NAMES))
for i, reg in enumerate(REGIONS):
    print(f'{reg["label"]:6s}' + ''.join(f'{100 * f:7.1f}' for f in detfrac[i]))
print('--- end ---')

# ---------------------------------------------------------------------------
# Example SED figure
# ---------------------------------------------------------------------------
print('\nBuilding example SED figure ...')
outdir = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
os.makedirs(outdir, exist_ok=True)

# Each example is drawn from the region named in the caption, so that the two
# panels illustrate different comparison populations, and at a high but not
# extreme percentile of the residual: the single largest residual is a
# pathological object in both cases and would suggest that the flag fires only
# on extremes, whilst the 90th percentile is barely visible by eye.
EXAMPLE_PCT = 0.99
CLEAN_SIGMA = 1.0   # "consistent with the median" for the bands held normal
J_IDX = BAND_NAMES.index('J')
H_IDX = BAND_NAMES.index('H')
K_IDX = BAND_NAMES.index('K')

# 'clean' lists bands that must sit within CLEAN_SIGMA of the region median,
# and 'exclude' bits that must NOT be set. Without them the highest-ranked
# W1/W2-excess sources are objects deviant in every band at once, which
# illustrates "extreme object" rather than "infrared excess": the point of the
# panel is a source that follows its region through the optical and the near
# infrared and then departs at 3-5 micron.
EXAMPLES = [
    {'title': 'NUV excess',      'bit':  2, 'rank_band': NUV_IDX, 'sign': +1,
     'region': 'K'},
    {'title': 'W1/W2 IR excess', 'bit': 16, 'rank_band': W1_IDX,  'sign': +1,
     'region': None, 'clean': (J_IDX, H_IDX, K_IDX), 'exclude': 1 | 2},
]

# The figure is printed at \columnwidth (88 mm) in a two-column A&A layout, so
# it is drawn at very nearly that physical size and the panels are stacked
# rather than placed side by side. A 12-inch-wide two-panel figure reduced to
# one column shrinks every label by a factor of three; at this size the font
# sizes set below are the sizes that reach the page.
fig_ex, axes_ex = plt.subplots(2, 1, figsize=(3.6, 5.6), sharex=True)

for ax, ex in zip(axes_ex, EXAMPLES):
    bit_mask = (flags_sel & np.uint16(ex['bit'])) > 0
    if ex.get('region') is not None:
        bit_mask &= (reg_sel == [r['label'] for r in REGIONS].index(ex['region']))
    if ex.get('exclude'):
        bit_mask &= (flags_sel & np.uint16(ex['exclude'])) == 0
    with np.errstate(invalid='ignore'):
        for j in ex.get('clean', ()):
            bit_mask &= np.isfinite(res[:, j]) & (np.abs(res[:, j]) < CLEAN_SIGMA)
    if not bit_mask.any():
        ax.text(0.5, 0.5, 'No flagged sources', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(ex['title'])
        continue

    # A representative rather than a record-breaking case: the source at
    # EXAMPLE_PCT of the (signed) residual among those carrying the bit.
    idx_bit = np.where(bit_mask)[0]
    with np.errstate(invalid='ignore'):
        score = res[idx_bit, ex['rank_band']] * ex['sign']
    rank = np.argsort(score)
    src  = int(idx_bit[rank[int(EXAMPLE_PCT * (len(rank) - 1))]])
    reg_i = int(reg_sel[src])
    print(f'  {ex["title"]:16s} candidates N={len(rank):>7,}  '
          f'chosen from region {REGIONS[reg_i]["label"]}  '
          f'residual percentiles '
          + '  '.join(f'{p:g}%:{np.percentile(score, p):.0f}'
                      for p in (50, 90, 99, 100)))
    if ex.get('clean'):
        print('    per-region candidates: ' + '  '.join(
            f'{REGIONS[i]["label"]}:{int((bit_mask & (reg_sel == i)).sum()):,}'
            for i in range(NR) if (bit_mask & (reg_sel == i)).any()))
    color = REGIONS[reg_i]['color']

    m_ref = med[reg_i]
    lo_ref, hi_ref = p16[reg_i], p84[reg_i]
    src_v = lflf[src]

    src_ok  = np.isfinite(src_v)
    med_ok  = np.isfinite(m_ref) & ~is_limit[reg_i]
    lim_ok  = np.isfinite(m_ref) & is_limit[reg_i]

    lam = BAND_LAMBDA

    # 16-84 percentile interval rather than median +/- sigma: the two are the
    # same 68-per-cent interval by construction (sigma = (p84 - p16)/2), but
    # the percentiles stay positive on a logarithmic axis for the strongly
    # skewed ultraviolet distributions.
    lbl = REGIONS[reg_i]['tex']
    if med_ok.any():
        ax.fill_between(lam[med_ok], lo_ref[med_ok], hi_ref[med_ok],
                        color=color, alpha=0.20, label='16--84th pct')
        ax.plot(lam[med_ok], m_ref[med_ok], 'o-', color=color,
                lw=1.3, ms=4, label=f'{lbl} median')

    # Bands in which the region itself is censored: shown as upper limits, as
    # in Fig. 5, so that the reference does not simply stop at W2.
    if lim_ok.any():
        ax.errorbar(lam[lim_ok], m_ref[lim_ok],
                    yerr=[m_ref[lim_ok] * 0.55, np.zeros(int(lim_ok.sum()))],
                    uplims=True, fmt='_', color=color, ms=5, lw=0.9,
                    elinewidth=0.9, alpha=0.75, label='median (limit)')

    if src_ok.any():
        ax.plot(lam[src_ok], src_v[src_ok], 's--', color='k',
                lw=1.3, ms=4.5, label='Example source')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.10, 30)
    ax.set_ylabel(r'$(\lambda F_\lambda)/(\lambda F_\lambda)_G$', fontsize=9)
    ax.set_title(f'{ex["title"]} ({lbl} region)', fontsize=9)
    ax.axhline(1.0, color='grey', lw=0.6, ls='--', alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9,
              handlelength=1.6, borderpad=0.4, labelspacing=0.3)
    ax.grid(True, which='both', lw=0.3, alpha=0.4)

axes_ex[-1].set_xlabel(r'$\lambda$  [$\mu$m]', fontsize=9)
fig_ex.tight_layout(h_pad=0.8)

for ext in ('pdf', 'png'):
    path = os.path.join(outdir, f'outlier_examples.{ext}')
    fig_ex.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')

print('Done.')
