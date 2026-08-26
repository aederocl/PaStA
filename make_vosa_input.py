#!/usr/bin/env python3
"""
make_vosa_input.py
Prepare a VOSA upload file for the bright GES-PaStA sources (W1 < 8 mag).

VOSA format: one line per photometric data point (band), 10 whitespace-
separated columns:
  name  RA  Dec  dist_pc  Av  filter  flux  error  point_opt  obj_opt

Magnitudes are passed in their native photometric system; the point option
'mag' tells VOSA to interpret columns 7-8 as magnitudes and to look up the
appropriate zero-point via the SVO Filter Profile Service filter name.

W1 and W2 are flagged 'nofit' for all sources (W1 < 8 mag selection means
every source in this file is near or past AllWISE's saturation limit); they
will appear in SED plots but will be excluded from the chi-squared fit.

Missing measurements are written as '---' in both flux and error columns.

Output: vosa_input.txt   (upload at https://svo2.cab.inta-csic.es/theory/vosa/)

Run from the paper root directory:
    python make_vosa_input.py
"""

import re
import numpy as np
from astropy.table import Table

MATCHED_FILE  = 'ges_pasta_matched.fits'
OUT_FILE      = 'vosa_input.txt'
W1_SAT_LIMIT  = 8.0   # mag  — AllWISE W1 saturation threshold

# ---------------------------------------------------------------------------
# Band → SVO Filter Profile Service identifier
# https://svo2.cab.inta-csic.es/theory/fps/
# ---------------------------------------------------------------------------
FILTERS = [
    ('FUV', 'GALEX/GALEX.FUV'),
    ('NUV', 'GALEX/GALEX.NUV'),
    ('BP',  'GAIA/GAIA3.Gbp'),
    ('G',   'GAIA/GAIA3.G'),
    ('RP',  'GAIA/GAIA3.Grp'),
    ('J',   '2MASS/2MASS.J'),
    ('H',   '2MASS/2MASS.H'),
    ('K',   '2MASS/2MASS.Ks'),
    ('W1',  'WISE/WISE.W1'),
    ('W2',  'WISE/WISE.W2'),
    ('W3',  'WISE/WISE.W3'),
    ('W4',  'WISE/WISE.W4'),
]

# W1 and W2 are flagged nofit for every source in this bright subset
NOFIT_BANDS = {'W1', 'W2'}

# Av fitting range passed to VOSA via column 10 (Fitzpatrick 1999 law).
# VOSA will grid-search Av within [AV_MIN, AV_MAX] simultaneously with Teff/logg/[Fe/H].
# Column 5 is left as '---' so this range drives the fit.
AV_RANGE = 'Av:0/3'


def clean_name(s):
    """Return a VOSA-safe name: alphanumeric + underscore only."""
    s = str(s).strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_]', '', s)
    return s or 'unknown'


def fmt_mag(val):
    """Return magnitude string or '---' if the value is missing/invalid."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '---'
    if not np.isfinite(v) or v <= 0:
        return '---'
    return f'{v:.4f}'


# ---------------------------------------------------------------------------
# Load and filter matched table
# ---------------------------------------------------------------------------
print(f'Reading {MATCHED_FILE} ...')
t   = Table.read(MATCHED_FILE)
w1  = np.array(t['pasta_W1'], dtype=np.float32)
sel = np.isfinite(w1) & (w1 < W1_SAT_LIMIT)
sub = t[sel]
print(f'Sources with W1 < {W1_SAT_LIMIT} mag: {len(sub)}')
print()

# ---------------------------------------------------------------------------
# Build VOSA file
# ---------------------------------------------------------------------------
header_lines = [
    '# VOSA input — GES x PaStA bright sources (W1 < 8 mag)',
    '# W1, W2 set to nofit (AllWISE saturation)',
    '# Columns: name  RA  Dec  dist_pc  Av  filter  mag  e_mag  pt_opt  obj_opt',
    '#',
]

data_lines = []

for row in sub:
    name = clean_name(row['ges_object'])
    ra   = f'{float(row["pasta_ra"]):.6f}'
    dec  = f'{float(row["pasta_dec"]):.6f}'

    dist_val = float(row['pasta_dist_pc'])
    dist = f'{dist_val:.1f}' if np.isfinite(dist_val) and dist_val > 0 else '---'

    av = '---'   # extinction not applied; VOSA will handle

    for band, filt in FILTERS:
        mag = fmt_mag(row[f'pasta_{band}'])
        err = fmt_mag(row[f'pasta_e_{band}'])

        if band in NOFIT_BANDS:
            pt_opt = 'nofit'
        elif mag == '---':
            pt_opt = '---'
        else:
            pt_opt = 'mag'

        obj_opt = AV_RANGE   # fit Av within [0, 3] mag using Fitzpatrick (1999)

        data_lines.append(
            f'{name:<30s}  {ra}  {dec}  {dist}  {av}  '
            f'{filt:<22s}  {mag}  {err}  {pt_opt}  {obj_opt}'
        )

all_lines = header_lines + [''] + data_lines

with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(all_lines) + '\n')

n_obj   = len(sub)
n_bands = len(FILTERS)
print(f'Written {OUT_FILE}')
print(f'  {n_obj} objects × {n_bands} bands = {n_obj * n_bands} data lines')

# ---------------------------------------------------------------------------
# Print a preview and a summary table
# ---------------------------------------------------------------------------
print()
print('--- Summary of bright sources ---')
print(f'{"GES object":<30s}  {"RA":>10}  {"Dec":>9}  {"dist(pc)":>9}  '
      f'{"G":>6}  {"W1":>6}  {"Teff_GES":>9}  {"logg_GES":>9}')
print('-' * 100)
for row in sub:
    g_val    = fmt_mag(row['pasta_G'])
    w1_val   = fmt_mag(row['pasta_W1'])
    teff_val = row['ges_teff']
    logg_val = row['ges_logg']
    teff_s   = f'{teff_val:.0f}' if np.isfinite(float(teff_val)) else '---'
    logg_s   = f'{logg_val:.2f}' if np.isfinite(float(logg_val)) else '---'
    print(f'{str(row["ges_object"]):<30s}  {float(row["pasta_ra"]):>10.5f}  '
          f'{float(row["pasta_dec"]):>9.5f}  {float(row["pasta_dist_pc"]):>9.1f}  '
          f'{g_val:>6}  {w1_val:>6}  {teff_s:>9}  {logg_s:>9}')

print()
print('Done.')
