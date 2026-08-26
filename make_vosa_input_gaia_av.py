#!/usr/bin/env python3
"""
make_vosa_input_gaia_av.py
Re-prepare the VOSA input file using Gaia DR3 GSP-Phot A_G as a fixed
interstellar extinction for each source, instead of a free Av range.

For the 7 sources without a Gaia GSP-Phot solution, a narrow fitting
range Av:0/2 is kept in column 10.

Chamaeleon star-forming region members (RA 163-168°, Dec -75 to -78°)
are identified and flagged; they are excluded from the statistical
comparison because their SEDs may include disc/accretion emission.

Conversion used: Av = A_G / 0.836   (Fitzpatrick 1999 law, Rv=3.1,
  coefficient from Wang & Chen 2019 for Teff ~5000 K).

Output: vosa_input_gaia_av.txt   (upload to VOSA in place of vosa_input.txt)

Run from the paper root directory:
    python make_vosa_input_gaia_av.py
"""

import re
import numpy as np
import pandas as pd
from astropy.table import Table
from astroquery.gaia import Gaia

MATCHED_FILE  = 'ges_pasta_matched.fits'
OUT_FILE      = 'vosa_input_gaia_av.txt'
W1_SAT_LIMIT  = 8.0
AG_TO_AV      = 1.0 / 0.836   # A_G → A_V conversion factor

# Chamaeleon SFR: RA 163–168°, Dec −78 to −74°
CHA_RA_RANGE  = (163.0, 169.0)
CHA_DEC_RANGE = (-78.0, -74.0)

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
NOFIT_BANDS = {'W1', 'W2'}


def clean_name(s):
    s = str(s).strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_]', '', s)
    return s or 'unknown'


def fmt_mag(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '---'
    if not np.isfinite(v) or v <= 0:
        return '---'
    return f'{v:.4f}'


# ---------------------------------------------------------------------------
# Load matched table, apply W1 < 8 filter
# ---------------------------------------------------------------------------
print(f'Reading {MATCHED_FILE} ...')
t   = Table.read(MATCHED_FILE)
w1  = np.array(t['pasta_W1'], dtype=np.float32)
sel = np.isfinite(w1) & (w1 < W1_SAT_LIMIT)
sub = t[sel]
print(f'Sources with W1 < {W1_SAT_LIMIT} mag: {len(sub)}')

sids = np.array(sub['pasta_source_id'], dtype=np.int64)

# ---------------------------------------------------------------------------
# Query Gaia DR3 astrophysical_parameters for A_G
# ---------------------------------------------------------------------------
print('Querying Gaia DR3 for A_G (GSP-Phot) ...')
sid_list = '(' + ','.join(str(s) for s in sids) + ')'
query = f'''
SELECT source_id, ag_gspphot, ag_gspphot_lower, ag_gspphot_upper,
       ebpminrp_gspphot
FROM gaiadr3.astrophysical_parameters
WHERE source_id IN {sid_list}
'''
job    = Gaia.launch_job(query)
gaia_r = job.get_results()
print(f'  {len(gaia_r)} rows returned from Gaia.')

# Build source_id → A_G lookup
ag_lookup = {}
for row in gaia_r:
    sid = int(row['source_id'])
    ag  = float(row['ag_gspphot']) if row['ag_gspphot'] is not None and \
          np.isfinite(float(row['ag_gspphot'])) else np.nan
    ag_lookup[sid] = ag

# ---------------------------------------------------------------------------
# Identify Chamaeleon members by sky position
# ---------------------------------------------------------------------------
def is_cha(ra, dec):
    return (CHA_RA_RANGE[0] <= ra <= CHA_RA_RANGE[1] and
            CHA_DEC_RANGE[0] <= dec <= CHA_DEC_RANGE[1])

# ---------------------------------------------------------------------------
# Build VOSA file
# ---------------------------------------------------------------------------
header_lines = [
    '# VOSA input — GES x PaStA bright sources (W1 < 8 mag)',
    '# Extinction: fixed Av from Gaia DR3 GSP-Phot A_G (Av = A_G / 0.836)',
    '# Sources without Gaia Av use a fitting range Av:0/2 (column 10)',
    '# W1, W2 marked nofit (AllWISE saturation)',
    '# Chamaeleon YSO candidates marked in comments',
    '# Columns: name  RA  Dec  dist_pc  Av  filter  mag  e_mag  pt_opt  obj_opt',
    '#',
]

data_lines = []
summary    = []

for row in sub:
    sid  = int(row['pasta_source_id'])
    ra   = float(row['pasta_ra'])
    dec  = float(row['pasta_dec'])
    name = clean_name(row['ges_object'])
    dist = float(row['pasta_dist_pc'])
    dist_s = f'{dist:.1f}' if np.isfinite(dist) and dist > 0 else '---'

    ag_val = ag_lookup.get(sid, np.nan)
    cha    = is_cha(ra, dec)

    if np.isfinite(ag_val):
        av_val  = ag_val * AG_TO_AV
        av_col5 = f'{av_val:.3f}'   # fixed Av in column 5
        obj_opt = '---'
    else:
        av_col5 = '---'
        obj_opt = 'Av:0/2'          # narrow fitting range if no Gaia solution

    if cha:
        data_lines.append(f'# --- Chamaeleon YSO candidate: {name}')

    for band, filt in FILTERS:
        mag = fmt_mag(row[f'pasta_{band}'])
        err = fmt_mag(row[f'pasta_e_{band}'])

        if band in NOFIT_BANDS:
            pt_opt = 'nofit'
        elif mag == '---':
            pt_opt = '---'
        else:
            pt_opt = 'mag'

        data_lines.append(
            f'{name:<30s}  {ra:.6f}  {dec:.6f}  {dist_s}  {av_col5}  '
            f'{filt:<22s}  {mag}  {err}  {pt_opt}  {obj_opt}'
        )

    summary.append({
        'name':    str(row['ges_object']).strip(),
        'G':       float(row['pasta_G']),
        'W1':      float(row['pasta_W1']),
        'ag_gaia': ag_val,
        'av_gaia': ag_val * AG_TO_AV if np.isfinite(ag_val) else np.nan,
        'cha':     cha,
        'ges_teff':float(row['ges_teff']),
    })

with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(header_lines + [''] + data_lines) + '\n')
print(f'\nWritten {OUT_FILE}  ({len(sub)} objects × {len(FILTERS)} bands)')

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
df = pd.DataFrame(summary)
n_gaia_av = df['ag_gaia'].notna().sum()
n_cha     = df['cha'].sum()
print(f'\nSources with Gaia A_G:    {n_gaia_av}/{len(df)}')
print(f'Chamaeleon candidates:    {n_cha}')
print()
print(f'{"GES object":<30s}  {"G":>5}  {"A_G":>5}  {"Av":>5}  {"Cha":>4}  {"Teff_GES":>9}')
print('-' * 68)
for _, r in df.iterrows():
    ag_s  = f'{r["ag_gaia"]:.3f}' if np.isfinite(r['ag_gaia']) else '  ---'
    av_s  = f'{r["av_gaia"]:.3f}' if np.isfinite(r['av_gaia']) else '  ---'
    t_s   = f'{r["ges_teff"]:.0f}' if np.isfinite(r['ges_teff']) else '   ---'
    cha_s = ' yes' if r['cha'] else '  no'
    print(f'{r["name"]:<30s}  {r["G"]:>5.2f}  {ag_s:>5}  {av_s:>5}  {cha_s}  {t_s:>9}')

print('\nDone.')
