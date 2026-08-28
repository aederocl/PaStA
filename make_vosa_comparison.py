#!/usr/bin/env python3
"""
make_vosa_comparison.py
Compare VOSA best-fit stellar parameters against GES DR5 spectroscopic values
for the 28 bright GES-PaStA sources (W1 < 8 mag).

Outputs
-------
  fig/vosa_comparison.{pdf,png}   — 3-panel comparison plot (Teff, logg, [Fe/H])
  vosa_ges_comparison.csv         — merged table for the paper/appendix

Run from the paper root directory:
    python make_vosa_comparison.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.table import Table

VOSA_FILE    = 'vosa_results_96498/results/bestfitp.dat'
MATCHED_FILE = 'ges_pasta_matched.fits'
W1_SAT_LIMIT = 8.0
VGFB_LIMIT   = 15.0
OUTDIR       = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))

# Cleaned object names ('+' and '-' stripped by clean_name())
CHA_NAMES = {
    '105559737724399', '105901087722407', '105906997701404',
    '110311607721042', '110205247525093', '110605117511454',
    '110946467628575', '111146327620092', '111150007620360',
    '111224417637064',
}
M67_NAMES = {
    '085128981150330', '085117111148160', '085122831148015',
}
# Sources for which no Gaia A_G was available; VOSA fitted Av freely 0-2 mag
FREE_AV_NAMES = {
    '085128981150330', '085117111148160', '085122831148015',  # M67
    '110946467628575', '111150007620360',                      # Cha
    '221343730539416', '223754130604257',                      # field
}

# ---------------------------------------------------------------------------
# Parse VOSA bestfitp.dat
# ---------------------------------------------------------------------------
print('Parsing VOSA results ...')
with open(VOSA_FILE) as f:
    lines = f.readlines()

hdr_line = lines[2].lstrip('#').strip()
cols     = re.split(r'\s{2,}', hdr_line)
sep_idx  = next(i for i, l in enumerate(lines) if l.startswith('#='))
data_lines = [l for l in lines[sep_idx+1:] if not l.startswith('#') and l.strip()]

rows = [l.split() for l in data_lines]
df_vosa = pd.DataFrame(rows, columns=cols[:len(rows[0])])

num_cols = ['Teff', 'logg', 'Meta.', 'Av', 'Chi2', 'Vgfb',
            'e_Teff', 'e_logg', 'e_Meta.',
            'Teff_min_68cl', 'Teff_max_68cl',
            'logg_min_68cl', 'logg_max_68cl',
            'Meta_min_68cl', 'Meta_max_68cl']
for c in num_cols:
    if c in df_vosa.columns:
        df_vosa[c] = pd.to_numeric(df_vosa[c], errors='coerce')

print(f'  {len(df_vosa)} VOSA fits loaded.')

# ---------------------------------------------------------------------------
# Load GES matched table and apply W1 < 8 filter
# ---------------------------------------------------------------------------
print('Loading GES matched sample ...')
t   = Table.read(MATCHED_FILE)
w1  = np.array(t['pasta_W1'], dtype=np.float32)
sub = t[np.isfinite(w1) & (w1 < W1_SAT_LIMIT)]

def clean_name(s):
    s = str(s).strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_]', '', s)
    return s or 'unknown'

df_ges = pd.DataFrame({
    'vosa_name':  [clean_name(r['ges_object']) for r in sub],
    'ges_object': [str(r['ges_object']).strip() for r in sub],
    'ges_teff':   np.array(sub['ges_teff'],   dtype=np.float64),
    'ges_e_teff': np.array(sub['ges_e_teff'], dtype=np.float64),
    'ges_logg':   np.array(sub['ges_logg'],   dtype=np.float64),
    'ges_e_logg': np.array(sub['ges_e_logg'], dtype=np.float64),
    'ges_feh':    np.array(sub['ges_feh'],    dtype=np.float64),
    'ges_e_feh':  np.array(sub['ges_e_feh'],  dtype=np.float64),
    'pasta_G':    np.array(sub['pasta_G'],    dtype=np.float32),
    'pasta_W1':   np.array(sub['pasta_W1'],   dtype=np.float32),
    'pasta_dist': np.array(sub['pasta_dist_pc'], dtype=np.float32),
})

# ---------------------------------------------------------------------------
# Merge on cleaned object name
# ---------------------------------------------------------------------------
df_vosa = df_vosa.rename(columns={
    'Object': 'vosa_name', 'Meta.': 'vosa_feh',
    'Teff':   'vosa_teff', 'logg': 'vosa_logg', 'Av': 'vosa_Av',
})

merged = pd.merge(
    df_ges,
    df_vosa[['vosa_name', 'vosa_teff', 'vosa_logg', 'vosa_feh',
             'vosa_Av', 'Chi2', 'Vgfb',
             'Teff_min_68cl', 'Teff_max_68cl',
             'logg_min_68cl', 'logg_max_68cl',
             'Meta_min_68cl', 'Meta_max_68cl']],
    on='vosa_name', how='left',
)

print(f'  Matched rows after join: {len(merged)}')

# Group flags
merged['is_cha']      = merged['vosa_name'].isin(CHA_NAMES)
merged['is_m67']      = merged['vosa_name'].isin(M67_NAMES)
merged['is_free_av']  = merged['vosa_name'].isin(FREE_AV_NAMES)
merged['is_field']    = ~(merged['is_cha'] | merged['is_m67'])
merged['good_fit']    = merged['Vgfb'] <= VGFB_LIMIT

# Convenience: field with good fit (used for statistics)
has_ges = (np.isfinite(merged['ges_teff'].values) & (merged['ges_teff'].values > 0))
merged['has_ges'] = has_ges

m_field_good = merged[merged['is_field'] & merged['good_fit'] & merged['has_ges']].copy()

print(f'  Sources with GES Teff: {has_ges.sum()}')
print(f'  Field sources with good fit (Vgfb ≤ {VGFB_LIMIT}): {len(m_field_good)}')

# ---------------------------------------------------------------------------
# Per-source table
# ---------------------------------------------------------------------------
print('\n--- Per-source comparison ---')
print(f'{"GES object":<30s}  {"Teff_GES":>8}  {"Teff_VOSA":>9}  '
      f'{"logg_GES":>8}  {"logg_VOSA":>9}  '
      f'{"FeH_GES":>7}  {"FeH_VOSA":>8}  {"Av":>5}  {"Vgfb":>6}  {"Group":>8}')
print('-' * 120)
for _, row in merged.iterrows():
    ges_t = f'{row["ges_teff"]:.0f}' if np.isfinite(float(row['ges_teff'])) else ' ---'
    ges_g = f'{row["ges_logg"]:.2f}' if np.isfinite(float(row['ges_logg'])) else ' ---'
    ges_f = f'{row["ges_feh"]:.2f}'  if np.isfinite(float(row['ges_feh']))  else ' ---'
    vgfb  = f'{row["Vgfb"]:.1f}'     if np.isfinite(float(row['Vgfb']))      else '  ---'
    grp   = 'Cha' if row['is_cha'] else ('M67' if row['is_m67'] else 'field')
    print(f'{row["ges_object"]:<30s}  {ges_t:>8}  {row["vosa_teff"]:>9.0f}  '
          f'{ges_g:>8}  {row["vosa_logg"]:>9.2f}  '
          f'{ges_f:>7}  {row["vosa_feh"]:>8.2f}  {row["vosa_Av"]:>5.2f}  '
          f'{vgfb:>6}  {grp:>8}')

# ---------------------------------------------------------------------------
# Statistics — field sources with good fits only
# ---------------------------------------------------------------------------
print(f'\n--- Statistics: field sources, good fits (Vgfb ≤ {VGFB_LIMIT}), N={len(m_field_good)} ---')
for param, ges_col, vosa_col, label in [
    ('Teff',  'ges_teff', 'vosa_teff', 'ΔTeff (VOSA − GES) [K]'),
    ('logg',  'ges_logg', 'vosa_logg', 'Δlogg'),
    ('[Fe/H]','ges_feh',  'vosa_feh',  'Δ[Fe/H]'),
]:
    ok = (np.isfinite(m_field_good[ges_col].values) &
          np.isfinite(m_field_good[vosa_col].values))
    delta = m_field_good[vosa_col].values[ok].astype(float) - \
            m_field_good[ges_col].values[ok].astype(float)
    if len(delta):
        print(f'  {label}  (N={len(delta)})')
        print(f'    median = {np.median(delta):+.1f}   '
              f'σ = {np.std(delta):.1f}   '
              f'rms = {np.sqrt(np.mean(delta**2)):.1f}')

# ---------------------------------------------------------------------------
# Save merged table
# ---------------------------------------------------------------------------
merged.to_csv('vosa_ges_comparison.csv', index=False)
print('\nSaved vosa_ges_comparison.csv')

# ---------------------------------------------------------------------------
# Figure: 3-panel comparison plot with error bars
# ---------------------------------------------------------------------------
print('Building comparison figure ...')

params = [
    ('Teff',   'ges_teff', 'vosa_teff', 'ges_e_teff',
     'Teff_min_68cl', 'Teff_max_68cl',
     r'$T_{\rm eff}^{\rm GES}$ [K]', r'$T_{\rm eff}^{\rm VOSA}$ [K]',
     (3500, 7500)),
    ('logg',   'ges_logg', 'vosa_logg', 'ges_e_logg',
     'logg_min_68cl', 'logg_max_68cl',
     r'$\log g^{\rm GES}$',           r'$\log g^{\rm VOSA}$',
     (-0.2, 5.5)),
    ('[Fe/H]', 'ges_feh',  'vosa_feh',  'ges_e_feh',
     'Meta_min_68cl', 'Meta_max_68cl',
     r'$[\rm Fe/H]^{\rm GES}$',        r'$[\rm Fe/H]^{\rm VOSA}$',
     (-2.0, 0.6)),
]

# Plot style per group
STYLE = {
    # (color, marker, zorder, ms, alpha, label)
    'field_good': ('#2196F3', 'o', 4, 4, 0.90, f'Field, good fit (Vgfb $\\leq$ {VGFB_LIMIT})'),
    'field_poor': ('#FF7043', '^', 3, 4, 0.75, f'Field, poor fit (Vgfb $>$ {VGFB_LIMIT})'),
    'cha':        ('#E91E63', 's', 5, 4, 0.90, 'Chamaeleon (YSO candidates)'),
    'm67':        ('#4CAF50', 'D', 5, 4, 0.90, 'M67 (free $A_V$)'),
}

# A&A text width is 17.6 cm.  The figure is drawn at that size and included
# at width=\textwidth, so it is not rescaled by LaTeX and the font sizes below
# are the sizes that reach the page.  Drawing it larger and letting
# \includegraphics shrink it is what made the earlier version illegible.
fig, axes = plt.subplots(1, 3, figsize=(6.93, 2.35))

for ax, (label, gc, vc, ge, vmin_col, vmax_col, xl, yl, lim) in zip(axes, params):
    for _, row in merged[merged['has_ges']].iterrows():
        gv = float(row[gc])
        vv = float(row[vc])
        if not (np.isfinite(gv) and np.isfinite(vv)):
            continue

        # Group
        if row['is_cha']:
            sty = STYLE['cha']
        elif row['is_m67']:
            sty = STYLE['m67']
        elif row['good_fit']:
            sty = STYLE['field_good']
        else:
            sty = STYLE['field_poor']

        color, marker, zo, ms, alpha, _ = sty

        # Error bars: GES (x), VOSA 68% CI (y)
        xerr = float(row[ge]) if np.isfinite(float(row[ge])) else 0.0
        vmin = float(row[vmin_col])
        vmax = float(row[vmax_col])
        yerr_lo = max(vv - vmin, 0.0) if np.isfinite(vmin) else 0.0
        yerr_hi = max(vmax - vv, 0.0) if np.isfinite(vmax) else 0.0

        ax.errorbar(gv, vv,
                    xerr=xerr,
                    yerr=[[yerr_lo], [yerr_hi]],
                    fmt=marker, color=color, ms=ms,
                    elinewidth=0.6, capsize=1.5, capthick=0.6,
                    alpha=alpha, zorder=zo)

    ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.4, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(label, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.4)

# Legend on first panel
handles = [
    Line2D([0],[0], marker=s[1], color=s[0], ls='', ms=4, label=s[5])
    for s in STYLE.values()
]
axes[0].legend(handles=handles, fontsize=5.5, loc='lower right',
               framealpha=0.9, borderpad=0.4)

# No suptitle: the caption names the figure, and a title repeated inside the
# frame only competes with it.
fig.tight_layout()

os.makedirs(OUTDIR, exist_ok=True)
for ext in ('pdf', 'png'):
    path = os.path.join(OUTDIR, f'vosa_comparison.{ext}')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')

print('Done.')
