#!/usr/bin/env python3
"""
analyse_simbad_xmatch.py
Analyse the SIMBAD cross-match results for PaStA: object-type statistics,
CMD distribution of known classes, and comparison with outlier_flag.

Inputs
------
  simbad_xmatch.fits     — output of make_simbad_xmatch.py
  pasta1_public.fits     — released PaStA catalogue (CMD columns, AB, J2016.0)
  outlier_flag.npz       — output of make_outlier_flag.py

Outputs
-------
  fig/simbad_otypes.pdf/.png        — bar chart of top SIMBAD object types
  fig/simbad_cmd.pdf/.png           — CMD colour-coded by SIMBAD type
  fig/simbad_outlier_otypes.pdf/.png — otype breakdown of outlier-flagged sources
  simbad_otype_counts.csv           — full type count table

Run from the paper root directory:
    python analyse_simbad_xmatch.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.table import Table, join
from astropy.io import fits

XMATCH_FILE  = 'simbad_xmatch.fits'
PASTA_FILE   = 'pasta1_public.fits'   # deduplicated, single epoch, AB
FLAG_FILE    = 'outlier_flag.npz'
OUTDIR       = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print('Loading cross-match results ...', flush=True)
xm = Table.read(XMATCH_FILE)
print(f'  {len(xm):,} SIMBAD matches loaded.')

print('Loading PaStA catalogue (CMD columns + outlier_flag) ...', flush=True)
with fits.open(PASTA_FILE, memmap=True) as hdul:
    data = hdul[1].data
    src  = np.array(data['source_id'])
    dist = np.array(data['distance'], dtype=np.float64)
    g    = np.array(data['G'],        dtype=np.float64)
    bp   = np.array(data['BP'],       dtype=np.float64)
    rp   = np.array(data['RP'],       dtype=np.float64)

with np.errstate(divide='ignore', invalid='ignore'):
    mg   = np.where((dist > 0) & np.isfinite(dist) & np.isfinite(g),
                    g - 5.0 * np.log10(dist) + 5.0, np.nan).astype(np.float32)
    bprp = (bp - rp).astype(np.float32)

with np.load(FLAG_FILE) as npz:
    assert np.array_equal(npz['source_id'], src), \
        f'{FLAG_FILE} is not aligned with {PASTA_FILE}'
    outlier_flag = npz['flag'].astype(np.int32)

t = Table({
    'source_id':    src,
    'BP_RP':        bprp,
    'M_G':          mg,
    'outlier_flag': outlier_flag,
})
print(f'  {len(t):,} PaStA sources.')

# Merge on source_id
xm.rename_column('pasta_source_id', 'source_id')
merged = join(xm, t, keys='source_id', join_type='left')
print(f'  {len(merged):,} rows after merge.')

# Convenient pandas frame for grouping
df = merged.to_pandas()
for col in ['otype', 'otypes', 'main_id', 'sp_type']:
    if col in df.columns:
        df[col] = df[col].apply(
            lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip()
        )

# ---------------------------------------------------------------------------
# Object-type statistics
# ---------------------------------------------------------------------------
counts = df['otype'].value_counts()
total_matched = len(df)
total_pasta   = len(t)

print(f'\nTotal PaStA sources:      {total_pasta:>10,}')
print(f'Sources with SIMBAD match:{total_matched:>10,}  '
      f'({100*total_matched/total_pasta:.2f} per cent)')
print(f'\nTop 30 SIMBAD object types:')
print(f'  {"otype":<20s}  {"N":>8}  {"% matched":>10}  {"% total":>10}')
print('  ' + '-' * 55)
for otype, n in counts.head(30).items():
    print(f'  {otype:<20s}  {n:>8,}  {100*n/total_matched:>9.2f}%  '
          f'{100*n/total_pasta:>9.3f}%')

# Save full table
counts.reset_index().rename(
    columns={'index': 'otype', 'otype': 'N'}
).to_csv('simbad_otype_counts.csv', index=False)
print('\nSaved simbad_otype_counts.csv')

# ---------------------------------------------------------------------------
# Group otypes into broad classes for plotting
# ---------------------------------------------------------------------------
# SIMBAD type hierarchy: map to broad label
OTYPE_GROUPS = {
    # Compact & degenerate
    'WD*':  ('White dwarf',    '#7B68EE'),
    'WD?':  ('White dwarf?',   '#7B68EE'),
    'sdB':  ('Hot subdwarf',   '#00BCD4'),
    'sdO':  ('Hot subdwarf',   '#00BCD4'),
    'sdOB': ('Hot subdwarf',   '#00BCD4'),
    'HS*':  ('Hot subdwarf',   '#00BCD4'),
    # Cataclysmic & interacting binaries
    'CV*':  ('CV / interacting','#FF5722'),
    'No*':  ('CV / interacting','#FF5722'),
    'DN*':  ('CV / interacting','#FF5722'),
    'AM*':  ('CV / interacting','#FF5722'),
    'XB*':  ('X-ray binary',    '#E91E63'),
    'LXB':  ('X-ray binary',    '#E91E63'),
    'HXB':  ('X-ray binary',    '#E91E63'),
    'Sy*':  ('Symbiotic star',  '#9C27B0'),
    # Young / pre-MS
    'TT*':  ('T Tauri / YSO',  '#4CAF50'),
    'TTau*':('T Tauri / YSO',  '#4CAF50'),
    'Y*O':  ('T Tauri / YSO',  '#4CAF50'),
    'Ae*':  ('T Tauri / YSO',  '#4CAF50'),
    'Or*':  ('T Tauri / YSO',  '#4CAF50'),
    # Variable stars (broad)
    'V*':   ('Variable star',  '#FF9800'),
    'RR*':  ('RR Lyr',         '#FFC107'),
    'Cep':  ('Cepheid',        '#FFD54F'),
    'Mira': ('Mira',           '#FFCC02'),
    'LP*':  ('Long-period var','#FFA726'),
    # Giants / evolved
    'RG*':  ('Red giant',      '#F44336'),
    'HB*':  ('Horiz.-branch',  '#FF7043'),
    'AGB*': ('AGB star',       '#EF5350'),
    'sg*':  ('Supergiant',     '#B71C1C'),
    'WR*':  ('Wolf-Rayet',     '#880E4F'),
    # Emission-line stars
    'Be*':  ('Be star',        '#03A9F4'),
    'Em*':  ('Emission-line',  '#29B6F6'),
    # Extragalactic
    'AGN':  ('AGN',            '#607D8B'),
    'QSO':  ('QSO',            '#546E7A'),
    'BLL':  ('BL Lac',         '#455A64'),
    'Sy1':  ('Seyfert',        '#37474F'),
    'Sy2':  ('Seyfert',        '#37474F'),
    'G':    ('Galaxy',         '#78909C'),
    # Generic stellar (keep separate — usually majority)
    '*':    ('Star (generic)', '#BDBDBD'),
    '**':   ('Double star',    '#9E9E9E'),
}

def map_group(otype):
    return OTYPE_GROUPS.get(otype, ('Other', '#E0E0E0'))

df['broad_class'] = df['otype'].map(lambda x: map_group(x)[0])
df['broad_color'] = df['otype'].map(lambda x: map_group(x)[1])

# ---------------------------------------------------------------------------
# Figure 1: bar chart of top otypes (excluding generic '*' from display)
# ---------------------------------------------------------------------------
interesting = counts[counts.index != '*'].head(25)

fig, ax = plt.subplots(figsize=(10, 6))
colors = [map_group(o)[1] for o in interesting.index]
bars = ax.barh(interesting.index[::-1], interesting.values[::-1],
               color=colors[::-1], edgecolor='white', linewidth=0.3)
ax.set_xlabel('Number of sources', fontsize=11)
ax.set_title(f'Top SIMBAD object types in PaStA ({total_matched:,} matched sources)',
             fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.grid(axis='x', lw=0.3, alpha=0.5)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUTDIR, f'simbad_otypes.{ext}'), dpi=150,
                bbox_inches='tight')
print(f'Saved fig/simbad_otypes.{{pdf,png}}')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: CMD with SIMBAD types overlaid
# ---------------------------------------------------------------------------
bp_rp = np.array(t['BP_RP'], dtype=np.float64)
m_g   = np.array(t['M_G'],   dtype=np.float64)

fig, ax = plt.subplots(figsize=(8, 7))

# Grey background: all PaStA sources (density hexbin)
ok_bg = np.isfinite(bp_rp) & np.isfinite(m_g)
ax.hexbin(bp_rp[ok_bg], m_g[ok_bg], gridsize=300, cmap='Greys',
          bins='log', linewidths=0, zorder=1)

# Overlay interesting SIMBAD classes
PLOT_CLASSES = [
    ('White dwarf',     'o', 5, 20),
    ('Hot subdwarf',    's', 5, 20),
    ('CV / interacting','D', 5, 18),
    ('X-ray binary',    'P', 5, 25),
    ('Symbiotic star',  '*', 5, 30),
    ('T Tauri / YSO',   '^', 4, 18),
    ('Variable star',   'o', 3,  8),
    ('QSO',             'x', 5, 20),
    ('AGN',             '+', 5, 20),
]

# Build colour map from OTYPE_GROUPS
class_color = {}
for ot, (bc, col) in OTYPE_GROUPS.items():
    class_color.setdefault(bc, col)

handles = []
for cls, marker, zo, ms in PLOT_CLASSES:
    sel = df['broad_class'] == cls
    if sel.sum() == 0:
        continue
    src_ids = df.loc[sel, 'source_id'].values
    # match back to pasta row indices
    pasta_ids = np.array(t['source_id'])
    idx = np.where(np.isin(pasta_ids, src_ids))[0]
    x = bp_rp[idx]
    y = m_g[idx]
    valid = np.isfinite(x) & np.isfinite(y)
    col = class_color.get(cls, '#999999')
    ax.scatter(x[valid], y[valid], marker=marker, s=ms, color=col,
               alpha=0.8, zorder=zo, label=f'{cls} (N={valid.sum():,})',
               linewidths=0.3)

ax.set_xlim(-0.94, 3.16)   # the old Vega limits (-0.6, 3.5) shifted to AB
ax.set_ylim(16.11, -3.89)  # the old Vega limits (16, -4) shifted to AB
ax.set_xlabel(r'$(G_{\rm BP} - G_{\rm RP})_{\rm AB}$ [mag]', fontsize=12)
ax.set_ylabel(r'$M_{G,\,\rm AB}$ [mag]', fontsize=12)
ax.set_title('PaStA CMD — SIMBAD-classified sources', fontsize=11)
ax.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.85)
ax.grid(True, lw=0.2, alpha=0.3)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUTDIR, f'simbad_cmd.{ext}'), dpi=150,
                bbox_inches='tight')
print(f'Saved fig/simbad_cmd.{{pdf,png}}')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: otype breakdown of outlier-flagged sources
# ---------------------------------------------------------------------------
outlier_flag = np.array(t['outlier_flag'], dtype=np.int32)
# Flag bit names (from §6 of the paper)
BIT_LABELS = {
    0:  'FUV excess',
    1:  'NUV excess',
    4:  'W1/W2 excess',
    5:  'W3/W4 excess',
    6:  'W1/W2 deficit',
    7:  'W3/W4 deficit',
    8:  'Shape outlier',
}

pasta_ids_arr = np.array(t['source_id'])
flagged_ids   = pasta_ids_arr[outlier_flag > 0]
df_flagged    = df[df['source_id'].isin(flagged_ids)]

if len(df_flagged) > 0:
    flag_counts = df_flagged['otype'].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_f = [map_group(o)[1] for o in flag_counts.index]
    ax.barh(flag_counts.index[::-1], flag_counts.values[::-1],
            color=colors_f[::-1], edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Number of outlier-flagged sources with SIMBAD match', fontsize=10)
    ax.set_title('SIMBAD types of PaStA outlier-flagged sources', fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='x', lw=0.3, alpha=0.5)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTDIR, f'simbad_outlier_otypes.{ext}'), dpi=150,
                    bbox_inches='tight')
    print(f'Saved fig/simbad_outlier_otypes.{{pdf,png}}')
    plt.close(fig)

    print(f'\nOutlier-flagged sources with SIMBAD match: {len(df_flagged):,}')
    print('Top otypes among outliers:')
    for otype, n in flag_counts.items():
        print(f'  {otype:<20s}  {n:>7,}')

print('\nDone.')
