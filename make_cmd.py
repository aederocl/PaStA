#!/usr/bin/env python3
"""
make_cmd.py
Colour-magnitude diagram (CMD) density map for PaStA Paper I.

Axes: M_G (absolute, observed) vs (BP-RP) (observed, not dereddened).
Extinction is deliberately NOT applied; the dereddened CMD will appear
in Camargo et al. (Paper II, in prep.).

Run from the paper root directory:
    python make_cmd.py

Output: fig/cmd_density.{pdf,png}
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits

# ---------------------------------------------------------------------------
# Load columns
# ---------------------------------------------------------------------------
PASTA = 'pasta.fits'
print('Reading columns ...')
with fits.open(PASTA, memmap=True) as hdul:
    t    = hdul[1].data
    dist = np.array(t['distance'],         dtype=float)
    g    = np.array(t['phot_g_mean_mag'],  dtype=float)
    bp   = np.array(t['phot_bp_mean_mag'], dtype=float)
    rp   = np.array(t['phot_rp_mean_mag'], dtype=float)

# ---------------------------------------------------------------------------
# Compute CMD quantities
# ---------------------------------------------------------------------------
ok   = np.isfinite(dist) & (dist > 0) & np.isfinite(g) & np.isfinite(bp) & np.isfinite(rp)
mg   = g[ok]  - 5.0 * np.log10(dist[ok]) + 5.0   # absolute G magnitude
bprp = bp[ok] - rp[ok]                             # observed BP-RP colour

print(f'Sources plotted: {ok.sum():,}')

# ---------------------------------------------------------------------------
# Plot limits (clip extreme tails for a clean figure)
# ---------------------------------------------------------------------------
BPRP_MIN, BPRP_MAX = -0.6,  4.0
MG_MIN,   MG_MAX   = -5.0, 17.0
NBINS_X, NBINS_Y   = 350,  450     # bin resolution

# ---------------------------------------------------------------------------
# 2D histogram
# ---------------------------------------------------------------------------
counts, xedges, yedges = np.histogram2d(
    bprp, mg,
    bins=[NBINS_X, NBINS_Y],
    range=[[BPRP_MIN, BPRP_MAX], [MG_MIN, MG_MAX]]
)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 8))

img = ax.pcolormesh(
    xedges, yedges, counts.T,
    norm=mcolors.LogNorm(vmin=1, vmax=counts.max()),
    cmap='magma_r',
    rasterized=True
)

cbar = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label('Number of sources per bin', fontsize=10)

ax.set_xlim(BPRP_MIN, BPRP_MAX)
ax.set_ylim(MG_MAX, MG_MIN)          # inverted y-axis (bright at top)
ax.set_xlabel(r'$(G_{\rm BP} - G_{\rm RP})$  [mag]', fontsize=12)
ax.set_ylabel(r'$M_G$  [mag]',                        fontsize=12)
ax.set_title('PaStA colour–magnitude diagram\n(observed, not dereddened)',
             fontsize=11)

ax.grid(True, color='grey', lw=0.3, alpha=0.4)
plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
outdir = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
os.makedirs(outdir, exist_ok=True)

for ext in ('pdf', 'png'):
    path = os.path.join(outdir, f'cmd_density.{ext}')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')

plt.show()
