#!/usr/bin/env python3
"""
make_filter_curves.py
Transmission curves of the twelve PaStA passbands.

Replaces fig/PASTA_responsecurves.png, which was an orphan: no script
survived, it carried a seaborn grey background and an embedded title,
its legend was not in wavelength order, and it was raster.

Design notes
------------
* Colour groups the bands by instrument (GALEX, Gaia, 2MASS, WISE)
  rather than giving twelve arbitrary colours, using the Okabe-Ito
  colourblind-safe palette.  Bands are told apart by their labels and
  their position, which is what the reader actually needs.
* Bands are labelled in place, so there is no twelve-entry legend to
  cross-reference; only the four instruments are in the legend.
* Wavelength is in micron, matching the 0.15-22 micron range quoted in
  the text, on a logarithmic axis.
* Vector output, white background, no title: the caption does that job.

The profiles come from the SVO Filter Profile Service, under exactly the
identifiers that Appendix B tells the reader to submit to VOSA, so that
the curves drawn here and the curves the service applies to a
reproduction of that analysis are the same objects.  SVO repackages the
primary published response functions:

    GALEX FUV, NUV     GALEX team (Morrissey et al. 2007)
    Gaia BP, G, RP     ESA DR3 passbands (GAIA3 revision; Riello et al. 2021)
    2MASS J, H, Ks     Cohen et al. (2003) relative spectral response
    WISE W1-W4         Wright et al. (2010) weighted mean RSR

An earlier version of this figure read the curves from the speclite
library, because svo2.cab.inta-csic.es was refusing connections on every
port on 14/Aug/2026.  The service came back on 17/Aug and the profiles
were swapped, but not for the reason recorded at the time.  The concern
was the GALEX sampling, 15 points in FUV and 25 in NUV against 471 and
1321 from SVO; those 15 points turn out to lie on the SVO curve to
better than 0.05 in normalised transmission, and at the width these
bands occupy in a single-column figure the two are indistinguishable.

The swap is worth making for provenance instead.  The 2MASS profiles are
not the same objects in the two libraries: the pivot wavelengths differ
by 38, 37 and 36 A in J, H and Ks, and Table 1 of the paper quotes the
SVO values.  The figure and the table therefore used to describe
slightly different passbands.  They now come from one source, under the
identifiers Appendix B hands to the reader, and the pivot printed below
for each band reproduces its row in Table 1 exactly.

Each VOTable is cached under filters/ on first use, so the figure
rebuilds without network access thereafter and the exact profiles
travel with the code.

Input
-----
  filters/<band>.xml, fetched from SVO on first run

Output
------
  fig/pasta_filter_curves.{pdf,png}

Run from the paper root directory:
    python make_filter_curves.py
"""

import os
import urllib.request

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astropy.io.votable import parse

OUTDIR = os.environ.get('PASTA_FIGDIR',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))
FILTERDIR = 'filters'
SVO_URL = 'http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID='

# Okabe-Ito, colourblind safe
COLOUR = {
    'GALEX': '#CC79A7',
    'Gaia': '#0072B2',
    '2MASS': '#009E73',
    'WISE': '#D55E00',
}

# (label, SVO filter identifier, instrument, label row, tint)
# The identifiers are those of Appendix B and of the Table 1 caption, one
# consistent revision per instrument (GAIA3 for the Gaia passbands).
# label row staggers the in-place labels so that neighbouring bands do
# not collide; 0 is the lower row, 1 the upper one.
# tint lightens the instrument colour towards white, 0 = full strength.
# Bands within an instrument run light to dark with wavelength, so that
# adjacent and overlapping passbands stay individually readable.
BANDS = [
    ('FUV',  'GALEX/GALEX.FUV', 'GALEX', 0, 0.35),
    ('NUV',  'GALEX/GALEX.NUV', 'GALEX', 0, 0.00),
    ('BP',   'GAIA/GAIA3.Gbp',  'Gaia',  1, 0.42),
    ('G',    'GAIA/GAIA3.G',    'Gaia',  0, 0.00),
    ('RP',   'GAIA/GAIA3.Grp',  'Gaia',  1, 0.00),
    ('J',    '2MASS/2MASS.J',   '2MASS', 0, 0.40),
    ('H',    '2MASS/2MASS.H',   '2MASS', 1, 0.20),
    ('Ks',   '2MASS/2MASS.Ks',  '2MASS', 0, 0.00),
    ('W1',   'WISE/WISE.W1',    'WISE',  1, 0.45),
    ('W2',   'WISE/WISE.W2',    'WISE',  0, 0.30),
    ('W3',   'WISE/WISE.W3',    'WISE',  1, 0.15),
    ('W4',   'WISE/WISE.W4',    'WISE',  0, 0.00),
]


def tint(hexcolour, f):
    """Blend a colour towards white by fraction f."""
    r, g, b = mpl.colors.to_rgb(hexcolour)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

os.makedirs(OUTDIR, exist_ok=True)

mpl.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})


def load(name, filter_id):
    """SVO transmission curve as (wavelength/micron, transmission, pivot)."""
    path = os.path.join(FILTERDIR, f'{name}.xml')
    if not os.path.exists(path):
        print(f'  fetching {filter_id} from SVO ...', flush=True)
        urllib.request.urlretrieve(SVO_URL + filter_id, path)
    table = parse(path).get_first_table()
    wl = np.asarray(table.array['Wavelength'], dtype=float)   # Angstrom
    tr = np.asarray(table.array['Transmission'], dtype=float)
    pivot = float({p.name: p.value for p in table.params}['WavelengthPivot'])
    return wl / 1e4, tr / tr.max(), pivot / 1e4


os.makedirs(FILTERDIR, exist_ok=True)

print('Loading filter profiles from SVO ...', flush=True)
curves = []
for name, fid, inst, row, tf in BANDS:
    wl, tr, pivot = load(name, fid)
    wmean = np.trapz(wl * tr, wl) / np.trapz(tr, wl)
    curves.append((name, inst, row, wl, tr, wmean, tf))
    print(f'  {name:3s} {inst:6s} mean {wmean:7.3f} um  pivot {pivot:7.3f} um  '
          f'({len(wl)} points)', flush=True)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(3.5, 2.1))
ax = fig.add_axes([0.115, 0.185, 0.875, 0.80])

for name, inst, row, wl, tr, wmean, tf in curves:
    base = COLOUR[inst]
    c = tint(base, tf)
    # Gaia G spans the whole optical and encloses BP and RP.  Filling all
    # three stacks the transparencies into a dark bar where the BP and RP
    # cutoffs cross, so G is drawn as an envelope instead.
    if name == 'G':
        ax.plot(wl, tr, color=base, lw=1.0, zorder=4)
    else:
        ax.fill_between(wl, 0, tr, color=c, alpha=0.75, linewidth=0)
        ax.plot(wl, tr, color=c, lw=0.5, zorder=3)
    ax.text(wmean, 1.04 + 0.115 * row, name,
            ha='center', va='bottom', fontsize=5.5, color=base)

ax.set_xscale('log')
ax.set_xlim(0.12, 32)
ax.set_ylim(0, 1.40)

ticks = [0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20]
ax.set_xticks(ticks)
ax.set_xticklabels([('%g' % t) for t in ticks])
ax.set_xticks([], minor=True)
ax.set_yticks([0, 0.5, 1.0])

ax.set_xlabel(r'wavelength ($\mu$m)')
ax.set_ylabel('normalised transmission')

ax.legend(handles=[Patch(facecolor=COLOUR[k], alpha=0.55, edgecolor=COLOUR[k],
                         label=k) for k in ('GALEX', 'Gaia', '2MASS', 'WISE')],
          loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=4,
          frameon=False, fontsize=5.5, handlelength=1.1,
          handleheight=0.7, columnspacing=1.2, handletextpad=0.4)

for side in ('top', 'right'):
    ax.spines[side].set_visible(False)

for ext in ('pdf', 'png'):
    path = os.path.join(OUTDIR, f'pasta_filter_curves.{ext}')
    fig.savefig(path, dpi=300, facecolor='white', edgecolor='none')
    print(f'Saved {path}')

plt.close(fig)
print('Done.')
