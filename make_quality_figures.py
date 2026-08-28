#!/usr/bin/env python3
"""
make_quality_figures.py
Three of the Appendix A diagnostics for PaStA Paper I, redrawn.

Replaces a set of orphan PNGs -- fig/Gaia_AllWISE_coordinates.png,
fig/Gaia_Galex_coordinates.png and the seven panels under fig/sigma/ -- which
had no script and whose axis labels were illegible in print. The illegibility
was not a property of the images: they were drawn at one size and then shrunk
to fit the column, either by \\includegraphics or by a \\resizebox around a
LaTeX tabular, so the lettering came down with them by the same factor.

The rule applied here is that a figure is drawn at the width it will occupy on
the page and included at that width, so that LaTeX does not rescale it and the
font sizes set below are the sizes that reach the reader. A&A's text width is
17.6 cm (6.93 in) and its column width 8.8 cm (3.46 in).

Outputs, all to $PASTA_FIGDIR (default ./fig):

  coords_diff.{pdf,png}     Gaia minus survey position, AllWISE and GALEX
  gaia_mag_error.{pdf,png}  Gaia magnitude against its uncertainty, BP, G, RP
  wise_mag_error.{pdf,png}  AllWISE magnitude against its uncertainty, W1-W4

Inputs
------
  pasta1_public.fits    magnitudes and uncertainties (AB, J2016.0)
  pasta1_internal.fits  needed only for coords_diff, which compares the Gaia
                        position with the GALEX and AllWISE ones; the public
                        table carries a single position by design

Run from the paper root:
    python3 make_quality_figures.py            # all three
    python3 make_quality_figures.py coords     # or name the ones wanted
    python3 make_quality_figures.py gaia wise
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits

PUBLIC = 'pasta1_public.fits'
INTERNAL = 'pasta1_internal.fits'
OUTDIR = os.environ.get('PASTA_FIGDIR',
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig'))

TEXTWIDTH = 6.93          # in, A&A \textwidth
LABELSIZE = 8
TICKSIZE = 7
TITLESIZE = 8.5


def save(fig, stem):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(OUTDIR, '%s.%s' % (stem, ext))
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print('  saved %s' % path)
    plt.close(fig)


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=LABELSIZE)
    if title:
        ax.set_title(title, fontsize=TITLESIZE)
    ax.tick_params(labelsize=TICKSIZE)


# ---------------------------------------------------------------------------
# magnitude against uncertainty
# ---------------------------------------------------------------------------
def mag_error_panel(ax, mag, err, band):
    """Density of magnitude against uncertainty, with the running median.

    Only sources with a finite uncertainty are shown: in W3 and W4 a null
    uncertainty marks a censored upper limit rather than a measurement
    (Sect. 3.4), and plotting those would draw a locus that is an artefact of
    the AllWISE sensitivity floor rather than a property of the photometry.
    """
    ok = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    m, e = mag[ok], err[ok]
    if len(m) == 0:
        ax.text(0.5, 0.5, 'no measurements', ha='center', transform=ax.transAxes)
        return

    ax.hexbin(m, e, yscale='log', gridsize=110, mincnt=1,
              cmap='Greys', norm=LogNorm(), linewidths=0)

    # running median in 0.25 mag bins, drawn only where the bin is populated
    lo, hi = np.percentile(m, [0.05, 99.95])
    edges = np.arange(lo, hi + 0.25, 0.25)
    idx = np.digitize(m, edges)
    xs, ys = [], []
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() >= 20:
            xs.append(0.5 * (edges[b - 1] + edges[b]))
            ys.append(np.median(e[sel]))
    if xs:
        ax.plot(xs, ys, color='#2E7D32', lw=1.4, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_yscale('log')
    style(ax, '$%s$' % band, r'$\sigma$ (mag)')
    ax.grid(True, lw=0.3, alpha=0.35)


def figure_mag_error(bands, stem, source_note):
    with fits.open(PUBLIC, memmap=True) as h:
        d = h[1].data
        cols = {}
        for b in bands:
            cols[b] = (np.asarray(d[b], float), np.asarray(d['e_%s' % b], float))

    # One row, however many bands.  A 2x2 grid reads more comfortably, but
    # these are full-width floats in an appendix with a page budget, and a
    # single row costs less than half the vertical space for the same
    # information.
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(TEXTWIDTH, 2.05))
    axes = np.atleast_1d(axes).ravel()
    for ax, b in zip(axes, bands):
        mag_error_panel(ax, cols[b][0], cols[b][1], b)
        print('    %-3s %9d measurements' % (b, np.isfinite(cols[b][1]).sum()))
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    save(fig, stem)


# ---------------------------------------------------------------------------
# coordinate differences
# ---------------------------------------------------------------------------
def figure_coords():
    """Gaia minus survey position, for AllWISE and for GALEX.

    The comparison is made on the coordinate pair the cross-match itself used:
    the Gaia position propagated back to J2000.0 against the survey position as
    the survey publishes it. That is the residual of the pairing, and it is
    bounded by the search radius by construction, so no match lies outside the
    frame. Comparing the Gaia J2016.0 position with a survey position
    propagated forward instead would be a different quantity, would not match
    the separations quoted in Sect. 3.2, and would place a few hundred sources
    beyond the radius through the propagation itself.
    """
    with fits.open(INTERNAL, memmap=True) as h:
        d = h[1].data
        g = lambda c: np.asarray(d[c], float)
        ra0, dec0 = g('ra_gaia_2000'), g('dec_gaia_2000')
        sets = {}
        for s, name in (('allwise', 'AllWISE'), ('galex', 'GALEX')):
            dra = (ra0 - g('ra_%s_2000' % s) + 180.0) % 360.0 - 180.0
            dra *= 3600.0 * np.cos(np.radians(dec0))     # to arcsec, on the sky
            ddec = (dec0 - g('dec_%s_2000' % s)) * 3600.0
            sets[name] = (dra, ddec)

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
    for ax, (name, (dra, ddec)) in zip(axes, sets.items()):
        ok = np.isfinite(dra) & np.isfinite(ddec)
        x, y = dra[ok], ddec[ok]
        sep = np.hypot(x, y)
        hb = ax.hexbin(x, y, gridsize=140, extent=(-1.1, 1.1, -1.1, 1.1),
                       mincnt=1, cmap='viridis', norm=LogNorm(), linewidths=0)
        ax.axhline(0, color='w', lw=0.5, alpha=0.6)
        ax.axvline(0, color='w', lw=0.5, alpha=0.6)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        style(ax,
              r'$\Delta\alpha\,\cos\delta$ (arcsec)',
              r'$\Delta\delta$ (arcsec)',
              '%s: median %.2f arcsec' % (name, np.median(sep)))
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=TICKSIZE - 1)
        cb.set_label('sources per cell', fontsize=LABELSIZE - 1)
        print('    %-8s median separation %.3f arcsec, 95th %.3f'
              % (name, np.median(sep), np.percentile(sep, 95)))
    fig.tight_layout()
    save(fig, 'coords_diff')


TASKS = {
    'coords': figure_coords,
    'gaia': lambda: figure_mag_error(['BP', 'G', 'RP'], 'gaia_mag_error', 'Gaia DR3'),
    'wise': lambda: figure_mag_error(['W1', 'W2', 'W3', 'W4'], 'wise_mag_error', 'AllWISE'),
}

if __name__ == '__main__':
    wanted = sys.argv[1:] or list(TASKS)
    unknown = [w for w in wanted if w not in TASKS]
    if unknown:
        sys.exit('unknown figure(s): %s; choose from %s' % (unknown, list(TASKS)))
    for name in wanted:
        print('%s ...' % name)
        TASKS[name]()
    print('done.')
