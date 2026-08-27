#!/usr/bin/env python3
"""
verify_paper_numbers.py
Recompute every number quoted in PaStA Paper I from the data products, so that
the draft can be checked against the catalogue rather than against notes.

Written for the sweep of 17/Aug/2026, which found six wrong numbers in a draft
whose numbers had already been checked once.  The failure mode it exists to
catch is a value that was correct when it was written and was not updated when
the catalogue changed: the AB harmonisation of 11/Aug regenerated every table
and figure but not the prose, and the deduplication of 30/Jul changed counts
derived from the cross-matches.

Sections (run all, or name the ones wanted on the command line):

    counts      row and column counts, per-band detection fractions, the G-bin
                table of Appendix A, the censoring statistics of Sect. 3.4,
                proper motions and distances (Sect. 3.2)
    sky         MOC sky coverage and the Galactic-latitude occupancy (Sect. 3.1)
    match       cross-match separations and the AB offsets of Table 2
    epoch       the Sect. 3.1 bound on the cross-match epoch approximation:
                proper motions, the displacement a residual baseline error
                would cause, and the separation-vs-proper-motion residual
    pivot       the pivot wavelengths of Table 1, against the SVO service
    cmd         CMD regions (Table 3), the outlier flag (Table 4), the
                per-region SED statistics (Sect. 5) and the Appendix A
                argument about faint W3 detections
    flagdiag    the Sect. 6.1 diagnostics: the 6.5/3.5/0.51 chain, the S/N of
                the raw excesses, and what the significance condition keeps
    external    the SIMBAD census (Table C.1) and the Gaia-ESO sample

Every value is printed next to the value the paper quotes, so a disagreement is
visible without cross-referencing the tex.

    python verify_paper_numbers.py            # everything
    python verify_paper_numbers.py cmd flagdiag

Requires astropy and astropy-healpix; the pivot section needs network access to
the SVO Filter Profile Service.  Run from the paper root directory.
"""

import os
import sys
import csv
import urllib.request

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

PUBLIC = 'pasta1_public.fits'
INTERNAL = 'pasta1_internal.fits'
# The cross-match radius actually used: sr=0.00027 deg in getcat_pasta.py.
# The paper rounds it to 1 arcsec in prose; the difference matters to none of
# the numbers below, and 0.972 is the conservative choice for the fraction of
# sources an epoch error would push outside it.
MATCH_RADIUS = 0.972
SIMBAD = 'simbad_xmatch.fits'
GES = 'ges_pasta_matched.fits'
VOSA_CSV = 'vosa_ges_comparison.csv'
CACHE = os.environ.get('SVO_CACHE', 'filters')   # shared with make_filter_curves.py

# Band, and the pivot wavelength in micron used throughout the analysis.
BANDS = [('FUV', 0.1535079), ('NUV', 0.2300785), ('BP', 0.5109712),
         ('G', 0.6217590), ('RP', 0.7769023), ('J', 1.2393089),
         ('H', 1.6494947), ('Ks', 2.1638606), ('W1', 3.3682213),
         ('W2', 4.6179057), ('W3', 12.0718118), ('W4', 22.1944039)]
NAMES = [b[0] for b in BANDS]
LAM = np.array([b[1] for b in BANDS])
NB = len(BANDS)
I = {n: i for i, n in enumerate(NAMES)}

# CMD regions, AB system.  Identical to make_sed_regions.py and
# make_outlier_flag.py; the boxes are disjoint and half-open, [lo, hi).
REGIONS = [('OB', (-0.94, -0.24), (-4.89, 2.11)),
           ('A', (-0.24, 0.11), (0.11, 4.11)),
           ('F', (0.11, 0.31), (2.61, 5.11)),
           ('G', (0.31, 0.51), (4.11, 6.11)),
           ('K', (0.51, 1.16), (5.61, 8.11)),
           ('M', (1.16, 3.66), (8.11, 17.11)),
           ('RGB', (0.66, 1.46), (-2.89, 0.41)),
           ('RC', (0.66, 0.96), (0.41, 1.41)),
           ('WD', (-0.74, 0.06), (11.11, 16.11))]
NR = len(REGIONS)
N_SIGMA = 3.0        # deviation threshold in units of the region scatter
SIG_SOURCE = 3.0     # and in units of the source's own photometric error
DETFRAC_MIN = 0.50   # below this the region median is an upper limit
SKY_DEG2 = 4 * np.pi * (180.0 / np.pi) ** 2


def head(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def load_photometry(path=PUBLIC, extra=()):
    """Magnitudes, uncertainties and any extra columns, as float arrays."""
    with fits.open(path, memmap=True) as hdul:
        t = hdul[1].data
        n = len(t)
        ncol = len(hdul[1].columns)
        mag = np.column_stack([np.array(t[b], dtype=np.float64) for b in NAMES])
        err = np.column_stack([np.array(t['e_' + b], dtype=np.float64)
                               for b in NAMES])
        other = {c: np.array(t[c]) for c in extra}
    return n, ncol, mag, err, other


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------
def section_counts():
    head('COUNTS, DETECTION FRACTIONS, CENSORING, ASTROMETRY')
    n, ncol, mag, err, o = load_photometry(
        extra=('distance', 'PMRA', 'PMDEC'))
    print(f'rows {n:,} [paper 9,705,879]   columns {ncol} [paper 46]')

    det = np.isfinite(mag) & np.isfinite(err)
    has = np.isfinite(mag)

    print('\n--- overall fractions (a band counts only with a finite error) ---')
    quoted = {'FUV': 4.1, 'NUV': 100.0, 'BP': 99.8, 'G': 100.0, 'RP': 99.8,
              'J': 99.6, 'H': 99.6, 'Ks': 98.8, 'W1': 99.9, 'W2': 100.0,
              'W3': 40.2, 'W4': 5.3}
    for b in NAMES:
        j = I[b]
        print(f'  {b:>3s}: {det[:, j].sum():>9,}  {100*det[:, j].mean():6.2f} '
              f'per cent  [paper {quoted[b]:5.1f}]   '
              f'(magnitude alone: {has[:, j].sum():>9,})')

    print('\n--- sources lacking Gaia photometry ---')
    noG = ~has[:, I['G']]
    print(f'  no G {noG.sum():,} [paper 214]; no BP {int((~has[:, I["BP"]]).sum()):,}; '
          f'no RP {int((~has[:, I["RP"]]).sum()):,}')
    print(f'  no G but has RP: {int((noG & has[:, I["RP"]]).sum()):,} [paper 76]')
    print(f'  no G, all of NUV/W1/W2/J measured: '
          f'{int((noG & det[:, I["NUV"]] & det[:, I["W1"]] & det[:, I["W2"]] & det[:, I["J"]]).sum()):,}'
          f' [paper: all 214]')

    print('\n--- Table B.1: per cent measured, by G bin ---')
    G = mag[:, I['G']]
    edges = [(-np.inf, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20),
             (20, np.inf)]
    labels = ['<10', '10-12', '12-14', '14-16', '16-18', '18-20', '>20']
    print('G bin       N_G      ' + ' '.join(f'{b:>6s}' for b in NAMES))
    total = 0
    for (lo, hi), lab in zip(edges, labels):
        m = has[:, I['G']] & (G >= lo) & (G < hi)
        ng = int(m.sum())
        total += ng
        print(f'{lab:<8s} {ng:>10,}  '
              + ' '.join(f'{100*(det[:, I[b]] & m).sum()/ng:6.1f}' for b in NAMES))
    m = has[:, I['G']]
    print(f'{"Total":<8s} {total:>10,}  '
          + ' '.join(f'{100*(det[:, I[b]] & m).sum()/total:6.1f}' for b in NAMES))
    print(f'  sum of N_G {total:,} [paper 9,705,665]; catalogue minus sum '
          f'{n - total} [paper 214]')

    print('\n--- Sect. 3.4: censoring ---')
    for b in ('W3', 'W4', 'Ks'):
        j = I[b]
        nolim = has[:, j] & ~np.isfinite(err[:, j])
        print(f'  {b}: {100*nolim.sum()/has[:, j].sum():.1f} per cent of the '
              f'tabulated magnitudes carry no uncertainty '
              f'[paper W3 59.8, W4 94.7]')
    j4 = I['W4']
    lim = has[:, j4] & ~np.isfinite(err[:, j4])
    for lab, m, want in (('upper limits', lim, '0.65'),
                         ('detections', det[:, j4], '1.62')):
        v = mag[m, j4]
        p16, p50, p84 = np.percentile(v, [16, 50, 84])
        print(f'  W4 {lab:<12s}: N={int(m.sum()):>9,} median {p50:.2f} '
              f'[paper 15.55 for the limits], 16/84 spread {p84-p16:.2f} '
              f'[paper {want}]')
    for lab, m in (('limits', lim), ('detections', det[:, j4])):
        mm = m & has[:, I['G']]
        r = np.corrcoef(G[mm], mag[mm, j4])[0, 1]
        print(f'  corr(G, W4), {lab:<10s}: r = {r:+.3f} '
              f'[paper +0.12 and +0.65]')

    print('\n--- Sect. 3.2: proper motion, displacement, distance ---')
    pm = np.hypot(o['PMRA'].astype(np.float64), o['PMDEC'].astype(np.float64))
    ok = np.isfinite(pm)
    print(f'  total PM median {np.median(pm[ok]):.1f} [7.5], 95th '
          f'{np.percentile(pm[ok], 95):.1f} [24.6], max {pm[ok].max():.0f} '
          f'[319, and none above 320] mas/yr')
    print(f'  N(PM>200) {int((pm[ok] > 200).sum())} [14]; '
          f'N(PM>62.5, i.e. 1 arcsec in 16 yr) {int((pm[ok] > 62.5).sum()):,} [13,663]')
    disp = pm * 16.0 / 1000.0
    print(f'  16-yr displacement median {np.median(disp[ok]):.2f}" [0.12], '
          f'95th {np.percentile(disp[ok], 95):.2f}" [0.39], '
          f'>1" {int((disp[ok] > 1).sum()):,} = '
          f'{100*(disp[ok] > 1).sum()/n:.2f} per cent [13,663 / 0.14]')
    d = o['distance'].astype(np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    print(f'  distance median {np.median(d):,.0f} pc [paper "1.4 kpc"]')
    g = G[has[:, I['G']]]
    print(f'  G (AB) range {g.min():.2f} to {g.max():.2f} [3.86 to 21.05]')


# ---------------------------------------------------------------------------
# sky
# ---------------------------------------------------------------------------
def section_sky():
    head('SKY COVERAGE (Sect. 3.1)')
    from astropy_healpix import HEALPix
    with fits.open(PUBLIC, memmap=True) as h:
        ra = np.array(h[1].data['ra_2016'], dtype=np.float64)
        dec = np.array(h[1].data['dec_2016'], dtype=np.float64)

    # A MOC built with MOC.from_lonlat(max_norder=k) covers exactly the order-k
    # cells holding at least one source, so its sky fraction is the occupied
    # fraction below.  mocpy is therefore not needed to reproduce the figure.
    occ8 = None
    for order in (5, 6, 7, 8, 9):
        hp = HEALPix(nside=2 ** order, order='nested')
        occ = np.zeros(hp.npix, dtype=bool)
        occ[hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)] = True
        frac = occ.sum() / hp.npix
        flag = '  <-- quoted in the paper' if order == 8 else ''
        print(f'  order {order}: {100*frac:5.2f} per cent = '
              f'{frac*SKY_DEG2:,.0f} deg^2 [paper 64.9 / 26,785]{flag}')
        if order == 6:
            occ6, hp6 = occ, hp
        if order == 8:
            occ8, hp8 = occ, hp

    print('\n--- occupancy by Galactic latitude ---')
    for order, hp, occ in ((6, hp6, occ6), (8, hp8, occ8)):
        lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
        b = np.abs(SkyCoord(ra=lon, dec=lat, frame='icrs').galactic.b.deg)
        lo = occ[b < 5].mean() * 100
        hi = occ[b > 30].mean() * 100
        note = '  <-- the paper quotes these' if order == 6 else ''
        print(f'  order {order}: |b|<5 {lo:5.1f} per cent, |b|>30 {hi:5.1f} '
              f'per cent [paper 6 and 97]{note}')


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------
def section_match():
    head('CROSS-MATCH SEPARATIONS AND THE AB OFFSETS (Sects. 3.1 and 3.3)')
    d2r = np.pi / 180.0

    def sep(ra1, de1, ra2, de2):
        lam1, phi1, lam2, phi2 = (x * d2r for x in (ra1, de1, ra2, de2))
        a = (np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2)
             * np.sin((lam2 - lam1) / 2) ** 2)
        return 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / d2r * 3600.0

    cols = ('ra_gaia_2000', 'dec_gaia_2000', 'ra_gaia_2016', 'dec_gaia_2016',
            'ra_galex_2000', 'dec_galex_2000', 'ra_galex_2016', 'dec_galex_2016',
            'ra_allwise_2000', 'dec_allwise_2000', 'ra_allwise_2016',
            'dec_allwise_2016')
    with fits.open(INTERNAL, memmap=True) as h:
        c = {k: np.array(h[1].data[k], dtype=np.float64) for k in cols}

    # All four epoch pairings, because the quoted percentiles identify which
    # one they refer to: only J2000 vs J2000 stays inside the 1 arcsec radius.
    print('  [paper: AllWISE median 0.15, 95th 0.54; GALEX 0.55, 0.92;')
    print('   and "all matched separations are within the 1 arcsec radius"]')
    for survey in ('allwise', 'galex'):
        for eg, es in (('2000', '2000'), ('2016', '2016'),
                       ('2016', '2000'), ('2000', '2016')):
            s = sep(c[f'ra_gaia_{eg}'], c[f'dec_gaia_{eg}'],
                    c[f'ra_{survey}_{es}'], c[f'dec_{survey}_{es}'])
            ok = np.isfinite(s)
            print(f'  Gaia@{eg} vs {survey.upper():>7s}@{es}: median '
                  f'{np.median(s[ok]):.3f}  95th {np.percentile(s[ok], 95):.3f}'
                  f'  max {s[ok].max():.3f}  N(>1")={int((s[ok] > 1).sum()):,}')

    print('\n--- Table 2, as actually applied to the released table ---')
    paper = {'FUV': 0.0, 'NUV': 0.0, 'BP': 0.0155, 'G': 0.1136, 'RP': 0.3561,
             'J': 0.8938, 'H': 1.3743, 'Ks': 1.8402, 'W1': 2.6733,
             'W2': 3.3126, 'W3': 5.1483, 'W4': 6.5942}
    with fits.open(INTERNAL, memmap=True) as h:
        t = h[1].data
        for b in NAMES:
            ab = np.array(t[f'{b}_ab'], dtype=np.float64)
            vega = np.array(t[f'{b}_vega'], dtype=np.float64)
            d = ab - vega
            d = d[np.isfinite(d)]
            applied = np.round(d, 4)
            uniq = np.unique(applied)
            if b in ('FUV', 'NUV'):
                # GALEX is native AB; the _vega column is the derived one, so
                # the difference is the AB-to-Vega offset, not a conversion.
                print(f'  {b:>3s}: native AB, Table 2 lists 0.0000; the derived '
                      f'Vega column sits at {uniq[0]:+.4f}')
                continue
            state = 'OK' if abs(uniq[0] - paper[b]) < 5e-4 else 'MISMATCH'
            print(f'  {b:>3s}: applied {uniq[0]:+.4f}  paper {paper[b]:+.4f}  '
                  f'{state}  ({len(uniq)} distinct value)')

    print('\n--- colour shifts quoted in the text ---')
    print(f'  (BP-RP) {paper["BP"]-paper["RP"]:+.4f} [-0.34]   '
          f'(G-W1) {paper["G"]-paper["W1"]:+.4f} [-2.56]')
    print(f'  (NUV-G) {-paper["G"]:+.4f} [-0.11]   '
          f'(J-W1) {paper["J"]-paper["W1"]:+.4f} [-1.78]   '
          f'M_G {paper["G"]:+.4f} [+0.11]')


# ---------------------------------------------------------------------------
# pivot
# ---------------------------------------------------------------------------
def section_pivot():
    head('TABLE 1 PIVOT WAVELENGTHS, AGAINST THE SVO FILTER PROFILE SERVICE')
    from astropy.io.votable import parse
    ids = [('FUV', 'GALEX/GALEX.FUV'), ('NUV', 'GALEX/GALEX.NUV'),
           ('BP', 'GAIA/GAIA3.Gbp'), ('G', 'GAIA/GAIA3.G'),
           ('RP', 'GAIA/GAIA3.Grp'), ('J', '2MASS/2MASS.J'),
           ('H', '2MASS/2MASS.H'), ('Ks', '2MASS/2MASS.Ks'),
           ('W1', 'WISE/WISE.W1'), ('W2', 'WISE/WISE.W2'),
           ('W3', 'WISE/WISE.W3'), ('W4', 'WISE/WISE.W4')]
    os.makedirs(CACHE, exist_ok=True)
    print(f'{"band":>4s} {"SVO pivot":>12s} {"paper (A)":>12s} {"diff":>8s} '
          f'{"SVO lam_eff":>12s} {"points":>7s}')
    for band, fid in ids:
        path = os.path.join(CACHE, f'{band}.xml')
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                f'http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={fid}',
                path)
        tab = parse(path).get_first_table()
        params = {p.name: p.value for p in tab.params}
        svo = float(params['WavelengthPivot'])
        eff = float(params.get('WavelengthEff', np.nan))
        paper = LAM[I[band]] * 1e4
        print(f'{band:>4s} {svo:12.2f} {paper:12.2f} {svo-paper:+8.2f} '
              f'{eff:12.2f} {len(tab.array):7d}')
    print('  [Sect. 5.1 also claims lambda_eff differs from the mean by almost')
    print('   900 A for Gaia G; the SVO mean is 6720 A against lam_eff 5822 A]')


# ---------------------------------------------------------------------------
# shared CMD machinery
# ---------------------------------------------------------------------------
def build_cmd():
    """Region assignment, normalised SEDs and region references."""
    n, _, mag, err, o = load_photometry(extra=('distance', 'outlier_flag'))
    dist = o['distance'].astype(np.float64)
    flag = o['outlier_flag'].astype(np.int32)
    ok = (np.isfinite(dist) & (dist > 0) & np.isfinite(mag[:, I['G']])
          & np.isfinite(mag[:, I['BP']]) & np.isfinite(mag[:, I['RP']]))
    ok_idx = np.where(ok)[0]
    mg = mag[ok, I['G']] - 5.0 * np.log10(dist[ok]) + 5.0
    bprp = mag[ok, I['BP']] - mag[ok, I['RP']]

    region_id = np.full(int(ok.sum()), -1, dtype=np.int8)
    for i, (_, (blo, bhi), (mlo, mhi)) in enumerate(REGIONS):
        box = (bprp >= blo) & (bprp < bhi) & (mg >= mlo) & (mg < mhi)
        assert not (region_id[box] >= 0).any(), 'region boxes overlap'
        region_id[box] = i
    sel = np.where(region_id >= 0)[0]
    return dict(n=n, mag=mag, err=err, dist=dist, flag=flag, ok=ok,
                ok_idx=ok_idx, region_id=region_id, sel=sel,
                reg_sel=region_id[sel], mags_sel=mag[ok_idx[sel]],
                errs_sel=err[ok_idx[sel]])


def normalised_sed(mags, errs, require_error=True):
    mask = np.isfinite(mags) & (np.isfinite(errs) if require_error else True)
    with np.errstate(invalid='ignore', divide='ignore'):
        fnu = np.where(mask, 10.0 ** (-0.4 * mags), np.nan)
        out = (fnu / fnu[:, I['G']][:, None]) * (LAM[I['G']] / LAM[None, :])
    out[~np.isfinite(fnu[:, I['G']]), :] = np.nan
    return out


def region_reference(lflf, reg_sel):
    med = np.full((NR, NB), np.nan)
    sig = np.full((NR, NB), np.nan)
    dfr = np.zeros((NR, NB))
    for i in range(NR):
        d = lflf[reg_sel == i]
        for j in range(NB):
            f = d[:, j][np.isfinite(d[:, j])]
            dfr[i, j] = len(f) / len(d) if len(d) else 0.0
            if len(f) >= 10:
                med[i, j] = np.median(f)
                p16, p84 = np.percentile(f, [16, 84])
                sig[i, j] = max((p84 - p16) / 2.0, 1e-6)
    return med, sig, dfr


# ---------------------------------------------------------------------------
# cmd
# ---------------------------------------------------------------------------
def section_cmd():
    head('CMD REGIONS, OUTLIER FLAG, AVERAGE SEDs, APPENDIX A')
    c = build_cmd()
    mag, err, flag, n = c['mag'], c['err'], c['flag'], c['n']
    det = np.isfinite(mag) & np.isfinite(err)

    print(f'  CMD sources (finite distance, G, BP, RP): {int(c["ok"].sum()):,} '
          f'[paper 9,687,001]')
    ccr = (np.isfinite(mag[:, I['BP']]) & np.isfinite(mag[:, I['RP']])
           & det[:, I['J']] & det[:, I['W1']])
    print(f'  right colour-colour panel (BP, RP, J, W1): {int(ccr.sum()):,} '
          f'[paper 9,647,823]')

    print('\n--- Table 3 ---')
    quoted = [4577, 53399, 196426, 1762321, 566446, 16004, 113605, 202584, 867]
    for i, (lab, _, _) in enumerate(REGIONS):
        k = int((c['reg_sel'] == i).sum())
        print(f'  {lab:<4s} {k:>10,}  [paper {quoted[i]:,}]'
              f'{"" if k == quoted[i] else "   DIFF"}')
    n_reg = len(c['sel'])
    print(f'  total {n_reg:,} [paper 2,916,229]; outside all boxes '
          f'{n - n_reg:,} [paper 6,789,650]')

    print('\n--- Table 4 ---')
    nz = flag != 0
    print(f'  any flag {int(nz.sum()):,} = {100*nz.sum()/n_reg:.2f} per cent '
          f'[paper 232,545 / 7.97]')
    labels = ['FUV excess', 'NUV excess', 'FUV deficit', 'NUV deficit',
              'W1/W2 excess', 'W3/W4 excess', 'W1/W2 deficit',
              'W3/W4 deficit', 'Shape outlier']
    want = [12141, 167013, 0, 0, 48820, 14870, 2137, 467, 32731]
    for b, lab in enumerate(labels):
        k = int(((flag >> b) & 1).sum())
        print(f'  bit {b} {lab:<14s} {k:>8,} {100*k/n_reg:5.2f} per cent '
              f'[paper {want[b]:,}]{"" if k == want[b] else "   DIFF"}')
    b5 = ((flag >> 5) & 1).astype(bool)
    faint = b5 & np.isfinite(mag[:, I['G']]) & (mag[:, I['G']] >= 16)
    print(f'  bit 5 with G>=16: {int(faint.sum()):,} = '
          f'{100*faint.sum()/b5.sum():.1f} per cent [paper 4,738 / 31.9]')

    lflf = normalised_sed(c['mags_sel'], c['errs_sel'])
    med, sigma, detfrac = region_reference(lflf, c['reg_sel'])

    print('\n--- Sect. 5: measured fraction per region and band (per cent) ---')
    print('  region ' + ''.join(f'{b:>7s}' for b in NAMES))
    for i, (lab, _, _) in enumerate(REGIONS):
        print(f'  {lab:<6s} ' + ''.join(f'{100*f:7.1f}' for f in detfrac[i]))
    print('  [paper: W3 measured for 98.1 (RGB) and 96.2 (RC) per cent, 25.8')
    print('   to 49.8 on the main sequence, 5 to 98 across all nine regions;')
    print('   W4 never above 49; WD J/H/Ks 17.2 / 10.3 / 5.0; FUV 26.7 (M),')
    print('   1.2 (G), 1.4 (K); bands below 50 per cent are drawn as limits]')

    print('\n--- Sect. 5.2: the M-dwarf NUV point ---')
    vals = {}
    for lab in ('G', 'K', 'M'):
        i = [r[0] for r in REGIONS].index(lab)
        v = lflf[c['reg_sel'] == i, I['NUV']]
        v = v[np.isfinite(v)]
        p16, p50, p84 = np.percentile(v, [16, 50, 84])
        vals[lab] = p50
        print(f'  {lab} region: median NUV/G {p50:.3f}, 16/84 spread '
              f'{np.log10(p84/p16):.2f} dex')
    print(f'  M/K ratio {vals["M"]/vals["K"]:.1f} [paper 14]; medians quoted '
          f'0.161, 0.029, 0.409 and 1.41 dex')

    print('\n--- Sect. 5.1: W4 median with and without the censored values ---')
    lflf_all = normalised_sed(c['mags_sel'], c['errs_sel'], require_error=False)
    for i, (lab, _, _) in enumerate(REGIONS):
        m = c['reg_sel'] == i
        a = lflf[m, I['W4']]
        b = lflf_all[m, I['W4']]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) > 10 and len(b) > 10:
            print(f'  {lab:<4s} detections {np.median(a):9.4f}  all tabulated '
                  f'{np.median(b):9.4f}  ratio {np.median(b)/np.median(a):5.2f}')
    print('  [paper: factors of 2 to 20 in the main-sequence regions]')

    print('\n--- Appendix A: the faint W3 detections ---')
    G, W3 = mag[:, I['G']], mag[:, I['W3']]
    d3 = det[:, I['W3']]
    lim3 = np.isfinite(W3) & ~d3
    faint = np.isfinite(G) & (G >= 18)
    print(f'  G>=18 with a W3 measurement: {int((faint & d3).sum()):,} '
          f'[paper 1,713]')
    print(f'    median W3 {np.median(W3[faint & d3]):.2f} [17.51] against '
          f'{np.median(W3[faint & lim3]):.2f} [17.72] for the censored limits')
    mid = np.isfinite(G) & (G >= 12) & (G < 14) & d3
    col = np.median(G[mid] - W3[mid])
    print(f'  median (G-W3) at G=12-14: {col:.2f} [-3.40], so the locus places')
    print(f'    G=18-21.05 at W3 = {18-col:.1f} to {G[np.isfinite(G)].max()-col:.1f} '
          f'[paper 21.4 to 24.4]')
    print(f'    G=16-18 at W3 = {16-col:.1f} to {18-col:.1f} [paper 19.4 to 21.4]')
    print(f'    G=14-16 at W3 = {14-col:.1f} to {16-col:.1f} [paper "17.5 '
          f'against a floor of 17.5"]')
    for lo, hi, lab in ((14, 16, '14-16'), (16, 18, '16-18'), (18, 99, '>=18')):
        m = np.isfinite(G) & (G >= lo) & (G < hi)
        print(f'    censored median W3 at G={lab}: '
              f'{np.median(W3[m & lim3]):.2f}')
    with fits.open(PUBLIC, memmap=True) as h:
        qph = np.array(h[1].data['qph']).astype(str)
        ra = np.array(h[1].data['ra_2016'], dtype=np.float64)
        dec = np.array(h[1].data['dec_2016'], dtype=np.float64)
    q3 = np.array([s[2:3] if len(s) >= 3 else '' for s in qph])
    sel1 = faint & d3
    sel2 = np.isfinite(G) & (G >= 16) & (G < 18) & d3
    print(f'  ph_qual C: {100*(q3[sel1] == "C").mean():.1f} per cent at G>=18 '
          f'[55.5]; {100*(q3[sel2] == "C").mean():.1f} per cent of the '
          f'{int(sel2.sum()):,} at G=16-18 [78.2 / 108,770]')
    b = np.abs(SkyCoord(ra=ra*u.deg, dec=dec*u.deg).galactic.b.deg)
    print(f'  median |b|: G>=18 with W3 {np.median(b[sel1]):.0f} deg, '
          f'G=16-18 with W3 {np.median(b[sel2]):.0f} deg, catalogue '
          f'{np.median(b):.0f} deg')
    print('  [the 35 deg of the paper is the G>=18 subset only]')


# ---------------------------------------------------------------------------
# flagdiag
# ---------------------------------------------------------------------------
def section_flagdiag():
    head('SECT. 6.1 DIAGNOSTICS (and a bit-for-bit check of outlier_flag)')
    c = build_cmd()
    lflf = normalised_sed(c['mags_sel'], c['errs_sel'])
    med, sigma, detfrac = region_reference(lflf, c['reg_sel'])
    is_limit = detfrac < DETFRAC_MIN
    reg = c['reg_sel']

    with np.errstate(invalid='ignore', divide='ignore'):
        res = (lflf - med[reg]) / sigma[reg]
        dev = 0.4 * np.log(10.0) * np.sqrt(
            c['errs_sel'] ** 2 + c['errs_sel'][:, I['G'], None] ** 2)
        sig_src = np.abs(lflf - med[reg]) / (np.abs(lflf) * dev)
        significant = sig_src >= SIG_SOURCE
        hi = (res > N_SIGMA) & significant
        lo = (res < -N_SIGMA) & significant & ~is_limit[reg]

    flags = np.zeros(len(c['sel']), dtype=np.uint16)
    for cond, bit in ((hi[:, I['FUV']], 1), (hi[:, I['NUV']], 2),
                      (lo[:, I['FUV']], 4), (lo[:, I['NUV']], 8),
                      (hi[:, I['W1']] & hi[:, I['W2']], 16),
                      (hi[:, I['W3']] | hi[:, I['W4']], 32),
                      (lo[:, I['W1']] & lo[:, I['W2']], 64),
                      (lo[:, I['W3']] | lo[:, I['W4']], 128),
                      (np.sum(hi | lo, axis=1) >= 4, 256)):
        flags[cond] |= np.uint16(bit)
    rebuilt = np.zeros(c['n'], dtype=np.uint16)
    rebuilt[c['ok_idx'][c['sel']]] = flags
    print(f'  outlier_flag reproduced from the released catalogue: '
          f'{np.array_equal(rebuilt, c["flag"].astype(np.uint16))}')

    n_reg = len(c['sel'])
    print('\n--- the W3/W4 excess bit under three criteria ---')
    raw = (res[:, I['W3']] > N_SIGMA) | (res[:, I['W4']] > N_SIGMA)
    lflf_all = normalised_sed(c['mags_sel'], c['errs_sel'], require_error=False)
    med2, sig2, _ = region_reference(lflf_all, reg)
    with np.errstate(invalid='ignore', divide='ignore'):
        res2 = (lflf_all - med2[reg]) / sig2[reg]
    raw2 = (res2[:, I['W3']] > N_SIGMA) | (res2[:, I['W4']] > N_SIGMA)
    print(f'  magnitude alone        {int(np.nansum(raw2)):>8,} = '
          f'{100*np.nansum(raw2)/n_reg:.2f} per cent [paper 6.5]')
    print(f'  finite uncertainty     {int(np.nansum(raw)):>8,} = '
          f'{100*np.nansum(raw)/n_reg:.2f} per cent [paper 3.5]')
    print(f'  + source significance  {int((flags & 32).astype(bool).sum()):>8,} = '
          f'{100*(flags & 32).astype(bool).sum()/n_reg:.2f} per cent [paper 0.51]')

    print('\n--- S/N of the raw excesses, and what significance keeps ---')
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = 1.0857 / c['errs_sel']
    for b in ('FUV', 'NUV', 'W1', 'W2', 'W3', 'W4'):
        j = I[b]
        r = res[:, j] > N_SIGMA
        k = int(np.nansum(r))
        if not k:
            continue
        print(f'  {b:>3s} excess: raw {k:>8,}  S/N<5 '
              f'{100*np.nansum(snr[r, j] < 5)/k:5.1f} per cent  kept '
              f'{100*hi[:, j].sum()/k:5.1f} per cent')
    print('  [paper: S/N<5 for 89.5 (W3), 95.1 (W4), 0.3 (W1), 0.2 (W2)')
    print('   per cent; kept 99 (W1), 88 (NUV), 16 (W3), 10 (W4); and "a')
    print('   fifth" of the raw NUV excesses would be lost to a flat S/N cut]')

    print('\n--- where W3/W4 deficits are possible, and where they occur ---')
    for i, (lab, _, _) in enumerate(REGIONS):
        m = reg == i
        k = int((lo[m, I['W3']] | lo[m, I['W4']]).sum())
        print(f'  {lab:<4s} W3 measured {100*detfrac[i, I["W3"]]:5.1f} per cent, '
              f'deficit bit {"permitted" if not is_limit[i, I["W3"]] else "suppressed":>10s}'
              f', deficits {k:,}')
    print('  [paper: all 467 fall in the red clump; the RGB also permits them')
    print('   and produces none]')

    print('\n--- what 3 sigma means as a colour ---')
    for lab in ('RC', 'RGB'):
        i = [r[0] for r in REGIONS].index(lab)
        m, s = med[i, I['W3']], sigma[i, I['W3']]
        up = 2.5 * np.log10((m + 3 * s) / m)
        dn = 2.5 * np.log10(m / (m - 3 * s)) if m > 3 * s else np.nan
        print(f'  {lab}: +3 sigma in W3 = {up:.2f} mag in (G-W3), -3 sigma = '
              f'{dn:.2f} mag [paper quotes 0.55 for the red clump]')

    print('\n--- the Sect. 6.3 example selection ---')
    b4 = (flags & 16).astype(bool)
    clean = np.ones(len(flags), dtype=bool)
    for band in ('J', 'H', 'Ks'):
        clean &= np.abs(res[:, I[band]]) <= 1.0
    no_uv = (flags & 15) == 0
    print(f'  W1/W2 excess {int(b4.sum()):,} [48,820]; of those, clean in '
          f'JHKs and free of UV flags: {int((b4 & clean & no_uv).sum()):,} [3,501]')
    raw_bit = np.where(np.isnan(res[:, I['W1']]) | np.isnan(res[:, I['W2']]),
                       False, (res[:, I['W1']] > N_SIGMA) & (res[:, I['W2']] > N_SIGMA))
    print(f'  W1/W2 excess bit keeps {100*b4.sum()/raw_bit.sum():.1f} per cent '
          f'of its raw flags [paper "lose only 1 per cent"]')


# ---------------------------------------------------------------------------
# external
# ---------------------------------------------------------------------------
def section_external():
    head('SIMBAD CENSUS AND THE GAIA-ESO SAMPLE')
    with fits.open(PUBLIC, memmap=True) as h:
        sid = np.array(h[1].data['source_id'], dtype=np.int64)
        flag = np.array(h[1].data['outlier_flag'], dtype=np.int32)
    n_cat = len(sid)
    with fits.open(SIMBAD, memmap=True) as h:
        ssid = np.array(h[1].data['pasta_source_id'], dtype=np.int64)
        otype = np.array([str(x).strip() for x in h[1].data['otype']])
    n_match = len(ssid)
    print(f'  matches {n_match:,} [957,331] = {100*n_match/n_cat:.2f} per cent '
          f'[9.86]; unique ids {len(np.unique(ssid)):,}; orphans '
          f'{int((~np.isin(ssid, sid)).sum())}')

    paper = [('*', 802704), ('EB*', 42550), ('SB*', 22975), ('Pe*', 14760),
             ('PM*', 13561), ('RG*', 8814), ('HB*', 6059), ('Em*', 5513),
             ('RR*', 4595), ('V*', 4581), ('RS*', 3093), ('Pu*', 2948),
             ('**', 2531), ('LP*', 2239), ('BY*', 1713), ('WD*', 1600),
             ('dS*', 1427), ('gD*', 1343), ('G', 1342), ('Er*', 498),
             ('HV*', 388), ('CV*', 380)]
    types, counts = np.unique(otype, return_counts=True)
    cnt = dict(zip(types, counts))
    print('\n--- Table C.1 ---')
    for k, want in paper:
        got = int(cnt.get(k, 0))
        print(f'  {k:>4s} {got:>8,} {100*got/n_match:5.2f} per cent '
              f'[paper {want:,}]{"" if got == want else "   DIFF"}')
    hs = int(cnt.get('HS*', 0)) + int(cnt.get('HS?', 0))
    print(f'  HS*+HS? {hs:,} = {100*hs/n_match:.2f} per cent [2,065 / 0.22]')
    print(f'  RS*+BY* {int(cnt.get("RS*", 0)) + int(cnt.get("BY*", 0)):,} '
          f'[paper "about 4,800"]')
    missing = [(k, int(cnt[k])) for k in types
               if cnt[k] >= 300 and k not in dict(paper)
               and k not in ('HS*', 'HS?')]
    print('  types with N>=300 absent from the table (all candidates, which')
    print('  the caption now says are excluded): '
          + ', '.join(f'{k} {v:,}' for k, v in sorted(missing, key=lambda x: -x[1])))

    order = np.argsort(sid)
    sflag = flag[order[np.searchsorted(sid, ssid, sorter=order)]]
    flagged = sflag != 0
    pe = otype == 'Pe*'
    print(f'\n  matched and flagged {int(flagged.sum()):,} [23,505]; Pe* among '
          f'them {int((pe & flagged).sum()):,} [4,098] = '
          f'{100*(pe & flagged).sum()/flagged.sum():.1f} per cent [17.4]')
    print(f'  Pe* overall {100*pe.sum()/n_match:.2f} per cent [1.54]; '
          f'enrichment {((pe & flagged).sum()/flagged.sum())/(pe.sum()/n_match):.1f} '
          f'[paper "elevenfold"]')
    gal = otype == 'G'
    print(f'  galaxies {int(gal.sum()):,} = {100*gal.sum()/n_match:.2f} per cent '
          f'of matched [1,342 / 0.14]')

    print('\n--- Gaia-ESO ---')
    with fits.open(GES, memmap=True) as h:
        g = h[1].data
        gw1 = np.array(g['pasta_W1'], dtype=np.float64)
        obj = np.array([str(x).strip() for x in g['ges_object']])
        gra = np.array(g['ges_ra'], dtype=np.float64)
        gdec = np.array(g['ges_dec'], dtype=np.float64)
        gteff = np.array(g['ges_teff'], dtype=np.float64)
    print(f'  in common {len(gw1):,} [6,409]; with W1<8 {int((gw1 < 8).sum())} '
          f'[28]; with a GES Teff among those {int(np.isfinite(gteff[gw1 < 8]).sum())} [26]')

    rows = list(csv.DictReader(open(VOSA_CSV)))
    groups = {k: sum(r[k] == 'True' for r in rows)
              for k in ('is_cha', 'is_m67', 'is_field', 'is_free_av')}
    print(f'  groups {groups} [Cha 10, M67 3, field 15, free Av 7]')
    field = [r for r in rows if r['is_field'] == 'True']
    good = [r for r in field if r['good_fit'] == 'True']
    print(f'  field {len(field)} [15], good fits {len(good)} [14]; the '
          f'fifteenth has Vgfb = '
          f'{float([r for r in field if r not in good][0]["Vgfb"]):.0f} [29]')
    for key, lab, want in (('teff', 'Teff', '+327 K, sigma 451, rms 657'),
                           ('logg', 'logg', '+0.4, sigma 1.3'),
                           ('feh', '[Fe/H]', '-0.6, sigma 0.5')):
        d = np.array([float(r['vosa_' + key]) - float(r['ges_' + key])
                      for r in good])
        print(f'  d{lab:<7s} median {np.median(d):+8.2f}  sigma {d.std():7.2f}  '
              f'rms {np.sqrt((d**2).mean()):7.2f}   [paper {want}]')
    gt = np.array([float(r['ges_teff']) for r in good])
    print(f'  GES Teff range of the good field stars {gt.min():.0f}-{gt.max():.0f} '
          f'[paper 3,900-5,800]')
    cha = [r['ges_object'] for r in rows if r['is_cha'] == 'True']
    m = np.isin(obj, cha)
    print(f'  Chamaeleon: {int(m.sum())} sources spanning RA '
          f'{gra[m].min():.2f}-{gra[m].max():.2f}, Dec {gdec[m].max():.2f} to '
          f'{gdec[m].min():.2f} [paper RA 164-169, Dec -75 to -78]')
    vgfb = sorted(float(r['Vgfb']) for r in rows if r['is_cha'] == 'True')
    print(f'  Cha Vgfb: {[round(v, 1) for v in vgfb]} '
          f'[paper: seven above 15, the other three at 2.8, 9.8, 10.4]')
    m67 = [r for r in rows if r['is_m67'] == 'True']
    print('  M67: ' + '; '.join(f'Av={r["vosa_Av"]}, VOSA Teff={r["vosa_teff"]}, '
                                f'GES={r["ges_teff"]}, Vgfb={float(r["Vgfb"]):.0f}'
                                for r in m67))
    print('  [paper: two at Av=1.40 and 6,250 K against about 4,590 K, an')
    print('   overestimate of about 1,660 K; the third at Av=0.10, Vgfb=32]')


def section_epoch():
    """The Sect. 3.1 bound on the epoch approximation.

    The cross-match carried the GALEX and AllWISE positions to the Gaia epoch
    with dt = 16 yr.  That is an approximation; this quantifies the effect of
    a residual error `delta` in the baseline, which displaces a search
    position by mu * delta.

    Two things this deliberately does NOT do, matching the paper:

      - it does not solve for the effective epoch of either survey.  The
        matched sample was selected by pairing under the very assumption
        being tested, so a counterpart displaced beyond the search radius was
        never admitted and the surviving distribution is censored exactly
        where the answer lies.  (fit_survey_epoch.py tried; it fails its own
        null test.)
      - it puts no number on the resulting incompleteness, which is not
        measurable from the matched table for the same reason.

    The second half needs pasta1_internal.fits, which carries both the native
    and the propagated survey coordinates, and takes a couple of minutes.
    """
    head('EPOCH APPROXIMATION (Sect. 3.1)')

    with fits.open(PUBLIC, memmap=True) as h:
        d = h[1].data
        mu = np.hypot(np.asarray(d['PMRA'], np.float64),
                      np.asarray(d['PMDEC'], np.float64))
    mu = mu[np.isfinite(mu)]

    med, p95 = np.median(mu), np.percentile(mu, 95)
    print(f'  median total proper motion {med:.1f} mas/yr [paper 7.5]')
    print(f'  95th percentile            {p95:.1f} mas/yr [paper 24.6]')

    delta = 10.5                      # yr, the value the paper evaluates
    disp = mu * delta / 1000.0        # mas -> arcsec
    over = 100.0 * (disp > MATCH_RADIUS).sum() / len(mu)
    print(f'  for delta = {delta} yr:')
    print(f'    displacement, median     {np.median(disp):.3f}" [paper 0.08]')
    print(f'    displacement, 95th pct   {np.percentile(disp, 95):.3f}" [paper 0.26]')
    print(f'    beyond the {MATCH_RADIUS}" radius   {over:.4f} per cent '
          f'[paper "fewer than 0.03"]')

    # The residual trend with proper motion: flat if the baseline were exact.
    if not os.path.exists(INTERNAL):
        print(f'  ({INTERNAL} absent; skipping the separation residual)')
        return

    def sep_arcsec(ra1, dec1, ra2, dec2):
        cd = np.cos(np.radians(0.5 * (dec1 + dec2)))
        dra = (ra1 - ra2 + 180.0) % 360.0 - 180.0
        return np.hypot(dra * cd, dec1 - dec2) * 3600.0

    print(f'  reading {INTERNAL} for the separation residual ...')
    with fits.open(INTERNAL, memmap=True) as h:
        d = h[1].data
        g = lambda c: np.asarray(d[c], dtype=np.float64)
        pm = np.hypot(g('PMRA'), g('PMDEC'))
        ra16, dec16 = g('ra_gaia_2016'), g('dec_gaia_2016')
        surveys = {s: (g(f'ra_{s}_2016'), g(f'dec_{s}_2016'),
                       g(f'ra_{s}_2000'), g(f'dec_{s}_2000'))
                   for s in ('galex', 'allwise')}

    quoted = {'allwise': '[paper 0.09 low PM -> 0.55 high PM; 0.12 -> 2.18 native]',
              'galex': '[paper: no trend discernible]'}
    for s, (r6, d6, r0, d0) in surveys.items():
        ok = np.isfinite(r6) & np.isfinite(pm)
        prop = sep_arcsec(ra16[ok], dec16[ok], r6[ok], d6[ok])
        nat = sep_arcsec(ra16[ok], dec16[ok], r0[ok], d0[ok])
        p = pm[ok]
        lo_m, hi_m = p < 5.0, p >= 100.0
        print(f'  Gaia-{s.upper()}  {quoted[s]}')
        print(f'    mu < 5 mas/yr    propagated {np.median(prop[lo_m]):.3f}"  '
              f'native {np.median(nat[lo_m]):.3f}"')
        print(f'    mu > 100 mas/yr  propagated {np.median(prop[hi_m]):.3f}"  '
              f'native {np.median(nat[hi_m]):.3f}"')


SECTIONS = {'counts': section_counts, 'sky': section_sky, 'match': section_match,
            'epoch': section_epoch, 'pivot': section_pivot, 'cmd': section_cmd,
            'flagdiag': section_flagdiag, 'external': section_external}

if __name__ == '__main__':
    wanted = sys.argv[1:] or list(SECTIONS)
    unknown = [w for w in wanted if w not in SECTIONS]
    if unknown:
        sys.exit(f'unknown section(s): {unknown}; choose from {list(SECTIONS)}')
    for name in wanted:
        SECTIONS[name]()
    print('\ndone.')
