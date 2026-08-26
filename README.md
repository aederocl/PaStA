# PaStA — the Panchromatic Stellar Atlas

Software that builds and verifies the Panchromatic Stellar Atlas: a cross-match
of Gaia DR3, the GALEX All-sky Imaging Survey and AllWISE (with 2MASS), giving
9,705,879 sources with photometry in twelve bands from the far-ultraviolet at
0.15 µm to WISE W4 at 22 µm.

This repository accompanies

> Ederoclite, A., Camargo, M. S., Coelho, P. R. T., Teixeira, R.,
> *The Panchromatic Stellar Atlas (PaStA). I. Catalogue construction and
> verification*, Astronomy & Astrophysics (submitted).

Every figure, table and quoted number in that paper is produced by a script
here.

## What is in the released catalogue

`pasta1_public.fits` is harmonised in two respects that the parent catalogues
are not:

- **One position, one epoch.** A single coordinate pair at the Gaia DR3
  reference epoch J2016.0, in the ICRS. The GALEX and AllWISE positions used
  during the cross-match are not carried through.
- **One photometric system.** All twelve bands are on the AB system, so a
  single zero point applies throughout: F_nu = 3631 Jy × 10^(−0.4 m_AB). Every
  magnitude and colour is AB unless stated otherwise.

Two further properties matter when the photometry is used:

- **NUV is required of every source; FUV is present for only 4.1 per cent**
  (393,632 sources).
- **W3 and W4 are largely upper limits.** AllWISE tabulates a magnitude for
  undetected sources but sets the uncertainty to null, which makes that value a
  95 per cent confidence upper limit rather than a measurement. A band counts as
  measured here if, and only if, its uncertainty is finite: W3 is measured for
  40.2 per cent of sources and W4 for 5.3 per cent. No signal-to-noise cut is
  applied on top of that, deliberately — GALEX and Gaia never null an
  uncertainty, so cutting on signal-to-noise would bias the medians high.

## Getting the data

The catalogue itself is **not** in this repository; it is a 2.2 GB FITS file.
See the Data Availability section of the paper for the archive record and its
DOI. The scripts expect `pasta1_public.fits` (and, for a few of them,
`pasta.fits`, `pasta1_internal.fits`, `outlier_flag.npz`, `simbad_xmatch.fits`
and `ges_pasta_matched.fits`) in the working directory.

**Re-running the queries today will not reproduce 9,705,879 rows.** Gaia,
GALEX and AllWISE have all been re-released or re-reduced since the catalogue
was built, and the TAP services return what they hold now. The construction
scripts below are the record of how the catalogue was made; the catalogue
itself is the archived artefact.

## Installing

```sh
pip install -r requirements.txt
```

STILTS is also required for the construction stage (stages 0 and 1 below) and
must be on `$PATH`. It is not a Python package; see
<https://www.star.bris.ac.uk/~mbt/stilts/>.

Optional environment variables:

| variable | used by | default |
|---|---|---|
| `PASTA_FIGDIR` | every figure script | `fig/` beside the script |
| `DUSTMAPS_DATA_DIR` | `getcat_pasta.py` | whatever the local `dustmaps` config says |
| `GES_DR5_FITS` | `ges_crossmatch.py` | `Gaia_ESO_DR5.fits` in the working directory |

Run every script from the repository root.

## The pipeline

### Stage 0 — build the parent catalogue

The sky is queried in 5° × 5° tiles, because a whole-sky TAP query exceeds the
service limits.

```sh
python create_database.py      # sqlite bookkeeping of which tiles are done
python create_grid.py          # writes runPaStA.sh, one line per tile
sh runPaStA.sh                 # each line runs getcat_pasta.py on one tile
python combine_pasta.py        # STILTS tcat of the tiles -> pasta.fits
```

`getcat_pasta.py` performs the Gaia DR3 query, the positional cross-matches
against the GALEX AIS and AllWISE point-source catalogues (STILTS
`tapskymatch`, `find=best`, `sr = 0.00027 deg ≈ 0.97″`), and chains on
Gaia-ESO/APOGEE/LAMOST/SEGUE atmospheric parameters and CSFD/Edenhofer2023
extinction. Its docstring records four bugs fixed relative to the version that
originally built `pasta.fits`, and two methodological choices left deliberately
unchanged. Read it before re-running anything.

`insertLogToDB.py` is a one-off migration of logs from a pre-2021 version of
the tile builder. It is kept for the record and is not part of the pipeline.

### Stage 1 — harmonisation

```sh
python make_sed_regions.py        # -> sed_region_stats_ab.npz, region figures
python make_outlier_flag.py       # -> outlier_flag.npz, outlier_examples figure
python build_pasta1_internal.py   # -> pasta1_internal.fits (dual epoch, Vega+AB)
python make_pasta1_public.py      # -> pasta1_public.fits (J2016.0, AB)
```

**This stage is circular, and cannot be run once from top to bottom.**
`make_sed_regions.py` and `make_outlier_flag.py` both read
`pasta1_public.fits`, but `pasta1_public.fits` carries the `outlier_flag`
column that `make_outlier_flag.py` produces. The catalogue was therefore built
in two passes: `pasta1_internal.fits` and `pasta1_public.fits` were built
first without the flag, the region statistics and the flag were computed
against that table, and the two tables were then rebuilt with `outlier_flag`
merged in on `source_id` (not on row order). `build_pasta1_internal.py` as it
stands requires `outlier_flag.npz` to exist and will stop without it, so a
first pass needs that dependency removed by hand. This is a known rough edge
that is documented rather than hidden; the released tables are the second-pass
output.

### Stage 2 — figures

| script | figure |
|---|---|
| `make_sky_density.py` | Fig. 1, HEALPix order 6 source density, Mollweide |
| `make_filter_curves.py` | Fig. 2, the twelve transmission curves |
| `make_sed_regions.py` | CMD region boxes, and the 3×3 grid of average SEDs |
| `make_eda.py` | colour–colour diagrams, per-band magnitude histograms |
| `make_outlier_flag.py` | example SEDs of outlier-flagged sources |
| `make_cmd.py` | colour–magnitude density map |
| `make_moc_sky.py` | the Multi-Order Coverage map, and the 26,785 deg² (64.9 per cent) quoted for the sky coverage |

`make_moc_sky.py` is kept rather than folded into `make_sky_density.py`:
counting occupied pixels on the density map overestimates the covered area,
whereas the MOC does not.

`make_filter_curves.py` reads the twelve profiles cached in `filters/`, taken
from the SVO Filter Profile Service under the identifiers listed in Appendix B
of the paper. They are committed here so that the figure rebuilds offline and
the exact passbands travel with the code. Each band's pivot wavelength printed
by the script reproduces its row in Table 2.

### Stage 3 — external validation

```sh
python crossmatch_stats.py            # separations, per-band detection fractions
python make_simbad_xmatch.py          # CDS XMatch against SIMBAD, resumable
python analyse_simbad_xmatch.py       # object-type census, CMD by type
python ges_crossmatch.py              # Gaia-ESO DR5 -> ges_pasta_matched.fits
python make_vosa_input_gaia_av.py     # VOSA upload file, Av fixed from Gaia
python make_vosa_comparison.py        # VOSA fits against GES spectroscopy
```

`make_vosa_input.py` is the earlier variant with a free Av range;
`make_vosa_input_gaia_av.py` is the one used in the paper. The VOSA fits
themselves are run on the SVO service at
<https://svo2.cab.inta-csic.es/theory/vosa/> and its output directory is read
back by `make_vosa_comparison.py`.

### Stage 4 — verification

```sh
python verify_paper_numbers.py                      # everything, about two minutes
python verify_paper_numbers.py counts sky match     # or named sections
```

Sections: `counts sky match pivot cmd flagdiag external`. Each recomputes a
quoted number from the data products and prints it next to the value the paper
quotes. It exists because a number that was correct when written is not
necessarily correct after the catalogue changes; the sweep it was written for
found six such numbers.

The `external` section needs `ges_pasta_matched.fits`, which is not committed
here — regenerate it with `ges_crossmatch.py` from Gaia-ESO DR5.

## Small data products included

| file | what it is |
|---|---|
| `filters/*.xml` | the twelve SVO filter profiles, as VOTables |
| `sed_region_stats_ab.npz` | the average SEDs of the nine CMD regions, the reference against which the outlier flag is computed |
| `simbad_otype_counts.csv` | the SIMBAD object-type census |
| `vosa_ges_comparison.csv` | VOSA against Gaia-ESO parameters, 28 sources |
| `vosa_input_gaia_av.txt` | the VOSA upload file used in the paper |

Everything larger — the catalogue, the internal table, the outlier flag array
and the SIMBAD cross-match — is in the archive record, not in git.

## Notes on interpretation

- **`ruwe < 1.4` is not a binarity filter.** RUWE responds to photocentre
  motion, so it is selective by binary type and is nearly blind to short-period
  and equal-flux pairs. Of the 31,840 sources that Gaia's `non_single_star`
  marks non-single, exactly one is an astrometric binary, against 27,091
  spectroscopic and 4,648 eclipsing.
- **`distance` is 1/parallax**, at `parallax_over_error > 5`. Its *median* is
  unbiased, because 1/x is monotonic. Its *mean* is inflated by 3.8 per cent
  just above the cut. The effect that matters is the asymmetric scatter: the
  16th and 84th percentiles sit at −15 and +22 per cent at the limit, against
  ∓1 per cent above `parallax_over_error` of 40.
- **A finite W3 uncertainty is necessary but not sufficient** for the value to
  be photospheric. For G ≳ 16 the nominal W3 "detections" sit at the AllWISE
  sensitivity floor rather than on the stellar locus.
- **Extinction is not applied anywhere here.** The colour–magnitude diagrams
  are in observed colour. Dereddened photometry and atmospheric parameters are
  the subject of Paper II (Camargo et al.).

## Authors

- A. Ederoclite — Centro de Estudios de Física del Cosmos de Aragón (CEFCA),
  Teruel, Spain (corresponding)
- M. S. Camargo, P. R. T. Coelho, R. Teixeira — Instituto de Astronomia,
  Geofísica e Ciências Atmosféricas, Universidade de São Paulo, Brazil

## Licence

MIT. See `LICENSE`.
