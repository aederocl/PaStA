"""
Derive the publishable PaStA Paper I table from pasta1_internal.fits.

Publishable table keeps only:
  - a single coordinate pair, at the Gaia reference epoch (J2016.0) --
    the GALEX/AllWISE-specific epoch flavours were for internal
    crossmatch QC only and are dropped here.
  - magnitudes in the AB system only, for every band (AB is the
    physical system that maps directly onto Jy via F_nu = 3631 * 10^(-0.4*AB)).
  - parallax, proper motions, ruwe, distance, quality/variability flags,
    and the outlier flag.

No atmospheric parameters (Teff, logg, [Fe/H], extinction, AP_origin):
those live in Morgan's Paper II table, keyed on source_id + coordinates.

Run build_pasta1_internal.py first to produce pasta1_internal.fits.

Metadata
--------
The released file is self-describing: every column carries a unit and a
description, and the primary header records the provenance, the epoch, the
photometric system, the selection criteria and the null conventions. This
matters because the whole point of the released table is that it is
harmonised, and a bare FITS file does not say so anywhere -- a reader who
downloads it without the paper beside them would have no way to know that
the magnitudes are AB rather than the native systems, or that a W3 or W4
magnitude with a null uncertainty is an upper limit rather than a
measurement.

Two modes:

    python make_pasta1_public.py
        Full derivation from pasta1_internal.fits.

    python make_pasta1_public.py --annotate
        Apply the metadata to an existing pasta1_public.fits without
        re-deriving it. Header-only: the data are streamed through
        untouched, so this needs no more memory than a file copy. Used to
        annotate the released file in place rather than rebuilding a 3.5 GB
        intermediate.

The AllWISE flag descriptions below are taken verbatim from the VizieR
ReadMe of II/328 (allwise), not from memory.
"""
import os
import shutil
import sys
import time

from astropy.io import fits
from astropy.io.fits.convenience import table_to_hdu
from astropy.table import Table

INPUT_FITS = "pasta1_internal.fits"
OUTPUT_FITS = "pasta1_public.fits"
EXTNAME = "PASTA"

AB_BANDS = ["FUV", "NUV", "G", "BP", "RP", "W1", "W2", "W3", "W4", "J", "H", "Ks"]

FLAG_COLS = [
    "phot_variable_flag", "non_single_star", "has_xp_sampled",
    "has_epoch_photometry", "var", "qpm", "qph", "fdet", "outlier_flag",
]

# Which survey each band comes from, for the column descriptions.
BAND_SURVEY = {
    "FUV": "GALEX AIS", "NUV": "GALEX AIS",
    "G": "Gaia DR3", "BP": "Gaia DR3", "RP": "Gaia DR3",
    "W1": "AllWISE", "W2": "AllWISE", "W3": "AllWISE", "W4": "AllWISE",
    "J": "2MASS", "H": "2MASS", "Ks": "2MASS",
}

# name -> (unit, description).  Bands are added programmatically below.
COLUMN_META = {
    "source_id": ("", "Gaia DR3 source identifier"),
    "ra_2016": ("deg", "Right ascension, ICRS, at epoch J2016.0"),
    "dec_2016": ("deg", "Declination, ICRS, at epoch J2016.0"),
    "ra_error": ("mas", "Standard error of right ascension"),
    "dec_error": ("mas", "Standard error of declination"),
    "parallax": ("mas", "Gaia DR3 absolute stellar parallax"),
    "parallax_error": ("mas", "Standard error of parallax"),
    "PMRA": ("mas/yr", "Proper motion in RA, mu_alpha* = mu_alpha cos(delta)"),
    "PMRA_ERROR": ("mas/yr", "Standard error of PMRA"),
    "PMDEC": ("mas/yr", "Proper motion in declination"),
    "PMDEC_ERROR": ("mas/yr", "Standard error of PMDEC"),
    "ruwe": ("", "Gaia DR3 renormalised unit weight error (selection: <1.4)"),
    "distance": ("pc", "Inverse-parallax distance, 1000/parallax; see Sect. 4.1"),
    "phot_variable_flag": ("", "Gaia DR3 photometric variability flag"),
    "non_single_star": ("", "Gaia DR3 non-single-star flag (1 astrom, 2 spec, 4 ecl)"),
    "has_xp_sampled": ("", "Gaia DR3 sampled BP/RP spectrum available"),
    "has_epoch_photometry": ("", "Gaia DR3 epoch photometry available"),
    "var": ("", "AllWISE variability flag, one char per band W1-W4 (var_flg)"),
    "qpm": ("", "AllWISE motion estimation quality, format NQDDD (pmcode)"),
    "qph": ("", "AllWISE photometric quality per band W1-W4; U = upper limit"),
    "fdet": ("", "AllWISE bands with w?snr>2, bit-encoded (bits 0-3 = W1-W4)"),
    "outlier_flag": ("", "PaStA photometric outlier flag, 9 bits; see Paper I Table 4"),
}

for _b in AB_BANDS:
    COLUMN_META[_b] = ("mag", "%s %s magnitude, AB system" % (BAND_SURVEY[_b], _b))
    COLUMN_META["e_" + _b] = ("mag", "Uncertainty on %s" % _b)


PROVENANCE = [
    "",
    "The Panchromatic Stellar Atlas (PaStA), Paper I release.",
    "",
    "A cross-match of Gaia DR3, the GALEX All-sky Imaging Survey and",
    "AllWISE (which carries the 2MASS J, H, Ks photometry), giving twelve",
    "bands from 0.15 to 22 micron for 9,705,879 sources over 26,785 deg^2",
    "(64.9 per cent of the sky).",
    "",
    "HARMONISATION -- the two things that make this table different from",
    "its parent catalogues, and that a bare FITS file would not otherwise",
    "record:",
    "",
    "  1. ONE EPOCH. A single position, at the Gaia DR3 reference epoch",
    "     J2016.0, in the ICRS. The survey-specific positions used during",
    "     the cross-match are not carried through.",
    "  2. ONE PHOTOMETRIC SYSTEM. All twelve bands are AB, so a single",
    "     zero point applies throughout:  F_nu = 3631 Jy * 10^(-0.4 m_AB).",
    "     The native systems (Vega for the eight non-GALEX bands) have",
    "     already been converted; see Paper I, Sect. 3.3 and Table 2.",
    "",
    "SELECTION. Gaia DR3 sources with parallax_over_error > 5 and",
    "ruwe < 1.4, having a GALEX AIS NUV detection. Cross-match radius",
    "0.97 arcsec, nearest neighbour.",
    "",
    "NULL CONVENTIONS -- read this before using W3 or W4.",
    "",
    "  A band counts as MEASURED if, and only if, its uncertainty is",
    "  finite. AllWISE reports a magnitude for undetected sources but sets",
    "  the uncertainty to null, and that magnitude is then a 95 per cent",
    "  confidence UPPER LIMIT, not a measurement (see qph: U). Treating",
    "  those values as detections produces a spurious mid-infrared excess.",
    "  Measured fractions: W3 40.2 per cent, W4 5.3 per cent. W1 and W2",
    "  are essentially complete. NUV is present for every source by",
    "  construction; FUV for only 4.1 per cent.",
    "",
    "  No signal-to-noise cut is applied on top of that, deliberately:",
    "  GALEX and Gaia never null an uncertainty, so cutting there would",
    "  bias the medians high.",
    "",
    "  214 sources have no G and no BP (Gaia astrometry-only sources).",
    "",
    "DISTANCE. The distance column is simply 1000/parallax. Its median is",
    "unbiased, but the scatter is asymmetric near the parallax_over_error",
    "= 5 limit (16th/84th percentiles at -15/+22 per cent). It is not a",
    "Bayesian distance estimate.",
    "",
    "EXTINCTION is NOT applied. Colours and magnitudes are as observed.",
    "Dereddened photometry and atmospheric parameters are the subject of",
    "Paper II (Camargo et al.).",
    "",
]


def build_primary_header():
    """Provenance header for the primary HDU."""
    hdr = fits.Header()
    hdr["ORIGIN"] = ("CEFCA / IAG-USP", "Institutions responsible")
    hdr["CREATOR"] = ("make_pasta1_public.py", "Script that wrote this file")
    hdr["DATE"] = (time.strftime("%Y-%m-%dT%H:%M:%S"), "File creation date (UTC offset local)")
    hdr["RADESYS"] = ("ICRS", "Astrometric reference frame")
    hdr["EPOCH"] = (2016.0, "Epoch of the tabulated positions (Julian yr)")
    hdr["MAGSYS"] = ("AB", "Photometric system of every magnitude column")
    hdr["NSOURCE"] = (9705879, "Number of sources")
    hdr["REFERENC"] = ("2026A&A...Ederoclite (PaStA I)", "Describing publication")
    hdr["CODEURL"] = ("https://github.com/aederocl/PaStA", "Code repository")
    hdr["CODEDOI"] = ("10.5281/zenodo.22110791", "Code archive (concept DOI)")
    for line in PROVENANCE:
        hdr.add_comment(line)
    return hdr


def apply_column_metadata(hdu):
    """Write TUNIT/TCOMM for every column of `hdu`, plus EXTNAME.

    Operates on the header only, so the data are never touched.
    """
    hdu.header["EXTNAME"] = EXTNAME
    missing = []
    for i, col in enumerate(hdu.columns, start=1):
        meta = COLUMN_META.get(col.name)
        if meta is None:
            missing.append(col.name)
            continue
        unit, description = meta
        if unit:
            hdu.header["TUNIT%d" % i] = (unit, "")
        hdu.header["TCOMM%d" % i] = (description, "")
    if missing:
        raise SystemExit("No metadata defined for columns: %s" % ", ".join(missing))
    return hdu


def annotate_existing(path=OUTPUT_FITS):
    """Add the metadata to an existing file, without re-deriving it."""
    if not os.path.exists(path):
        raise SystemExit("%s not found" % path)
    tmp = path + ".annotated.tmp"
    print("Annotating %s (header only; data streamed through) ..." % path)
    with fits.open(path, memmap=True) as hdul:
        hdu = apply_column_metadata(hdul[1])
        primary = fits.PrimaryHDU(header=build_primary_header())
        print("  writing %s ..." % tmp)
        fits.HDUList([primary, hdu]).writeto(tmp, overwrite=True)
    print("  replacing %s ..." % path)
    shutil.move(tmp, path)
    print("Done.")


def main():
    print("Reading %s ..." % INPUT_FITS)
    t = Table.read(INPUT_FITS)
    print("  {:,} rows".format(len(t)))

    out = Table()
    out["source_id"] = t["source_id"]
    out["ra_2016"] = t["ra_gaia_2016"]
    out["dec_2016"] = t["dec_gaia_2016"]
    out["ra_error"] = t["ra_error"]
    out["dec_error"] = t["dec_error"]

    out["parallax"] = t["parallax"]
    out["parallax_error"] = t["parallax_error"]
    out["PMRA"] = t["PMRA"]
    out["PMRA_ERROR"] = t["PMRA_ERROR"]
    out["PMDEC"] = t["PMDEC"]
    out["PMDEC_ERROR"] = t["PMDEC_ERROR"]
    out["ruwe"] = t["ruwe"]
    out["distance"] = t["distance"]

    for band in AB_BANDS:
        out[band] = t["%s_ab" % band]
        out["e_%s" % band] = t["e_%s" % band]

    for col in FLAG_COLS:
        out[col] = t[col]

    print("Writing %s (%d columns, all magnitudes AB) ..." % (OUTPUT_FITS, len(out.colnames)))
    hdu = apply_column_metadata(table_to_hdu(out))
    primary = fits.PrimaryHDU(header=build_primary_header())
    fits.HDUList([primary, hdu]).writeto(OUTPUT_FITS, overwrite=True)
    print("Done.")


if __name__ == "__main__":
    if "--annotate" in sys.argv[1:]:
        annotate_existing()
    else:
        main()
