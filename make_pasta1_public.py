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
"""
from astropy.table import Table

INPUT_FITS = "pasta1_internal.fits"
OUTPUT_FITS = "pasta1_public.fits"

AB_BANDS = ["FUV", "NUV", "G", "BP", "RP", "W1", "W2", "W3", "W4", "J", "H", "Ks"]

FLAG_COLS = [
    "phot_variable_flag", "non_single_star", "has_xp_sampled",
    "has_epoch_photometry", "var", "qpm", "qph", "fdet", "outlier_flag",
]


def main():
    print(f"Reading {INPUT_FITS} ...")
    t = Table.read(INPUT_FITS)
    print(f"  {len(t):,} rows")

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
        out[band] = t[f"{band}_ab"]
        out[f"e_{band}"] = t[f"e_{band}"]

    for col in FLAG_COLS:
        out[col] = t[col]

    print(f"Writing {OUTPUT_FITS} ({len(out.colnames)} columns, all magnitudes AB) ...")
    out.write(OUTPUT_FITS, overwrite=True)
    print("Done.")


if __name__ == "__main__":
    main()
