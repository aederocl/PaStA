"""
Build the internal PaStA Paper I working table from pasta.fits + outlier_flag.npy.

Produces pasta1_internal.fits with:
  - Gaia, GALEX and AllWISE coordinates, each given at both the Gaia reference
    epoch (2016.0) and J2000.
  - every band in both Vega and AB magnitude systems.
  - parallax, proper motions, ruwe, distance, quality/variability flags,
    and the outlier flag merged in from outlier_flag.npy.

Atmospheric parameters (Teff, logg, [Fe/H], extinction, AP_origin) are
deliberately dropped: those belong to Morgan's Paper II table, not Paper I.

Epoch handling
--------------
Gaia's own ra/dec at J2016.0 are not stored in pasta.fits (only the
already-propagated raj2000/dej2000). They are recovered by exactly
inverting the linear transform used to build raj2000/dej2000
(ra_2016 = raj2000 + PMRA/(3.6e6)*16), so they reproduce Gaia's native
values bit-for-bit -- this inversion carries no new approximation error,
regardless of whether that original formula was itself a simplification.

GALEX and AllWISE have no proper motion of their own. Their catalogue
position (already in pasta.fits, labelled J2000 by VizieR convention) is
kept as the "2000" flavour, and a "2016" flavour is newly computed by
propagating that position forward with Gaia's proper motion, using the
standard cos(dec)-corrected formula (PMRA is mu_alpha* = mu_alpha*cos(dec),
so the RA offset must be divided by cos(dec) to get an angular RA
change). This differs from the simplified (non-cos-corrected) formula
used to build the existing raj2000/dej2000 columns; the difference is
negligible except at high declination / high proper motion, but is noted
here for transparency.

Magnitude systems
-----------------
Vega -> AB offsets (m_AB = m_Vega + offset), derived so as to be
consistent with the zero points already cited in Sect. 5 of the paper:
  - Gaia G/BP/RP: Riello et al. (2021, Gaia EDR3 Table 5.2) VEGAMAG/AB
    zero points directly (native Gaia mags are VEGAMAG, not exactly AB).
  - GALEX FUV/NUV: already AB in pasta.fits (Morrissey et al. 2007
    offsets baked in); Vega values recovered by inverting that offset.
  - WISE W1-W4 and 2MASS J/H/Ks: offset = -2.5*log10(F_vega_zp/3631 Jy),
    using the same Vega zero-point fluxes cited in Sect. 5
    (Wright et al. 2010 for WISE; Cohen et al. 2003 for 2MASS).
"""
import numpy as np
from astropy.table import Table

PASTA_FITS = "pasta.fits"
OUTLIER_FLAG_NPZ = "outlier_flag.npz"
OUTPUT_FITS = "pasta1_internal.fits"

DELTA_T = 16.0  # yr, 2016.0 - 2000.0
MAS_TO_DEG = 1.0 / (3600.0 * 1000.0)

# Vega -> AB offsets, m_AB = m_Vega + OFFSET
GAIA_AB_OFFSET = {"G": 0.1136, "BP": 0.0155, "RP": 0.3561}       # Riello+2021 EDR3 Table 5.2
GALEX_AB_OFFSET = {"FUV": -2.223, "NUV": -1.699}                  # Morrissey+2007 (already applied in pasta.fits)
VEGA_ZP_JY = {                                                    # as published, not rounded
    "W1": 309.540, "W2": 171.787, "W3": 31.674, "W4": 8.363,      # Wright et al. 2010, Table 1
    "J": 1594.0, "H": 1024.0, "Ks": 666.7,                        # Cohen et al. 2003, Table 2
}
F_AB_JY = 3631.0
OTHER_AB_OFFSET = {b: -2.5 * np.log10(f / F_AB_JY) for b, f in VEGA_ZP_JY.items()}


def mag_error_from_flux(flux, flux_error):
    return (2.5 / np.log(10.0)) * (flux_error / flux)


def propagate_forward(ra0, dec0, pmra_masyr, pmdec_masyr, dt=DELTA_T):
    """Propagate a J2000 position forward by dt years using Gaia's proper
    motion, with the standard cos(dec) correction for the RA component."""
    dra = (pmra_masyr * MAS_TO_DEG * dt) / np.cos(np.radians(dec0))
    ddec = pmdec_masyr * MAS_TO_DEG * dt
    return ra0 + dra, dec0 + ddec


def dedupe_by_source_id(t):
    """Drop exact-duplicate rows sharing a source_id, keeping the first
    occurrence. These arise from tile-boundary double-processing in the
    original per-tile crossmatch (adjacent RA/Dec tiles both inclusively
    capturing a source that sits exactly on their shared edge)."""
    sid = np.asarray(t["source_id"])
    _, first_pos = np.unique(sid, return_index=True)
    keep_mask = np.zeros(len(sid), dtype=bool)
    keep_mask[first_pos] = True
    n_dup = int((~keep_mask).sum())
    if n_dup:
        print(f"  removing {n_dup} duplicate rows ({len(np.unique(sid[~keep_mask]))} distinct "
              f"source_id, each duplicated once) -- tile-boundary double-processing")
    return t[keep_mask], keep_mask


def main():
    print(f"Reading {PASTA_FITS} ...")
    t = Table.read(PASTA_FITS)
    n_raw = len(t)
    print(f"  {n_raw:,} rows")

    t, keep_mask = dedupe_by_source_id(t)
    n = len(t)
    print(f"  {n:,} rows after deduplication")

    out = Table()
    out["source_id"] = t["source_id"]

    # --- Gaia coordinates: recover native 2016 by exact inversion, keep stored 2000 ---
    out["ra_gaia_2000"] = t["raj2000"]
    out["dec_gaia_2000"] = t["dej2000"]
    out["ra_gaia_2016"] = t["raj2000"] + t["PMRA"] * MAS_TO_DEG * DELTA_T
    out["dec_gaia_2016"] = t["dej2000"] + t["PMDEC"] * MAS_TO_DEG * DELTA_T
    out["ra_error"] = t["ra_error"]
    out["dec_error"] = t["dec_error"]

    # --- GALEX coordinates: native = 2000, propagate forward for 2016 ---
    out["ra_galex_2000"] = t["RAJ2000_galex"]
    out["dec_galex_2000"] = t["DEJ2000_galex"]
    ra16, dec16 = propagate_forward(
        t["RAJ2000_galex"].data.astype(np.float64),
        t["DEJ2000_galex"].data.astype(np.float64),
        t["PMRA"].data.astype(np.float64),
        t["PMDEC"].data.astype(np.float64),
    )
    out["ra_galex_2016"] = ra16
    out["dec_galex_2016"] = dec16

    # --- AllWISE coordinates: native = 2000, propagate forward for 2016 ---
    out["ra_allwise_2000"] = t["RAJ2000_allwise"]
    out["dec_allwise_2000"] = t["DEJ2000_allwise"]
    ra16, dec16 = propagate_forward(
        t["RAJ2000_allwise"].data.astype(np.float64),
        t["DEJ2000_allwise"].data.astype(np.float64),
        t["PMRA"].data.astype(np.float64),
        t["PMDEC"].data.astype(np.float64),
    )
    out["ra_allwise_2016"] = ra16
    out["dec_allwise_2016"] = dec16

    # --- astrometry ---
    out["parallax"] = t["parallax"]
    out["parallax_error"] = t["parallax_error"]
    out["PMRA"] = t["PMRA"]
    out["PMRA_ERROR"] = t["PMRA_ERROR"]
    out["PMDEC"] = t["PMDEC"]
    out["PMDEC_ERROR"] = t["PMDEC_ERROR"]
    out["ruwe"] = t["ruwe"]
    out["distance"] = t["distance"]

    # --- Gaia photometry: native Vega-like -> add AB ---
    for band, gcol in (("G", "g"), ("BP", "bp"), ("RP", "rp")):
        mag = t[f"phot_{gcol}_mean_mag"]
        flux = t[f"phot_{gcol}_mean_flux"]
        ferr = t[f"phot_{gcol}_mean_flux_error"]
        out[f"{band}_vega"] = mag
        out[f"{band}_ab"] = mag + GAIA_AB_OFFSET[band]
        out[f"e_{band}"] = mag_error_from_flux(flux, ferr)

    # --- GALEX photometry: native AB in pasta.fits -> recover Vega ---
    for band in ("FUV", "NUV"):
        ab = t[band.lower()]
        out[f"{band}_ab"] = ab
        out[f"{band}_vega"] = ab - GALEX_AB_OFFSET[band]
        out[f"e_{band}"] = t[f"e_{band}mag"]

    # --- WISE photometry: native Vega -> add AB ---
    for band in ("W1", "W2", "W3", "W4"):
        vega = t[f"{band}mag"]
        out[f"{band}_vega"] = vega
        out[f"{band}_ab"] = vega + OTHER_AB_OFFSET[band]
        out[f"e_{band}"] = t[f"e_{band}mag"]

    # --- 2MASS photometry (via AllWISE): native Vega -> add AB ---
    for band, col in (("J", "Jmag"), ("H", "Hmag"), ("Ks", "Kmag")):
        vega = t[col]
        out[f"{band}_vega"] = vega
        out[f"{band}_ab"] = vega + OTHER_AB_OFFSET[band]
        out[f"e_{band}"] = t[f"e_{col}"]

    # --- flags ---
    out["phot_variable_flag"] = t["phot_variable_flag"]
    out["non_single_star"] = t["non_single_star"]
    out["has_xp_sampled"] = t["has_xp_sampled"]
    out["has_epoch_photometry"] = t["has_epoch_photometry"]
    out["var"] = t["var"]
    out["qpm"] = t["qpm"]
    out["qph"] = t["qph"]
    out["fdet"] = t["fdet"]

    # The outlier flag is computed on the deduplicated, AB, single-epoch table
    # (make_outlier_flag.py reads pasta1_public.fits), so it is matched back on
    # source_id rather than by row position -- no assumption that the two files
    # share a row ordering.
    print(f"Loading {OUTLIER_FLAG_NPZ} ...")
    with np.load(OUTLIER_FLAG_NPZ) as npz:
        flag_sid, flag_val = npz["source_id"], npz["flag"]
    order = np.argsort(flag_sid)
    sid = np.asarray(out["source_id"])
    pos = np.searchsorted(flag_sid, sid, sorter=order)
    assert pos.max() < len(flag_sid), "source_id missing from the outlier flag file"
    idx = order[pos]
    assert np.array_equal(flag_sid[idx], sid), \
        "source_id mismatch between pasta.fits and the outlier flag file"
    out["outlier_flag"] = flag_val[idx]

    print(f"Writing {OUTPUT_FITS} ({len(out.colnames)} columns) ...")
    out.write(OUTPUT_FITS, overwrite=True)
    print("Done.")


if __name__ == "__main__":
    main()
