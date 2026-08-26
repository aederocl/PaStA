"""
create_grid.py
Write the driver script that builds PaStA one sky tile at a time.

The catalogue is assembled tile by tile because the Gaia archive, the GALEX
AIS and AllWISE are each queried through STILTS/TAP, and a whole-sky query
would exceed the service limits. This script emits runPaStA.sh, one
getcat_pasta.py invocation per tile, which is then run (serially or in
parallel) before combine_pasta.py concatenates the results.

Tile boundaries
---------------
getcat_pasta.py queries the half-open interval

    minRA <= ra < maxRA   and   minDec <= dec < maxDec

so that a source lying exactly on a tile boundary is captured by one tile
only. This is what removed the 377 duplicated sources present in the
original pasta.fits. The half-open convention has one consequence that must
be handled here and not there: the upper edge of the very last tile in each
coordinate is excluded, so the grid widens it.

  - In right ascension nothing is actually lost, because Gaia reports RA on
    [0, 360) and no source has ra == 360 exactly. The upper bound is widened
    all the same, so that the grid does not depend on that convention.
  - In declination the north celestial pole, dec == +90 exactly, would be
    dropped. The southern edge needs no treatment: dec == -90 satisfies
    dec >= minDec for the first tile.

Usage
-----
    python create_grid.py        # writes runPaStA.sh
    sh runPaStA.sh               # ~2600 tiles at the default 5 deg step
"""

SURVEY_MIN_DEC = -90
SURVEY_MAX_DEC = 90
SURVEY_MIN_RA = 0
SURVEY_MAX_RA = 360

DEC_STEP = 5
RA_STEP = 5

# Widening applied to the upper edge of the final tile in each coordinate,
# to compensate for the half-open interval used by getcat_pasta.py. Any
# value below the tile step works; it must not be zero.
EDGE_PAD = 0.001

OUTPUT = 'runPaStA.sh'
BUILDER = 'getcat_pasta.py'

with open(OUTPUT, 'w') as script:
    for declination in range(SURVEY_MIN_DEC, SURVEY_MAX_DEC, DEC_STEP):
        for right_ascension in range(SURVEY_MIN_RA, SURVEY_MAX_RA, RA_STEP):
            min_dec = float(declination)
            max_dec = float(declination + DEC_STEP)
            min_ra = float(right_ascension)
            max_ra = float(right_ascension + RA_STEP)

            if max_dec >= SURVEY_MAX_DEC:
                max_dec += EDGE_PAD
            if max_ra >= SURVEY_MAX_RA:
                max_ra += EDGE_PAD

            script.write('python %s %s %s %s %s\n'
                         % (BUILDER, min_ra, max_ra, min_dec, max_dec))

print('Wrote %s' % OUTPUT)
