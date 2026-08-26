#!/usr/bin/env python3
"""
make_simbad_xmatch.py
Cross-match the full PaStA catalogue against SIMBAD via the CDS XMatch
service to retrieve known object types for every matched source.

The catalogue is split into chunks to respect the XMatch service limits.
Each chunk result is saved immediately so the script is fully resumable:
re-running it will skip chunks whose output file already exists.

Output
------
  simbad_xmatch/chunk_NNNNN.fits   — per-chunk match tables
  simbad_xmatch.fits               — final concatenated match table

Columns in the output
---------------------
  pasta_source_id   — Gaia source_id from PaStA
  ra, dec           — PaStA J2000 coordinates used for the query
  angDist           — angular separation [arcsec] to SIMBAD position
  main_id           — SIMBAD primary identifier
  otype             — SIMBAD main object type
  otypes            — all SIMBAD object types (pipe-separated)
  sp_type           — spectral type (if available)

Run from the paper root directory:
    python make_simbad_xmatch.py
"""

import os
import sys
import time
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from astropy import units as u
from astroquery.xmatch import XMatch

PASTA_FILE   = 'pasta.fits'
CHUNK_DIR    = 'simbad_xmatch'
FINAL_FILE   = 'simbad_xmatch.fits'
CHUNK_SIZE   = 500_000      # rows per XMatch request
MATCH_RADIUS = 3.0          # arcsec
RETRY_MAX    = 3
RETRY_WAIT   = 30           # seconds between retries

os.makedirs(CHUNK_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load only the columns we need from the 3.4 GB catalogue
# ---------------------------------------------------------------------------
print('Loading PaStA positions ...', flush=True)
with fits.open(PASTA_FILE, memmap=True) as hdul:
    data = hdul[1].data
    t_full = Table({
        'source_id': np.array(data['source_id']),
        'raj2000':   np.array(data['raj2000'],  dtype=np.float64),
        'dej2000':   np.array(data['dej2000'],  dtype=np.float64),
    })
n_total = len(t_full)
n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f'  {n_total:,} sources → {n_chunks} chunks of ≤ {CHUNK_SIZE:,}', flush=True)

# ---------------------------------------------------------------------------
# Query each chunk
# ---------------------------------------------------------------------------
chunk_files = []
for i in range(n_chunks):
    chunk_file = os.path.join(CHUNK_DIR, f'chunk_{i:05d}.fits')
    chunk_files.append(chunk_file)

    if os.path.exists(chunk_file):
        n_rows = len(Table.read(chunk_file))
        print(f'  chunk {i+1:4d}/{n_chunks}  [SKIP — {n_rows} matches already saved]',
              flush=True)
        continue

    lo, hi = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, n_total)
    chunk = t_full[lo:hi]
    # XMatch expects columns named 'ra' and 'dec'
    chunk.rename_columns(['raj2000', 'dej2000'], ['ra', 'dec'])

    for attempt in range(1, RETRY_MAX + 1):
        try:
            result = XMatch.query(
                cat1=chunk,
                cat2='simbad',
                max_distance=MATCH_RADIUS * u.arcsec,
                colRA1='ra',
                colDec1='dec',
                responseformat='votable',
            )
            break
        except Exception as exc:
            print(f'    attempt {attempt} failed: {exc}', flush=True)
            if attempt < RETRY_MAX:
                print(f'    retrying in {RETRY_WAIT}s ...', flush=True)
                time.sleep(RETRY_WAIT)
            else:
                print('    giving up on this chunk; saving empty placeholder.',
                      flush=True)
                result = Table(
                    names=['source_id', 'ra', 'dec', 'angDist',
                           'main_id', 'otype', 'otypes', 'sp_type'],
                    dtype=['>i8', '>f8', '>f8', '>f4', 'U64', 'U32', 'U256', 'U32'],
                )

    # Keep only the closest match per PaStA source
    if len(result) > 0:
        result.sort('angDist')
        _, idx = np.unique(result['source_id'], return_index=True)
        result = result[idx]

    # Ensure consistent column names
    result.rename_column('source_id', 'pasta_source_id') \
        if 'source_id' in result.colnames else None

    result.write(chunk_file, overwrite=True)
    print(f'  chunk {i+1:4d}/{n_chunks}  rows {lo:>8,}–{hi:>8,}  '
          f'{len(result):>6,} matches  → {chunk_file}', flush=True)

# ---------------------------------------------------------------------------
# Concatenate
# ---------------------------------------------------------------------------
print('\nConcatenating chunks ...', flush=True)
tables = [Table.read(f) for f in chunk_files if os.path.exists(f)]
final = vstack(tables)
final.sort('angDist')
_, idx = np.unique(final['pasta_source_id'], return_index=True)
final = final[idx]
final.sort('pasta_source_id')

final.write(FINAL_FILE, overwrite=True)
print(f'Saved {FINAL_FILE}  ({len(final):,} matched sources out of {n_total:,} total)')
print(f'Match rate: {100 * len(final) / n_total:.2f} per cent')
