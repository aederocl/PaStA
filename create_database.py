import sqlite3
import os
import sys

if os.path.exists('pasta.db'):
 sys.exit('pasta.db already exists')
else:
 conn = sqlite3.connect('pasta.db')
 conn.execute('''CREATE TABLE PASTA
          (ID INT PRIMARY KEY     NOT NULL,
          RA_MIN           REAL    NOT NULL,
          RA_MAX           REAL    NOT NULL,
          DEC_MIN           REAL    NOT NULL,
          DEC_MAX           REAL    NOT NULL,
          N_OBJ_GAIA        REAL,
          N_OBJ_GALEX        REAL,
          N_OBJ_ALLWISE        REAL)
          ''')
 print("Table created successfully")

 conn.close()