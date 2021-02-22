import sqlite3
import os
import sys

"""

this program serves to upload the logs that I produced with the previous version of
the program

it is run

python insertLogToDB.py <logName> <step in RA> <step in Dec>

steps are in degrees

"""

inputFileName = sys.argv[1]
print(inputFileName)
stepRA = sys.argv[2]
stepDec = sys.argv[3]

if not os.path.exists('pasta.db'):
 sys.exit('pasta.db does not exists')
else:
 conn = sqlite3.connect('pasta.db')
 
 cur = conn.cursor()
 cur.execute('select ID from PASTA')
 rows = cur.fetchall()
 print(rows)
 if len(rows) == 0 :
  myID = 0
 else:
  myID = rows[0]

 if os.path.exists(inputFileName):
  inputFile = open(inputFileName,'r') 
  for eachLine in inputFile :
   eachItem = eachLine.split()
   print(myID,eachLine,eachItem)
   myID += 1
   minRA = eachItem[1]
   maxRA = float(eachItem[1]) + float(stepRA)
   minDec = eachItem[0]
   maxDec = float(eachItem[0]) + float(stepDec)
   n_obj_gaia = eachItem[2]
   n_obj_galex = eachItem[3]
   n_obj_allwise = eachItem[4]
   sqlCommand = 'INSERT INTO PASTA VALUES (' + \
    str(myID) + ',' + \
    str(minRA) + ',' + \
    str(maxRA) + ',' + \
    str(minDec) + ',' + \
    str(maxDec) + ',' + \
    str(n_obj_gaia) + ',' + \
    str(n_obj_galex) + ',' + \
    str(n_obj_allwise) + ')'
   cur.execute(sqlCommand)
  
  #'''
  #sqlCommand = '''
  #INSERT INTO PASTA 
  #
  #
  # ''''
  #
  #conn.execute('''CREATE TABLE PASTA
  #         (ID INT PRIMARY KEY     NOT NULL,
  #         RA_MIN           REAL    NOT NULL,
  #         RA_MAX           REAL    NOT NULL,
  #         DEC_MIN           REAL    NOT NULL,
  #         DEC_MAX           REAL    NOT NULL,
  #         N_OBJ_GAIA        REAL,
  #         N_OBJ_GALEX        REAL,
  #         N_OBJ_ALLWISE        REAL)
  #         ''')
  #print("Table created successfully")
  #'''
  
  inputFile.close()
 else:
  sys.exit(inputFileName + ' does not exist')
 conn.commit()

 conn.close()