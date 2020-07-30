#!/bin/bash

#python filter.py 

shuf -n 5075 Ia.csv > Ia.csv.5075

head -n 4553 Ia.csv.5075 > train.csv
head -n 913 Ib.csv >> train.csv
head -n 988 Ic.csv >> train.csv
head -n 734 IIn.csv >> train.csv
head -n 918 IIP.csv >> train.csv

tail -n 1522 Ia.csv.5075 | head -n 1015 > val.csv
tail -n 391 Ib.csv | head -n 261 >> val.csv
tail -n 423 Ic.csv | head -n 282 >> val.csv
tail -n 314 IIn.csv | head -n 210 >> val.csv
tail -n 394 IIP.csv | head -n 262 >> val.csv

tail -n 507 Ia.csv.5075 > test.csv
tail -n 130 Ib.csv >> test.csv
tail -n 141 Ic.csv >> test.csv
tail -n 132 IIP.csv >> test.csv
tail -n 104 IIn.csv >> test.csv
