#!/bin/bash

#python filter.py 

shuf -n 1500 Ia.csv > Ia.csv.1500

head -n 1050 Ia.csv.1500 > train.csv
head -n 876 Ib.csv >> train.csv
head -n 969 Ic.csv >> train.csv
head -n 715 IIn.csv >> train.csv
head -n 901 IIP.csv >> train.csv

tail -n 450 Ia.csv.1500 | head -n 300 > val.csv
tail -n 375 Ib.csv | head -n 250 >> val.csv
tail -n 416 Ic.csv | head -n 277 >> val.csv
tail -n 307 IIn.csv | head -n 204 >> val.csv
tail -n 386 IIP.csv | head -n 257 >> val.csv

tail -n 150 Ia.csv.1500 > test.csv
tail -n 125 Ib.csv >> test.csv
tail -n 139 Ic.csv >> test.csv
tail -n 103 IIn.csv >> test.csv
tail -n 129 IIP.csv >> test.csv
