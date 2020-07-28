#!/bin/bash

python filter.py > data.txt

grep ", Ia" data.txt | shuf -n 1500 > Ia.txt
grep ", Ib" data.txt | shuf > Ib.txt
grep ", Ic" data.txt | shuf > Ic.txt
grep ", IIP" data.txt | shuf > IIP.txt
grep ", IIn" data.txt | shuf > IIn.txt
shuf train.txt > shuffle-train.txt

tail -n 450 Ia.txt | head -n 300 > val.txt
tail -n 391 Ib.txt | head -n 261 >> val.txt
tail -n 423 Ic.txt | head -n 282 >> val.txt
tail -n 314 IIn.txt | head -n 210 >> val.txt
tail -n 394 IIP.txt | head -n 262 >> val.txt

tail -n 150 Ia.txt > test.txt
tail -n 130 Ib.txt >> test.txt
tail -n 141 Ic.txt >> test.txt
tail -n 132 IIP.txt >> test.txt
tail -n 104 IIn.txt >> test.txt
