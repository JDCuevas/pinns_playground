#!/bin/bash

for N_u in 256 512 1024
do
    for N_c in 5000 10000 15000 20000
    do
        python data_generation/displacement.py -N_u $N_u -N_c $N_c -L 1.0 -S 0.5 -d ./data/N_u_$N_u/N_c_$N_c/
    done
done

