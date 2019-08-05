#!/bin/bash

for N_c in 10000 15000 20000
do
    for n in 0.01 0.03 0.07 0.10
    do
        python train_seed_u.py -e 6000 -bs 1 -n $n -nb 10 -rd ./results/presentation/seed_u_v1 -h5 ./data/N_u_256/N_c_$N_c/dataset.hdf5
    done
done