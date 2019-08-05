#!/bin/bash

for n in 0.01 0.03 0.07 0.10
do
    python train_seed_u_E.py -e 10000 -bs 1 -n $n -nb 1 -rd ./results/presentation/seed_u_v1_E -h5 ./data/N_u_256/N_c_20000/dataset.hdf5
done
