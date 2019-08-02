#!/bin/bash

for Noise in 0.05 0.07 0.10
do
    for E in 6000 8000 10000 
    do
        for BS in 1 2 4 8 
        do
            for Boost in 60 80 100 150
            do
                python train_seed_u_v2.py -e $E -bs $BS -n $Noise -nb $Boost -rd ./results/parameters_u_seed_v2 -h5 ./data/N_u_256/N_c_10000/dataset.hdf5
            done
        done
    done
done


