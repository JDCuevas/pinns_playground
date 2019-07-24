#!/bin/bash

for N_u in 256 512 1024
do
    for N_c in 5000 10000 15000 20000
    do
        for Noise in 0.01 0.03 0.05 0.07
        do
            for E in 6000 8000 10000 
            do
                for BS in 2 4 8 
                do
                python train.py --e $E -bs $BS -n $Noise -h5 ./data/N_u_$N_u/N_c_$N_c/dataset.hdf5
                done
            done
        done
    done
done

