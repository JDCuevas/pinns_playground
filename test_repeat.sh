#!/bin/bash

for n in 1 2 3 4 5 6 7 8 9 10 
do
    python train.py --e 8000 -bs 2 -n 0.03 -rd ./results/repeat_test/test_$n -h5 ./data/repeat_test/dataset.hdf5
done
