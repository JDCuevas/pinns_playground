import numpy as np
from argparse import ArgumentParser
import h5py

parser = ArgumentParser(description = "HDF5 Viewer")

parser.add_argument('-f', '--filepath', default = './data/dataset.hdf5')

args = parser.parse_args()

key_tree = {}

def check_inside(f, items, spacing=''):

    for item in items:
        if isinstance(item[1], h5py.Group):
            print(spacing + '\t' + item[0] + ' -> Group')
            check_inside(f, item[1].items(), spacing=spacing + '\t')

        elif isinstance(item[1], h5py.Dataset):
            print(spacing + '\t' + item[0] + ' -> Dataset, Shape: ' + str(f[item[1].name][()].shape))

def make_key_tree(name, obj):
        split = name.split('/')
        key = split[len(split) - 1]
        if isinstance(obj, h5py.Dataset):
            key_tree.update({key : name})

with h5py.File(args.filepath, 'r') as f:
    print('Keys:')
    check_inside(f, f.items())

    f.visititems(make_key_tree)
    print('\n')
    print(key_tree)

    L = f.get(key_tree['L'])[()]
    sigma = f.get(key_tree['sigma'])[()]

    x_true = f.get(key_tree['x_true'])[()]
    E_true = f.get(key_tree['E_true'])[()]
    u_true = f.get(key_tree['u_true'])[()]

    # uniformly distributed collocation points
    x_c = f.get(key_tree['x_c'])[()]
    N_c = x_c.shape[0]
    
    # randomly located displacement measurements
    x_u = f.get(key_tree['x_u'])[()]
    u_u = f.get(key_tree['u_u'])[()]
    N_u = x_u.shape[0]

    # one Dirichlet BC
    x_D = f.get(key_tree['x_D'])[()]
    u_D = f.get(key_tree['u_D'])[()]
    N_D = x_D.shape[0]

    # one BC for sigma
    x_S = f.get(key_tree['sigma_bc'])[()]