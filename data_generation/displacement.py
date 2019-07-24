import os
import numpy as np
from argparse import ArgumentParser
import h5py

parser = ArgumentParser(description = "Displacement Data Generator")
parser.add_argument('-N_u','--num_measurements', default=256, type=int)
parser.add_argument('-N_c','--num_collocation', default=20000, type=int)
parser.add_argument('-L','--length', default=1.0, type=float)
parser.add_argument('-S','--sigma', default=0.5, type=float)

parser.add_argument('-n', '--name', default = 'dataset')
parser.add_argument('-d', '--data_dir', default = './data/')

args = parser.parse_args()

# Plot data dist
sigma = args.sigma
L = args.length
x_true = np.linspace(0, L)
E_true = (1 + x_true) ** 2
u_true = sigma * (1 - (1.0 / (1 + x_true)))

# uniformly distributed collocation points
N_c = args.num_collocation
x_c = np.linspace(0, 1, N_c)
x_c = x_c.reshape((-1, 1)) # (samples(N_c), n_features(x))

# randomly located displacement measurements
N_u = args.num_measurements
x_u = np.random.random(N_u)
u_u = sigma * (1 - 1.0 / (1 + x_u))
x_u = x_u.reshape((-1, 1))
u_u = u_u.reshape((-1, 1))

# one Dirichlet BC
N_D = 1
x_D = np.array([[0.0]])
u_D = np.array([0.0])

# notch E (not being used)
N_E = 1
x_E = np.random.random(N_E)
E_E = (1 + x_E) ** 2
x_E = x_E.reshape((-1, 1))
E_E = E_E.reshape((-1, 1))

# one BC for sigma
x_S = np.array([[L]])

# Create dir
if not os.path.exists(os.path.dirname(args.data_dir)):
            try:
                os.makedirs(os.path.dirname(args.data_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

with h5py.File(args.data_dir + args.name + '.hdf5', 'w') as f:
    L = f.create_dataset('L', data=np.array(L))
    sigma = f.create_dataset('sigma', data=np.array(sigma))

    true_data = f.create_group('true_data')
    true_data.create_dataset('x_true', data=x_true)
    true_data.create_dataset('u_true', data=u_true)
    true_data.create_dataset('E_true', data=E_true)

    measurements = f.create_group('measurements')
    measurements.create_dataset('x_u', data=x_u)
    measurements.create_dataset('u_u', data=u_u)

    collocation_pts = f.create_group('collocation_pts')
    collocation_pts.create_dataset('x_c', data=x_c)

    dirichlet_bc = f.create_group('dirichlet_bc')
    dirichlet_bc.create_dataset('x_D', data=x_D)
    dirichlet_bc.create_dataset('u_D', data=u_D)

    sigma_bc = f.create_dataset('sigma_bc', data=x_S)