import os
import time
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from argparse import ArgumentParser
from visualization import plotting
import h5py

parser = ArgumentParser(description = "PINNs Playground")
parser.add_argument('-e', '--epochs', default=8000, type=int)
parser.add_argument('-bs', '--batch_size_den', default=2, type=int) # batch size denominator -> DATASET SIZE / batch_size_den
parser.add_argument('-n','--noise', default=0.01, type=float)
parser.add_argument('-sd','--seed', default=1337, type=int)
parser.add_argument('-nb','--nn_boost', default=100, type=int)

parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)

parser.add_argument('-s', '--save', default = True, type=bool)
parser.add_argument('-h5', '--hdf5', default = './data/dataset.hdf5')
parser.add_argument('-rd', '--result_dir', default = './results')

args = parser.parse_args()

with h5py.File(args.hdf5, 'r') as f:

    key_tree = {}

    def make_key_tree(name, obj):
        split = name.split('/')
        key = split[len(split) - 1]
        if isinstance(obj, h5py.Dataset):
            key_tree.update({key : name})
    
    f.visititems(make_key_tree)

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

# Add Noise
std_dev = args.noise * np.mean(u_u)
np.random.seed(args.seed)
noise = np.random.normal(scale=std_dev, size=N_u)
noise = noise.reshape((-1, 1))
u_u = np.add(u_u, noise)

# PARAMETERS
DATASET_SIZE = N_u
BATCH_SIZE = int(DATASET_SIZE/args.batch_size_den)
ITERATIONS = int(DATASET_SIZE/BATCH_SIZE)
EPOCHS = args.epochs
INPUT_DIM = 1

print('\n\n######### SETTINGS #########\n')
print('EPOCHS: {}'.format(args.epochs))
print('BATCH SIZE: DATASET_SIZE / {} = {}'.format(args.batch_size_den, BATCH_SIZE))
print('NOISE Level: {}'.format(args.noise))
print('LEARNING RATE: {}'.format(args.learning_rate))
print('Material Length: {}'.format(L))
print('Stress (sigma): {}'.format(sigma))
print('N_u: {}'.format(N_u))
print('N_c: {}'.format(N_c))
print('\n')

# RESULTS DIRECTORY
model_dir = args.result_dir + '/N_u_' + str(N_u) + '/N_c_' + str(N_c) + '/Noise_' + str(args.noise) + '/E_' + str(args.epochs) + '/BS_' + str(BATCH_SIZE) + '/Boost_' + str(args.nn_boost) + '/'

if not os.path.exists(os.path.dirname(model_dir)):
            try:
                os.makedirs(os.path.dirname(model_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

# MODEL
def pinn_model():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))
    
    model.add(layers.Dense(64))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('sigmoid'))

    model.add(layers.Dense(64))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('sigmoid'))
    
    model.add(layers.Dense(1))
    
    return model

def pinn_model_alt():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))
    
    model.add(layers.Dense(64))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(64))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(64))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(1))
    
    return model    

nn_u = pinn_model()
nn_u.summary()

nn_E = pinn_model()
nn_E.summary()

# LOSS AND OPTIMIZER HELPER FUNCTIONS
mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

# TRAINING 

## Training step
@tf.function
def train_step(u_batch, u_expected, u_bc, u_bc_expected, collocation_points, sigma_bc):
    with tf.GradientTape(persistent=True) as u_tape, tf.GradientTape(persistent=True) as E_tape:
        u_tape.watch(collocation_points)
        u_tape.watch(sigma_bc)
        E_tape.watch(collocation_points)

        for i in range(args.nn_boost):
            nn_pred_u = nn_u(u_batch, training=True)
            nn_pred_u_D = nn_u(u_bc, training=True)

            loss_u = mse(u_expected, nn_pred_u)
            loss_u_D = mse(u_bc_expected, nn_pred_u_D)

            u_joint_loss = loss_u + loss_u_D   

            gradients_of_nn_u = u_tape.gradient(u_joint_loss, nn_u.trainable_variables)
            optimizer.apply_gradients(zip(gradients_of_nn_u, nn_u.trainable_variables))

        #nn_pred_u = nn_u(u_batch, traininnt_loss = loss_u + loss_u_D + loss_F + loss_S #g=True)
        #nn_pred_u_D = nn_u(u_bc, training=True)
        nn_pred_u_c = nn_u(collocation_points,training=True)
        nn_pred_u_S = nn_u(sigma_bc, training=True)
        nn_pred_E_c = nn_E(collocation_points, training=True)
        nn_pred_E_S = nn_E(sigma_bc, training=True)
        
        #loss_u = mse(u_expected, nn_pred_u)
        #loss_u_D = mse(u_bc_expected, nn_pred_u_D)

        u = nn_pred_u_c
        E = nn_pred_E_c
        x = collocation_points
        du_dx = u_tape.gradient(u, x)
        
        du_dxx = u_tape.gradient(du_dx,x)
        dE_dx = E_tape.gradient(E, x)
        
        f = tf.multiply(E, tf.cast(du_dxx, tf.float32)) + tf.multiply(tf.cast(dE_dx, tf.float32), tf.cast(du_dx, tf.float32))
        
        loss_F = tf.reduce_mean(tf.square(f))
        
        u_S = nn_pred_u_S
        E_S = nn_pred_E_S
        du_dx_S = u_tape.gradient(u_S, sigma_bc)

        loss_S = mse(tf.constant(sigma), tf.multiply(E_S, tf.cast(du_dx_S, tf.float32)))
        
        joint_loss = loss_F + loss_S
        
    gradients_of_nn_u = u_tape.gradient(joint_loss, nn_u.trainable_variables)
    gradients_of_nn_E = E_tape.gradient(joint_loss, nn_E.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients_of_nn_u, nn_u.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_nn_E, nn_E.trainable_variables))
    
    del u_tape
    del E_tape
    
    return joint_loss, loss_u, loss_u_D, loss_F, loss_S

## Train loop
def train():
    losses_per_epoch = []

    for epoch in tqdm(range(EPOCHS)):
        start = time.time()
        iterations_loss = []
        
        for iteration in range(ITERATIONS):
            u_batch = x_u[iteration * BATCH_SIZE : (iteration + 1) * BATCH_SIZE]
            u_expected = u_u[iteration * BATCH_SIZE : (iteration + 1) * BATCH_SIZE]
            u_bc = x_D
            u_bc_expected = u_D
            collocation_points = x_c
            sigma_bc = x_S
            
            joint_loss, loss_u, loss_u_D, loss_F, loss_S = train_step(u_batch, u_expected, u_bc, u_bc_expected, collocation_points, sigma_bc)
            
            ''' Accuracy Calculations
            predicted_u = nn_u.predict(x_c[iteration * N_u : (iteration + 1) * N_u])
            accuracy_u = np.linalg.norm((u_u - predicted_u)/u_u)
            
            predicted_E = nn_E.predict(x_c[iteration * N_u : (iteration + 1) * N_u])
            accuracy_E = np.linalg.norm((E_u - predicted_E)/E_u)
            '''
            iterations_loss.append({'joint_loss' : joint_loss.numpy(), 'loss_u' : loss_u.numpy(), 'loss_u_D' : loss_u_D.numpy(), 'loss_F' : loss_F.numpy(), 'loss_S' : loss_S.numpy()})

        joint_loss = np.array([0.0])
        loss_u = np.array([0.0])
        loss_u_D = np.array([0.0])
        loss_F = np.array([0.0])
        loss_S = np.array([0.0])

        for i in range(len(iterations_loss)):
            joint_loss =+ iterations_loss[i]['joint_loss']
            loss_u =+ iterations_loss[i]['loss_u']
            loss_u_D =+ iterations_loss[i]['loss_u_D']
            loss_F =+ iterations_loss[i]['loss_F']
            loss_S =+ iterations_loss[i]['loss_S']

        joint_loss = joint_loss / ITERATIONS
        loss_u = loss_u / ITERATIONS
        loss_u_D = loss_u_D / ITERATIONS
        loss_F = loss_F / ITERATIONS
        loss_S = loss_S / ITERATIONS

        losses_per_epoch.append({'epoch' : epoch, 'joint_loss' : joint_loss, 'loss_u' : loss_u, 'loss_u_D' : loss_u_D, 'loss_F' : loss_F, 'loss_S' : loss_S})

        if (epoch + 1) % 100 == 0:
            print('\nTime for epoch {} is {:.4f} sec'.format(epoch + 1, time.time() - start))
            print("Loss: {:.8f}\n".format(joint_loss))
            print("Loss u: {:.8f}\t Loss u_D: {:.8f}\t Loss F: {:.8f}\t Loss S: {:.8f}".format(loss_u, loss_u_D, loss_F, loss_S))
            #print("Accuracy u: {:.8f}\t Accuracy E: {:.8f}".format(accuracy_u, accuracy_E))

    with open(model_dir + 'losses_per_epoch.csv', mode='w') as csv_file:
        fieldnames = ['epoch', 'joint_loss', 'loss_u', 'loss_u_D', 'loss_F', 'loss_S']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(losses_per_epoch)


train()

# PREDICTIONS
predicted_u = nn_u.predict(x_c)
predicted_E = nn_E.predict(x_c)

plotting.plot_simple([(x_true, u_true), (x_c, predicted_u)], ['true', 'predicted'], "x", "u", filename=model_dir + "u_plot.png", name='Displacement vs. Position')
plotting.plot_simple([(x_true, E_true), (x_c, predicted_E)], ['true', 'predicted'], "x", "E", filename=model_dir + "E_plot.png", name='Stiffness vs. Position')