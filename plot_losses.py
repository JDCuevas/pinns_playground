import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from visualization import plotting
from sklearn.preprocessing import StandardScaler

def plot_losses(target, values, path="./results/N_u_256/N_c_20000/Noise_0.01/E_8000/BS_2/losses_per_epoch.csv"):
    model_dir = path
    substring_list = ['N_u', 'N_c', 'Noise', 'E', 'BS', 'Boost']
    substring_list.remove(target)

    path_index = model_dir.index(target + '_')
    path_beg = model_dir[:path_index]
    path_end = model_dir[path_index:][model_dir[path_index:].index("/"):] if len(model_dir[path_index:].split("/")) > 1 else ""

    for value in values:
        target_folder = target + '_' +  value
        p = path_beg + target_folder + path_end
        
        print(p)

        df = pd.read_csv(p)
        
        epochs = df['epoch']

        joint_loss = df['joint_loss'].values

        loss_u = df['loss_u'].values

        loss_u_D = df['loss_u_D'].values

        loss_F = df['loss_F'].values

        loss_S = df['loss_S'].values

        plotting.plot_simple([(epochs, loss_u)], ['loss_u'], "epoch", "loss", filename=p[:p.index('losses_per_epoch.csv')] + "u_loss_plot.png", y_lim=(np.min(loss_u), np.max(loss_u[500:])), loc='upper right', epochs=True)
        plotting.plot_simple([(epochs, loss_u_D)], ['loss_u_D'], "epoch", "loss", filename=p[:p.index('losses_per_epoch.csv')] + "u_D_loss_plot.png", y_lim=(np.min(loss_u_D), np.max(loss_u_D[500:])), loc='upper right', epochs=True)
        plotting.plot_simple([(epochs, loss_F)], ['loss_F'], "epoch", "loss", filename=p[:p.index('losses_per_epoch.csv')] + "F_loss_plot.png", y_lim=(np.min(loss_F), np.max(loss_F[500:])), loc='upper right', epochs=True)
        plotting.plot_simple([(epochs, loss_S)], ['loss_S'], "epoch", "loss", filename=p[:p.index('losses_per_epoch.csv')] + "S_loss_plot.png", y_lim=(np.min(loss_S), np.max(loss_S[500:])), loc='upper right', epochs=True)
        plotting.plot_simple([(epochs, joint_loss)], ['joint_loss'], "epoch", "loss", filename=p[:p.index('losses_per_epoch.csv')] + "joint_loss_plot.png", name='Joint loss vs. Epochs', y_lim=(np.min(joint_loss), np.max(joint_loss[500:])), loc='upper right', epochs=True)

if __name__ == '__main__':
    plot_losses(target='Boost', values=['30'], path="./results/presentation/seed_u_v1/N_u_256/N_c_20000/Noise_0.05/E_10000/BS_128/Boost_30/losses_per_epoch.csv")