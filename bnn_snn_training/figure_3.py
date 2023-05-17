import torch
import sys
sys.path.append('../../mpn') # Replace with your own relative path.
import int_data as syn
from networks import VanillaBNN
from utils import fit, to_dataset, cutoff_data, get_extreme_data, eval_on_test_set, sliding_window_states, plot_lr_decay, PCA_dim, plot_pca, plot_accuracy, c_vals
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title

import time
import numpy as np
import os

def average_loss(loss, n_avg = 40, truncate = False):
    ''' Run a sliding window of length n_avg over loss to compute averaged loss curve. '''
    out = []
    for i in range(len(loss)):
        end = min(i + n_avg, len(loss))
        if truncate and end < i + n_avg:
            break # Only average when window contains n_avg things.
        out.append(np.mean(loss[i:end]))
    return np.array(out)

def plot_accuracy(folders_full, methods):
    import json, glob
    accs = []

    for folder_full in folders_full:
        fl_names = [os.path.basename(fl) for fl in glob.glob(folder_full + '/save_*.pt')]
        fl_names.sort(key = lambda fl: int(fl[5:-3]))
        sd = torch.load(folder_full + '/' + fl_names[-1])
        hist = sd['hist']
        accs.append(hist['valid_acc'])

    min_len = min([len(acc) for acc in accs])
    accs = np.array([acc[:min_len] for acc in accs])

    plt.figure(figsize = (7, 4))
    for key in ['song', 'bp']:
        mean_acc = np.mean(accs[method[key]], 0)
        std_acc = np.std(accs[method[key]], 0)

        # Smooth over iterations with a small window to filter out some noise.
        mean_acc = average_loss(mean_acc, 10, truncate=True)
        std_acc = average_loss(std_acc, 10, truncate=True)

        # Mean line
        X = range(len(mean_acc))
        c = 'red' if key == 'song' else 'black'
        plt.plot(X, mean_acc, c = c, zorder = 10)

        # Std area
        plt.fill_between(X, mean_acc - std_acc, mean_acc + std_acc, color = c, alpha = 0.5, zorder = 0, linewidth=0, label = '_nolegend_')

    legend = plt.legend(['SONG', 'BP'])
    legend.get_frame().set_alpha(0) 
    plt.xlabel('Iteration')
    plt.ylabel('Prediction Accuracy (%)')
    plt.tight_layout()

    plt.savefig('accuracy_integrator.pdf')

    plt.show()
 
folder_1 = 'SAVES/song_all_lr0.005_std0.1_S100'
folder_2 = 'SAVES/song_all_lr0.005_std0.1_S100_RUN2'
folder_3 = 'SAVES/song_all_lr0.005_std0.1_S100_RUN3'
folder_4 = 'SAVES/song_all_lr0.005_std0.1_S100_RUN4'
folder_5 = 'SAVES/all_autodiff_lr0.005'
folder_6 = 'SAVES/all_autodiff_lr0.005_RUN2'
folder_7 = 'SAVES/all_autodiff_lr0.005_RUN3'
folder_8 = 'SAVES/all_autodiff_lr0.005_RUN4'
folder_9 = 'SAVES/all_autodiff_lr0.005_RUN5'
folders = [folder_1, folder_2, folder_3, folder_4, folder_5, folder_6, folder_7, folder_8, folder_9]
method = {'song': [0, 1, 2, 3], 'bp': [4, 5, 6, 7, 8]}
plot_accuracy(folders, method)
