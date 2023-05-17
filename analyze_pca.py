from utils import c_vals, c_vals_l, c_vals_d
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_pca(hs_pca, labels):
    color_type = 'labels' # 'labels', 'words', 'seq_idx', 'labels_1', 'labels_2'
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
    
    pcxs = (0, 0, 1)
    pcys = (1, 2, 2)
    hs_pca = hs_pca[:, :, :max(pcxs + pcys) + 1]
    ax_pair_lims = (None, None, None)
    
    cmap = plt.cm.get_cmap('plasma')
    # phrase_len = hs_fit.shape[1]
    
    for ax, pcx, pcy, ax_pair_lim in zip((ax1, ax2, ax3), pcxs, pcys, ax_pair_lims):
        for batch_idx in range(labels.shape[0]):
            if color_type == 'labels':
                color_labels_l = c_vals_l[labels[batch_idx]]
                color_labels = c_vals[labels[batch_idx]]

            ax.plot(hs_pca[batch_idx, :, pcx], hs_pca[batch_idx, :, pcy], linewidth=0.5,
                       color=color_labels_l, zorder=0, alpha=0.5)

#            ax.scatter(hs_pca[batch_idx, :, pcx], hs_pca[batch_idx, :, pcy], linewidth=0.0,
#                       marker='o', color=color_labels_l, zorder=0, alpha=0.5)
            ax.scatter(hs_pca[batch_idx, -1, pcx], hs_pca[batch_idx, -1, pcy], marker='o',
                       color=color_labels, zorder=5, alpha=0.5)
    
        # for ro_idx in range(net_params['n_outputs']):
        #     ax.plot([zero_matrix_pca[ro_idx, pcx], ro_matrix_pca[ro_idx, pcx]], 
        #             [zero_matrix_pca[ro_idx, pcy], ro_matrix_pca[ro_idx, pcy]],
        #              color=c_vals_d[ro_idx], linewidth=3.0, zorder=10)
    
        # Some sample paths
        for batch_idx in range(0):
            ax.plot(hs_pca[batch_idx, :, pcx], hs_pca[batch_idx, :, pcy], marker='.',
                    color=c_vals_d[labels[batch_idx]], zorder=5)
    
        ax.set_xlabel('PC{}'.format(pcx))
        ax.set_ylabel('PC{}'.format(pcy))
        # Sets square limits if desired
        if ax_pair_lim is not None:
            ax.set_xlim((-ax_pair_lim, ax_pair_lim))
            ax.set_ylim((-ax_pair_lim, ax_pair_lim))
    
    if color_type == 'labels':
        ax2.set_title('Hidden state PC plots (color by phrase label)')
    elif color_type == 'words':
        ax2.set_title('Hidden state PC plots (color by input word)')
    elif color_type == 'seq_idx':
        ax2.set_title('Hidden state PC plots (color by sequence location)')
    plt.show()

folder = sys.argv[1] + '/'
with open(folder + 'pca.npy', 'rb') as f:
    pca = np.load(f)

with open(folder + 'labels.npy', 'rb') as f:
    labels = np.load(f)

print(pca.shape, labels.shape)

plot_pca(pca, labels = labels)
