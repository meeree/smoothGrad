import matplotlib.pyplot as plt
import numpy as np
from utils import c_vals, c_vals_l, c_vals_d
import sys

data = []
prefix = sys.argv[1] + '/'
print('Prefix: ', prefix)
for name in ['spk_hidden.pt', 'spk_out.pt', 'labels_out.pt', 'W_ro.pt']:
    with open(prefix + name, 'rb') as f:
        print(prefix + name)
        data.append(np.load(f))
spk_hidden, spk_out, labels, W_ro = data 

def fixed_case(c):
    plt.figure(c)
    hidden, out, label = spk_hidden[c], spk_out[c], labels[c]

    plt.subplot(3,3,(1, 3))
    plt.plot(hidden)
    plt.title(label)

    vmax = np.max(np.abs(W_ro))
    for l in range(3):
        plt.subplot(3,3,4 + l)
        good = W_ro[l]

        hidden_spks = (hidden > -40).astype(float)
        colored = good.reshape(1, -1) * hidden_spks

        # Sort so that good is at top.
        inds = np.argsort(good)
#        colored = colored[:, inds]

        plt.imshow(colored.T, aspect='auto', cmap='seismic', vmin = -vmax, vmax = vmax)
        plt.colorbar()

    plt.subplot(3,3,(7,9))
    for i in range(out.shape[-1]):
        plt.plot(out[:, i], c = c_vals[i], label=str(i))
    plt.legend()

def grid_of_inputs_for_label(label_inds):
    vmax = np.max(np.abs(W_ro))
    vmin = np.min(W_ro)
    vmin = min(vmin, 0.0)
    W = 10
    for c in label_inds:
        hidden, out, label = spk_hidden[c], spk_out[c], labels[c]
        if len(label.shape) > 0:
            hidden, out, label = np.mean(hidden, 0), np.mean(out, 0), label[0]

        hidden = (hidden > -25).astype(float)

        convolved = []
        for l in range(3):
            good = W_ro[l]

            colored = good.reshape(1, -1) * hidden

            plt.subplot(6,4,label * 8 + l + 1)
            z = out[:, l]

            if l == 0:
                plt.ylabel(f'Label = {label}')

            smoothed = np.convolve(z, np.ones(W), 'valid')
            convolved.append(smoothed)
            smoothed = z
            plt.plot(smoothed)

            plt.subplot(6,4,label * 8 + l + 5)
            plt.imshow(colored.T, aspect='auto', cmap='seismic', vmin = vmin, vmax = vmax)
            plt.colorbar()

        convolved = np.array(convolved)
        winning = np.argmax(np.array(convolved), 0)
        print(winning[-1])
        plt.subplot(6,4,label * 8 + 4)
        plt.imshow(winning.reshape(1,-1), aspect='auto', cmap = 'seismic')
        plt.yticks([])
        cbar = plt.colorbar(ticks = [0, 1, 2])


plt.plot(spk_out[0, -30:, :])
plt.legend([0, 1, 2])
plt.title(f'Label = {labels[0]}')
plt.show()
exit()

inds = [np.where(labels == l)[0][0] for l in range(3)]
grid_of_inputs_for_label(inds)
#inds = [np.argwhere(labels == l).reshape(-1) for l in range(3)]
#grid_of_inputs_for_label(inds)
plt.show()
exit()

fixed_case(2)
fixed_case(0)
fixed_case(1)
plt.show()
