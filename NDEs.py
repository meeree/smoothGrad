import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import matplotlib as mpl

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

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5', atol = 1e-4, rtol = 1e-4)


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs('png')
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)



def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

#        ax_vecfield.cla()
#        ax_vecfield.set_title('Learned Vector Field')
#        ax_vecfield.set_xlabel('x')
#        ax_vecfield.set_ylabel('y')

#        y, x = np.mgrid[-2:2:21j, -2:2:21j]
#        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
#        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
#        dydt = (dydt / mag)
#        dydt = dydt.reshape(21, 21, 2)
#
#        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
#        ax_vecfield.set_xlim(-2, 2)
#        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('nde_comparison.pdf')
        plt.show()

class SmoothGrad():
    def __init__(self, shape, stddev):
        self.shape = shape
        self.stddev = stddev
        self.enabled = False
        
    def __call__(self, losses, offs):
        if not self.enabled:
            return torch.zeros(self.shape).to(losses.device)

        # Sum 1/S * loss(v_i) * (mu_i - v_i) / sigma_i^2 
        S = losses.shape[0]
        smpls = offs.reshape(S, -1)
        grad = torch.mean(losses.reshape(S, 1) * smpls, 0) / (self.stddev**2)
        return grad.reshape(self.shape)

class FiniteDiffGrad():
    def __init__(self, shape, diff):
        self.shape = shape
        self.diff = diff 
        self.enabled = False
        
    def __call__(self, losses, offs):
        if not self.enabled:
            return torch.zeros(self.shape).to(losses.device)
        
        l0 = losses[0]
        smpls = offs.reshape(S, -1)
        grad = (losses - l0) / smpls
        grad = torch.nan_to_num(grad, posinf=0, neginf=0)
        grad = torch.sum(grad, 0)
        return grad.reshape(self.shape)

class WeightSampler(torch.jit.ScriptModule):
    '''Class supporting adding S offsets to weight matrix'''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Linear(output_dim, input_dim, bias=False).weight # note transpose
        self.W = nn.Parameter(self.W)
        self.noise = torch.zeros(())
        self.W_noisy = torch.zeros(())
        self.set_params(1, 0.0)
        
    def set_params(self, S, stddev):
        self.S = S
        self.stddev = stddev
    
    def noisify(self, finite_diff=False):
        # Copies of weight for noisy sampling over batches and S samples.
        self.W_noisy = self.W
        self.W_noisy = self.W_noisy
        self.W_noisy = self.W_noisy.repeat(self.S, 1, 1, 1) # [S, 1, IN, OUT], 1 is for batch dim.

        if not finite_diff:
            self.noise = torch.normal(0.0, self.stddev * torch.ones_like(self.W_noisy))
        else:
            self.noise = torch.zeros_like(self.W_noisy)
            for s in range(1, self.S): # No noise in first sample.
                i, j = s % self.W.shape[0], s // self.W.shape[0]
                self.noise[s, 0, i, j] = self.stddev

        self.noise[0] *= 0.0 # No noise added to first sample. This is needed for gradient calculation!
        self.W_noisy += self.noise

    @torch.jit.script_method 
    def forward(self, x):
        # Reshape dims to extract sample dim and batch dim.
        dims = (self.S, -1, 1, x.shape[-1])
        z = torch.matmul(x.view(dims), self.W_noisy)
        return z.reshape((x.shape[0], 1, -1))


class ODEFunc(torch.jit.ScriptModule):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 1000),
            nn.Tanh(),
            nn.Linear(1000, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    @torch.jit.script_method 
    def forward(self, t, y):
        out = self.net(y**3)
        return out

class ODEFuncSampler(nn.Module):
    def __init__(self):
        super(ODEFuncSampler, self).__init__()

        self.net = nn.Sequential(
            WeightSampler(2, 1000),
            nn.Tanh(),
            WeightSampler(1000, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        out = self.net(y**3)
        return out

def save_summary(fl_name, summary, losses, func, duration):
    ''' Appends details to summary and write to file. '''
    summary['losses'] = losses
    summary['iterations'] = len(losses)
    summary['model'] = func.state_dict()
    summary['duration'] = duration
    torch.save(summary, fl_name)

def train_bp(out_file, max_steps = args.niters, min_loss = None):
    ''' Train with backprop until we hit max_steps or min_loss. '''
    func = ODEFunc().to(device)
    optimizers = [optim.RMSprop(func.net[0].parameters(), lr=1e-3), optim.RMSprop(func.net[2].parameters(), lr=1e-3)]
    loss_record = []

    init_time = time.time()
    for itr in tqdm(range(1, max_steps)):
        active, inactive = itr % 2, (itr + 1) % 2
        batch_y0, batch_t, batch_y = get_batch()

        # Autograd Training Step
        optimizers[active].zero_grad()
        pred_y = odeint(func, batch_y0, batch_t, rtol = 1e-4, atol = 1e-4).to(device)

        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss_record.append(loss.item())
        loss.backward() # direct autodiff.
        optimizers[active].step()

        # Check stop conditions.
        if max_steps is not None and itr == max_steps:
            break

        if min_loss is not None and loss <= min_loss:
            break

    duration = time.time() - init_time

    # Save results.
    summary = { 'method': 'BP' }
    save_summary(out_file, summary, loss_record, func, duration)
    return loss_record, func, duration

def train_song(S, std, out_file, max_steps = args.niters, min_loss = None, prog_bar = False):
    ''' Train with SONG method until we hit max_steps or min_loss. '''
    func = ODEFuncSampler().to(device)
    optimizers = [optim.RMSprop(func.net[0].parameters(), lr=1e-3), optim.RMSprop(func.net[2].parameters(), lr=1e-3)]

    ws = [func.net[0], func.net[2]]
    grad_fns = [SmoothGrad(w.W.shape, std) for w in ws]
    loss_record = []

    init_time = time.time()
    itr_obj = range(1, max_steps)
    if prog_bar:
        itr_obj = tqdm(itr_obj)
    for itr in itr_obj:
        active, inactive = itr % 2, (itr + 1) % 2
        batch_y0, batch_t, batch_y = get_batch()

        # SONG Training Step
        with torch.no_grad():
            ws[active].set_params(S, std)
            ws[inactive].set_params(1, 0)

            ws[0].noisify()
            ws[1].noisify()
            grad_fns[active].enabled = True
            grad_fns[inactive].enabled = False

            batch_y0, batch_y = batch_y0.repeat(S, 1, 1), batch_y.repeat(1, S, 1, 1)
            pred_y = odeint(func, batch_y0, batch_t, rtol = 1e-4, atol = 1e-4).to(device)

            pred_y = pred_y.transpose(0,1).reshape((S,-1))
            batch_y = batch_y.transpose(0,1).reshape((S,-1))

            losses = torch.mean(torch.abs(pred_y - batch_y), -1)
            grads = [grad_fn(losses, w.noise) for grad_fn, w in zip(grad_fns, ws)]

        ws[0].W.grad = grads[0]
        ws[1].W.grad = grads[1]
        loss = losses[0]
        loss_record.append(loss.item())
        optimizers[active].step()

        # Check stop conditions.
        if max_steps is not None and itr == max_steps:
            break

        if min_loss is not None and loss <= min_loss:
            break

    duration = time.time() - init_time

    # Save results.
    summary = {
        'method': 'SONG',
        'S': S,
        'std': std, 
    }
    save_summary(out_file, summary, loss_record, func, duration)
    return loss_record, func, duration

def song_sweep(S_vals, std_vals, goal_loss = None):
    # Create saves folder.
    if not os.path.exists('saves'):
        os.mkdir('saves')

    if not os.path.exists('saves/sweep_big'):
        os.mkdir('saves/sweep_big')

    # Evaluate backpropagation best loss. If passed to function, measures duration to this loss.
    losses, _, _ = train_bp('saves/sweep2/bp.pt', args.niters, goal_loss)
    goal_loss = losses[-1] 
    print(f'BP Goal Loss: {goal_loss}')

    # Run sweep.
    for S in S_vals:
        print(S)
        for std in tqdm(std_vals):
            out_file = f'saves/sweep_big/song_{S}_{std}.pt'
            train_song(S, std, out_file, min_loss = goal_loss, max_steps = 400, prog_bar = False)

def average_loss(loss, n_avg = 40, truncate = False):
    ''' Run a sliding window of length n_avg over loss to compute averaged loss curve. '''
    out = []
    for i in range(len(loss)):
        end = min(i + n_avg, len(loss))
        if truncate and end < i + n_avg:
            break # Only average when window contains n_avg things.
        out.append(np.mean(loss[i:end]))
    return np.array(out)

if __name__ == '__main__':
    plot_sweep = False 
    max_steps = 400
    if plot_sweep:
        # Run sweep. Change False to True to rerun.
        if False:
            S_vals = (10**np.linspace(np.log10(20.), np.log10(2000.), 50)).astype(int)
            std_vals = 10**np.linspace(-1, -4, 20)
            song_sweep(S_vals, std_vals)

        for quantity in ['iterations', 'duration']:
            fls = glob.glob('saves/sweep_big/song*')
            S_vals, durations, losses, std_vals = [], [], [], []
            print("Loading Data ... ")

            for fl in tqdm(fls):
                summary = torch.load(fl) 
                if summary['iterations'] == max_steps-1:
                    continue

                S_vals.append(summary['S'])
                std_vals.append(summary['std'])
                durations.append(summary[quantity])
                losses.append(summary['losses'])

            sweep_dict = {S:[[], []] for S in np.unique(S_vals)}
            for i in range(len(S_vals)):
                sweep_dict[S_vals[i]][0].append(durations[i])
                sweep_dict[S_vals[i]][1].append(std_vals[i])

            S_vals = sweep_dict.keys()
            best_indices = [np.argmin(l[0]) for l in sweep_dict.values()]
            best_std = [l[1][idx] for l, idx in zip(sweep_dict.values(), best_indices)]
            best_durations = [l[0][idx] for l, idx in zip(sweep_dict.values(), best_indices)]

            bp_fl = glob.glob('saves/sweep/bp*')[0]
            bp_duration = torch.load(bp_fl)[quantity]
            bp_loss = torch.load(bp_fl)['losses']

            avg_losses = [average_loss(loss, 70) for loss in losses]

            ymin, ymax = min(bp_loss), max([max(l) for l in avg_losses])

            # Compare runtimes.
            data_sorted = np.array(sorted(zip(S_vals, best_durations, best_std)))
            S_vals, durations, stds = data_sorted[:, 0], data_sorted[:, 1], data_sorted[:, 2]

            plt.figure(figsize=(5, 3))
            plt.plot(S_vals, durations, 'x', c = 'blue')

            plt.axhline(bp_duration, linestyle=':', c = 'black', markersize=2, linewidth=2)
            plt.axvline(2000, linestyle='--', c = 'black', markersize=2)
            format_quant = 'Iterations' if quantity == 'iterations' else 'Runtime'
            plt.ylabel(format_quant)
            plt.xlabel('Number of Samples')
            plt.legend([f'SONG {format_quant}', f'BP {format_quant}'])


            # Fit reciprcol linearly.
            recip = 1./durations
            m, b = np.polyfit(S_vals, recip, deg=1)
            b *= 0.0
            plt.plot(S_vals, 1./(m*S_vals+b), c = 'darkred', zorder=-1, linewidth = 3, alpha = 0.5)

            plt.ylim(np.min(durations) * 0.9, np.max(durations) * 1.1)
            plt.tight_layout()

            plt.savefig(f'{quantity}_v_samples_nde.pdf')

        plt.figure(figsize=(8, 4))
        plt.plot(S_vals, stds, 'x', c = 'blue')
        plt.ylabel('Best $\\sigma$')
        plt.xlabel('Number of Samples')
        plt.yscale('log')

        plt.suptitle('4,000 Parameters')
        plt.tight_layout()

        plt.savefig('best_variance_nde.pdf')
        plt.show()

        # Plot all loss curves.
        plt.figure(figsize = (6, 4))
        cmap = mpl.cm.get_cmap('Reds')
        bp_col = [0.0, 0.0, 1.0] # Invert final color
        plt.axvline(bp_duration, c = bp_col, linewidth=2.0)
        plt.text(bp_duration * 1.2, ymax * 1.05, 'BP Iterations', c = bp_col)
        norm = mpl.colors.LogNorm(min(S_vals), max(S_vals))
        for i in range(len(losses)):
            S = S_vals[i]
            plt.plot(avg_losses[i], c = cmap(norm(S)), zorder=10, linewidth=2)

        plt.xscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MAE)')
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        cb.set_label('# of Samples')
        plt.ylim(ymin, ymax * 1.1)
        plt.savefig('Time_to_Loss.pdf')
        plt.show()

    plot_acc = True 
    if plot_acc:
        losses_song_all = []
        losses_bp_all = []
        model = None
        for run in range(6):
            print('Run : ', run)
            fname1, fname2 = f'saves/song_acc_{run}.pt', f'saves/bp_acc_{run}.pt'
            if not os.path.exists(fname1):
                train_song(500, 1e-2, fname1, prog_bar = True)

            if not os.path.exists(fname2):
                train_bp(fname2)

            losses_song_all.append(torch.load(fname1)['losses'])
            losses_bp_all.append(torch.load(fname2)['losses'])
            model = torch.load(fname2)['model']

        # Take mean and std over multiple runs. 
        losses_song = np.mean(losses_song_all, 0)
        losses_bp = np.mean(losses_bp_all, 0)

        std_song = np.std(losses_song_all, 0)
        std_bp = np.std(losses_bp_all, 0)

        # Smooth over iterations with a small window to filter out some noise.
        losses_song = average_loss(losses_song, 10, truncate=True)
        losses_bp = average_loss(losses_bp, 10, truncate=True)
        std_song = average_loss(std_song, 10, truncate=True)
        std_bp = average_loss(std_bp, 10, truncate=True)

        plt.figure(figsize=(6,4))
        ax = plt.subplot(111)
        X = range(len(losses_song))

        # Mean lines
        ax.plot(X, losses_song, c = 'red', zorder = 10) 
        ax.plot(X, losses_bp, c = 'black', zorder = 10)

        # Std area
        ax.fill_between(X, losses_song - std_song, losses_song + std_song, color = 'red', alpha = 0.5, zorder = 0, linewidth=0)
        ax.fill_between(X, losses_bp - std_bp, losses_bp + std_bp, color = 'black', alpha = 0.5, zorder=0, linewidth=0)    

        ax.legend(['SONG', 'BP'])
        ax.set_ylabel('Loss (MAE)')
        ax.set_xlabel('Iteration')
        ax.set_xlim([0, len(losses_song)])
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('acc_compare_nde.pdf')
        plt.show()


        # Plot example flow.
        func = ODEFunc().cuda()
        func.load_state_dict(model)

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = func(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        plt.figure(figsize=(6,4))
        ax = plt.subplot(111)
        ax.set_title('Neural ODE')
        ax.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax.axis('off')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plt.savefig('flow_nde_example.pdf')
