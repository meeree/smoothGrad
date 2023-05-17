import argparse
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
import os
import time
from torch import jit

torch.set_default_dtype(torch.float64)

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


class ML(nn.Module):
    def __init__(self, L, softmax = True, adjoint=True, bias = True):
        super().__init__()
        self.L = L
        self.V = torch.ones(L) * -40.0
        self.w = torch.ones(L) * 0.014173
        self.Vt = torch.tensor([20.0])
        self.Kp = torch.tensor([100.0])
        self.W = nn.Parameter(torch.ones(L))
        self.softmax = softmax
        self.backwards = False

        self.El = torch.normal(torch.zeros(L) - 55., 2.5)

        self.odeint = odeint_adjoint if adjoint else odeint
    
    def compute_output(self, V):
        ''' Compute "output" of neuron given its voltage '''
        return torch.sigmoid((V - self.Vt) * self.Kp)

    def forward(self, t, state):
        V, w = state
        T = self.compute_output(V)
        z = self.W * self.Iapp

        phi, V1, V2, V3, V4, c = 2./30., -1.2, 18., 12., 17., 20.
        gca, Eca, gk, Ek, gl = 4., 120., 8., -84., 2.
        El = self.El 

        minf = .5 * (1 + torch.tanh((V - V1) / V2))
        winf = .5 * (1 + torch.tanh((V - V3) / V4))
        tauw = 1. / (torch.cosh((V - V3) / (2 * V4)))
        dVdt = z - gca * minf * (V - Eca) - gk * w * (V - Ek) - gl * (V - El)
        dwdt = phi * (winf - w) / tauw
        if self.backwards:
            dVdt, dwdt = -dVdt, -dwdt # Backwards in time!
        return dVdt, dwdt

    def load_state_from_file(self, fl, bsize):
        fl_v, fl_w = fl + '_V.pt', fl + '_w.pt'
        if os.path.exists(fl_v) and os.path.exists(fl_w):
            V, w = torch.load(fl_v), torch.load(fl_w)
        else:
            V = torch.normal(torch.ones(self.L) * -40.0, 5.0)
            w = torch.normal(torch.ones(self.L) * 0.5, 0.1)
            torch.save(V, fl_v)
            torch.save(w, fl_w)
        V = V.repeat(bsize, 1)
        w = w.repeat(bsize, 1)
        return (V, w)

    def simulate(self, Iapp, tt, tol = 1e-5, bsize=1, method = 'rk4'):
        self.Iapp = Iapp
        state = self.load_state_from_file(f'init_conds_{self.L}', bsize)

        # Simulate V and w ODE states over time.
        self.Vs, self.ws = self.odeint(self, state, tt, atol=tol, rtol=tol, method=method)

        # Compute outputs of neurons given voltages.
        self.Ts = self.compute_output(self.Vs)
        return self.Ts

    def sim_back_forth(self, Iapp, tt, tol = 1e-5, bsize=1):
        self.Iapp = Iapp
        state = self.load_state_from_file(f'init_conds_{self.L}', bsize)

        # Simulate forwards then simulate backwards form forwards.
        self.Vs, self.ws = self.odeint(self, state, tt, atol=tol, rtol=tol, method='rk4')
        state = (self.Vs[-1], self.ws[-1])
        tt = torch.flip(tt, [0])
        self.backwards = True
        self.Vs_back, self.ws_back = self.odeint(self, state, tt, atol=tol, rtol=tol, method='rk4')
        self.backwards = False
        return self.Vs, self.Vs_back


def adjoint_calculate(t, y, func, method, tol):
    ''' Adapted from torchdiffeq so that I can plot and save values of adjoint. 
        Input is timesteps for eval, state over time (outputted from solver) and solver parameters. '''
    with torch.no_grad():
        adjoint_rtol, adjoint_atol = tol, tol 
        adjoint_method = method
        adjoint_params = tuple(list(func.parameters()))
        grad_y = torch.zeros_like(y)
        grad_y[:, 0] = 1 / len(t) # Loss is mean, so need 1 / len

        ##################################
        #      Set up initial state      #
        ##################################

        # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
        aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
        aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

        ##################################
        #    Set up backward ODE func    #
        ##################################

        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = y_aug[1]
            adj_y = y_aug[2]
            # ignore gradients wrt time and parameters

            with torch.enable_grad():
                t_ = t.detach()
                t = t_.requires_grad_(True)
                y = y.detach().requires_grad_(True)

                # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                # wrt t here means we won't compute that if we don't need it.
                func_eval = func(t, y) # RETURNS TUPLE, NEED TO CONCAT. SEE BELOW.
                cat = [x.reshape(-1) for x in func_eval]
                func_eval = torch.cat(cat)

                # Workaround for PyTorch bug #39784
                _t = torch.as_strided(t, (), ())  # noqa
                _y = torch.as_strided(y, (), ())  # noqa
                _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

                vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                    func_eval, (t, y) + adjoint_params, -adj_y,
                    allow_unused=True, retain_graph=True
                )

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
            vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                          for param, vjp_param in zip(adjoint_params, vjp_params)]

            return (vjp_t, func_eval, vjp_y, *vjp_params)

        ##################################
        #       Solve adjoint ODE        #
        ##################################

        # RECORD AUGMENTED SYSTEM OVER ODE.
        record = []
        time_vjps = None
        record.append(aug_state)
        for i in range(len(t) - 1, 0, -1):
            # Run the augmented system backwards in time.
            aug_state = odeint(
                augmented_dynamics, tuple(aug_state),
                t[i - 1:i + 1].flip(0),
                rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method
            )
            aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
            aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
            aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point
            record.append(aug_state)

    return record



def get_loss(model, tt):
    Ts = model.simulate(100.0, tt, method = 'bosh3', tol = 1e-3)
    Vs = model.Vs
    loss = torch.mean(Ts, 0).squeeze() # Mean over time
    return loss, Vs

def train_ml_turn_off(timeframe, bp = True):
    ''' Train Morris Lecar neuron to turn off. '''
    tt = torch.linspace(0.0, timeframe, int(timeframe)) # Samples to plot. Sample every ms.

    fname = f'saves/morris_lecar_train_{timeframe}_{"bp" if bp else "song"}.pt'
    torch.manual_seed(0)
    S = 100
    L = 1 if bp else S
    model = ML(L)
    model.softmax = False
    model.W.data = torch.ones_like(model.W.data) * 0.5

    lr = 1e-2
    loss_out = []
    ws = []
    from tqdm import tqdm
    plt.figure(figsize=(12,5))
    ax = plt.subplot(121)
    ax2 = plt.subplot(122)
    for itr in tqdm(range(50)):
        if not bp: # Song setup
            model.W.data[1:] = torch.normal(model.W.data[0] + torch.zeros_like(model.W[1:].data), 0.2)

        losses, Vs = get_loss(model, tt)
        loss = losses if bp else losses[0]

        if bp:
            loss.backward()
            model.W.data -= lr * model.W.grad # Gradient descent step.
        else:
            model.W.data[0] -= lr * torch.mean(losses * (model.W.data - model.W.data[0]) / 0.2**2)
            ax2.cla()
            ax2.plot(model.W.data, losses.detach(), 'o')
            plt.pause(0.001)


        loss_out.append(loss.item())
        ws.append(model.W[0].detach().item())

        ax.cla()
        ax.plot(loss_out)
        plt.pause(0.001)

    output = {
        'times': tt,
        'losses': loss_out,
        'ws': ws
    }
    torch.save(output, fname)

#train_ml_turn_off(500, True) # BP
#train_ml_turn_off(500, False) # SONG

def run_and_save(timeframe, W_vals = [1.0], compute_grad = True):
    ''' Simulate model for times [0, timeframe] and save checkpoint to file including adjoints and gradients over timesteps. '''
    tt = torch.linspace(0.0, timeframe, int(timeframe)) # Samples to plot. Sample every ms.

    torch.manual_seed(0)
    L = len(W_vals)
    model = ML(L)
    model.softmax = False
    model.W.data = torch.tensor(W_vals)
    loss, Vs = get_loss(model, tt)

    fname = f'saves/morris_lecar_voltage_{timeframe}'
    if len(W_vals) > 1:
        fname += '_varied_W'
    fname += '.pt'

    output = {
        'times': tt,
        'loss': loss, 
        'voltages': Vs,
        'W_vals': W_vals,
    }

    if not compute_grad:
        torch.save(output, fname)
        return

    Vs, ws = model.Vs.reshape((-1, 1)), model.ws.reshape((-1, 1))
    ys = torch.cat([Vs, ws], 1)
    record = adjoint_calculate(tt, ys, model, 'bosh3', 1e-3)

    adjoints = torch.tensor([a[2][1].detach() for a in record])
    grads = torch.tensor([a[3][0].detach() for a in record])

    # Reverse so times align (these are in reverse time).
    adjoints, grads = adjoints.flip(0), grads.flip(0)

    # Write to file.
    output['adjoints'] = adjoints
    output['grads'] = grads 
    torch.save(output, fname)


#run_and_save(5000, compute_grad = True)
#run_and_save(10000, W_vals = torch.linspace(0.5, 1.5, 1000), compute_grad = False)

def plot_loss_landscape(fname):
    plt.figure(figsize=(8,4))
    ax = plt.subplot(111)
    save = torch.load(fname)
    loss = save['loss'].detach().numpy()
    W_vals = save['W_vals'].detach().numpy()
    mid = len(W_vals) // 2

    smooth_loss = []
    W = 20
    for i in range(W//2, len(loss)-W//2):
        blob = norm.pdf(W_vals, loc = W_vals[i], scale=0.003)
        smooth_loss.append(np.mean(blob * loss))

    smooth_loss = np.array(smooth_loss)

    # Clips range to convolved smoothed loss range.
    W_vals = W_vals[W//2:-W//2]
    loss = loss[W//2:-W//2]

    ax.plot(W_vals, loss, zorder=0, c = 'black')
    ax.set_xlabel('$I_{app}$')
    ax.set_ylabel('Loss')

    ax.plot(W_vals, smooth_loss, zorder=1, c = 'red', linewidth=3)
    ax.set_yticks([])

    ax3 = plt.gca().twinx().twiny()
    X = np.linspace(-2.0, 2.0, 1000)
    gaussian = norm.pdf(X, scale = 0.4)
    ax3.fill_between(X, 0 * gaussian, gaussian, color = 'red', alpha = 0.2, zorder = -1, linewidth = 0)
    ax3.set_ylim(0, np.max(gaussian) * 1.5)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.savefig('loss_landscape_turn_off.pdf')

    plt.figure(figsize=(8,4))
    dw = W_vals[1] - W_vals[0]
    dldw = (loss[1:] - loss[:-1]) / dw
    dsldw = (smooth_loss[1:] - smooth_loss[:-1]) / dw

    plt.plot(W_vals[:-1], dldw, c = 'black', zorder=0)
    plt.plot(W_vals[:-1], dsldw, c = 'red', zorder=1, linewidth=3)
    plt.savefig('smooth_loss_gradients.pdf')

def plot_train_comparison(fname1, fname2):
    losses_bp = torch.load(fname1)['losses']
    losses_song = torch.load(fname2)['losses']

    plt.figure(figsize=(8,4))
    ax = plt.subplot(111)

    ax.plot(losses_bp, zorder=0, c = 'black')
    ax.plot(losses_song, zorder=1, c = 'red', linewidth=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('"Minimize Output" Loss')

    legend = ax.legend(['BP', 'SONG'])
    legend.get_frame().set_alpha(0)

    plt.savefig('train_comparison_ml.pdf')
    plt.show()


plot_loss_landscape('saves/morris_lecar_voltage_3000_varied_W.pt')
plot_train_comparison('saves/morris_lecar_train_500_bp.pt', 'saves/morris_lecar_train_500_song.pt')

exit() 

def plot_voltage(fname, ax, time_end = -1):
    save = torch.load(fname)
    tt = save['times'].detach().numpy()
    if time_end < 0:
        time_end = len(tt)
    tt = tt[:time_end]
    Vs = save['voltages'][:time_end, 0].detach().numpy()
    ax.plot(tt, Vs, c = 'black')

def plot_adjoints(fname, ax, time_end = -1):
    save = torch.load(fname)
    tt = save['times'].detach().numpy()
    if time_end < 0:
        time_end = len(tt)
    tt = tt[:time_end]
    adjoints = save['adjoints'][:time_end].detach().numpy()
    grads = save['grads'][:time_end].detach().numpy()
    ax.plot(tt, adjoints)
    ax.plot(tt, grads)

for time_end, suffix in zip([-1, 500], ['zoom', 'full']):
    plt.figure(figsize=(5,4))
    ax = plt.subplot(211)
    plot_voltage('saves/morris_lecar_voltage_3000.pt', ax, time_end)
    plt.ylabel('Voltage (mV)')
    ax = plt.subplot(212)
    plot_adjoints('saves/morris_lecar_voltage_3000.pt', ax, time_end)
    plt.xlabel('Time (ms)')
    legend = plt.legend(['Adjoint, $\\partial L / \\partial V(t)$', 'Gradient, $\\partial L(t) / \\partial I_0$'])
    legend.get_frame().set_alpha(0)
    plt.tight_layout()
    plt.savefig(f'figures/adjoint_{suffix}.pdf')

plt.show()
exit()

#
#def landscape_plot(timeframe):
#    tt = torch.linspace(0.0, timeframe, int(timeframe)) # Samples to plot. Sample every ms.
#    Nw = 90 
#
#    torch.manual_seed(0)
#    model = ML(Nw)
#    model.softmax = False
#
#    w0 = model.w[0].item()
#    ws = torch.linspace(w0 - 0.01, w0 + 0.01, Nw)
#    model.w.data = ws
#
#    # Loss at all values of w.
#    losses_all = None 
#    Vs_all = None
#    losses, Vs = get_loss(model, tt)
#
#    plt.figure(figsize = (4,4))
#    plt.subplot(211)
#    plt.imshow(Vs[:, 0, :].detach().numpy().T, aspect='auto')
#    plt.yticks([0, Nw-1], [ws[0].item(), ws[-1].item()])
#    plt.ylabel('w')
#
#    plt.subplot(212)
#    plt.plot(ws, losses.detach())
#    plt.show()
#
##landscape_plot(1000)
##exit()
#
#def test_ml(L, timeframe, compare_ml = True):
#    torch.manual_seed(0)
#    model = ML(L)
#    model.softmax = False
#
#    tt = torch.linspace(0.0, timeframe, int(timeframe)) # Samples to plot. Sample every ms.
#    if not compare_ml and os.path.exists('adjoints.npy'):
#        with open('adjoints.npy', 'rb') as f:
#            adjoints = np.load(f)
#
#        with open('voltage.npy', 'rb') as f:
#            Vs = np.load(f)
#    else:
#        loss, Vs = get_loss(model, tt)
#        loss.backward()
#        grad_adjoint = model.w.grad
#        print(grad_adjoint)
#
#        Vs = Vs.detach().numpy()[:, 0, :]
#        with open('voltage.npy', 'wb') as f:
#            np.save(f, Vs)
#
#        # Use finite difference to compute derivative.
#        if compare_ml:
#            dw = 0.1
#            model.w.detach()[0,0] += dw
#            loss_perb, _ = get_loss(model)
#            grad_fd = (loss_perb - loss) / dw
#            print(grad_adjoint, grad_fd)
#
#    adjoints = adjoints[:, 1]
#    adjoints = np.concatenate(([0], adjoints)) # Add zero at start.
#
#    plt.figure(figsize=(2,4))
#    plt.plot(adjoints)
#    plt.plot(Vs)
#    plt.show()
##    npts = 100
##    for i in range(0, len(Vs), npts):
##        plt.scatter(Vs[i:i+npts], adjoints[i:i+npts], c = [[i / len(Vs), 0, 0]], s=0.4)
#
#    plt.subplot(211)
#    plt.imshow(Vs[2740:2900].reshape((1, -1)), aspect='auto', cmap='hot')
#    plt.colorbar()
#    plt.subplot(212)
#    plt.imshow(adjoints[2740:2900].reshape((1, -1)), aspect='auto', cmap='seismic')
#    plt.colorbar()
#    plt.show()
#
#    fig = plt.figure(figsize=(5,6))
#     
#    tt = tt / 1000 # Convert ms to s
#
#    plt.subplot(3, 2, (1,2))
#    plt.plot(tt[:108], Vs[:108], c='black')
#    plt.xlim(tt[0], tt[117])
#    plt.xlabel('Time (s)')
#    plt.title('Neural Activity')
#    plt.gca().spines[['right', 'top']].set_visible(False)
#
#    plt.subplot(3, 2, (3,4))
#    plt.plot(tt, adjoints, c = 'red', linewidth=1.0)
#    legend = plt.legend(['Adjoint Activity'], fontsize=BIGGER_SIZE)
#    legend.get_frame().set_alpha(0)
#    plt.gca().spines[['right', 'top']].set_visible(False)
#
#
#    zooms = [
#        [2740, 2900], # Regions where instability is clear.
#        [600, 2250] 
#    ]
#    i = 0
#    for zoom in zooms:
#        start, end = zoom
#        plt.subplot(3, 2, 5 + i)
#        plt.plot(tt[start:end], adjoints[start:end], c='red')
#        plt.gca().spines[['right', 'top']].set_visible(False)
#
#        plt.gca().twinx()
#        plt.gca().spines[['right', 'top']].set_visible(False)
#        plt.plot(tt[start:end], Vs[start:end], c ='black')
#        i += 1
#
#    plt.tight_layout()
#    plt.savefig('motivating_fig.pdf')
#    plt.show()
#
#test_ml(1, 3000, True)
