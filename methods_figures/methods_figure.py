from matplotlib import pyplot as plt
import numpy as np

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


X = np.linspace(-2, 2, 1000)
f = lambda x: x ** 3 + 3 * x**2
loss = f(X)
l0 = np.copy(loss)
loss = ((loss * 0.4).astype(int) / 0.4).astype(float) # Make jagged
loss = loss + np.random.normal(np.zeros_like(loss), 0.4)

std = 0.3
gaussian = (1 / (std * np.sqrt(2*np.pi))) * np.exp(-0.5 * (X / std)**2)

plt.figure(figsize = (12,2))
plt.subplot(121)
plt.plot(X, loss, zorder = 1, linewidth = 1, c = 'black')
plt.plot(X, f(X), c = 'red', linewidth = 3, zorder=0)
legend = plt.legend(['Loss, $L(x)$', 'Smooth Loss, $L_p(x)$'])
legend.get_frame().set_alpha(0)
plt.xlabel('Input $x$')
plt.xlim(np.min(X), np.max(X))
plt.axis('off')
ax2 = plt.gca().twinx()
plt.fill_between(X, 0 * gaussian, gaussian, color = 'red', alpha = 0.2, zorder = 0, linewidth = 0)
plt.ylim(0, np.max(gaussian) * 2)
plt.xlim(np.min(X), np.max(X))
plt.axis('off')


plt.subplot(122)
dx = X[1] - X[0]
dldx = (loss[1:] - loss[:-1]) / dx
dl0dx = (l0[1:] - l0[:-1]) / dx
f_prime = lambda x: 3 * x**2 + 6 * x
plt.plot(X[:-1], dldx, linewidth = 1, zorder = 0, c = 'black')
plt.plot(X[:-1], f_prime(X[:-1]), c = 'red', linewidth = 3, zorder = 1)
plt.axis('off')
legend = plt.legend(['Loss Gradient, $\\nabla_{\\theta} L(\\theta)$', 'Smooth Gradient, $\\nabla_{\\theta} L_p(\\theta)$'])
legend.get_frame().set_alpha(0)
plt.xlim(np.min(X), np.max(X[:-1]))
plt.savefig('smooth_loss_methods.pdf')

plt.figure(figsize = (4,4))
X = np.linspace(0.0, 1.0, 100)
grid = np.zeros((100, 100))
f = lambda x, y: (x - 0.3)**2 + (y - 0.3)**2
def f_jagged(x, y):
    x, y = x + np.random.normal() * 0.02, y + np.random.normal() * 0.02
    x, y = float(int(x * 50) / 50), float(int(y * 50) / 50)
    return f(x,y)

f_prime = lambda x, y: np.array([2 * (x - 0.3), 2 * (y - 0.3)])
for i in range(100):
    for j in range(100):
        grid[i,j] = f_jagged(X[i], X[j])

grid_smooth = np.zeros_like(grid)
for i in range(100):
    for j in range(100):
        grid_smooth[i,j] = f(X[i], X[j])


std = 0.1
np.random.seed(7)
center = np.array([0.35, 0.7])
smpl_centers = np.zeros((30, 2)) + center
smpls = np.random.normal(smpl_centers, std)

plt.imshow(grid, cmap = 'Blues', interpolation='none', aspect='auto', zorder=-1)
plt.contour(grid, colors = 'black', zorder=1)
plt.contour(grid_smooth, colors = 'red', zorder = 0, linewidths = 3)
plt.axis('off')
ax2 = plt.gca().twinx().twiny()

losses = np.array([f(smpl[0], smpl[1]) for smpl in smpls])

smpls = smpls[::2]
plt.scatter(smpls[:, 0], smpls[:, 1], marker='x', c = 'red', zorder=1)
for smpl in smpls:
    plt.arrow(center[0], center[1], smpl[0] - center[0], smpl[1] - center[1], color = 'black', zorder=0, linewidth = 0.4)

legend = plt.legend(['Smooth Loss Samples', 'Scores'])
legend.get_frame().set_alpha(0)

plt.xlim(X[0], X[-1])
plt.ylim(X[0], X[-1])
plt.xticks([])
plt.yticks([])
plt.savefig('sampling_method.pdf')
plt.show()
