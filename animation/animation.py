import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


params = {'text.usetex': True, 
          'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)


def f(t: float) -> float:
    if -0.5 <= t <= 0.5:
        return 1. - (t + 0.5)
    return 0.

def g(t: float) -> float:
    if -0.5 <= t <= 0.5:
        return 1.
    else:
        return 0.

x = np.arange(-2., 2., 0.001)

y_f = np.array([f(t) for t in x])
y_g = np.array([g(t) for t in x])
y_f_fl = np.flip(y_f)

conv = np.convolve(y_f, y_g, mode='same') / 1000.
corr = np.correlate(y_g, y_f, mode='same') / 1000.

frames = 301
def animate_conv(i):
    t = -1.5 + 3.*i / (frames - 1)
    plt.clf()

    j = -1500 + i*3000// (frames - 1)
    if j < 0:
        flipped = np.pad(y_f_fl[-j : ], (0, -j))
    elif j == 0:
        flipped = y_f_fl
    else:
        flipped = np.pad(y_f_fl[ : -j], (j, 0))
    
    plt.plot(x, flipped, '--', label="$f(x - \\tau)$")
    plt.fill_between(x, np.zeros(len(y_f)), flipped * y_g, alpha=0.2)

    plt.plot(x, y_g, '--', label="$g(\\tau)$")

    plt.plot(x[:min(len(y_f), 2000 + j)], conv[: min(len(y_f), 2000 + j)], label="$f * g$")


    plt.title("Faltung $(f*g)(x)= \\int_{\\mathbb{R}} f(x - \\tau) \\cdot g(\\tau) \\,\\mathrm{d}\\tau$")
    plt.yticks([0., 1.])
    plt.xticks([t], labels=["$x$"])
    plt.axis([-1.5, 1.5, -0.2, 1.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    return fig,

def animate_corr(i):
    t = -1.5 + 3.*i / (frames - 1)
    plt.clf()

    j = -1500 + i*3000// (frames - 1)
    if j < 0:
        padded = np.pad(y_f[-j : ], (0, -j))
    elif j == 0:
        padded = y_f
    else:
        padded = np.pad(y_f[ : -j], (j, 0))
    
    plt.plot(x, padded, '--', label="$f(x - \\tau)$")
    plt.fill_between(x, np.zeros(len(y_f)), padded * y_g, alpha=0.2)
    plt.plot(x, y_g, '--', label="$g(\\tau)$")
    plt.plot(x[:min(len(y_f), 2000 + j)], corr[: min(len(y_f), 2000 + j)], label="$f \star g$")
    
    plt.title("Kreuzkorrelation $(f\\star g)(x) = \\int_{\\mathbb{R}} f(x - \\tau) \\cdot g(\\tau) \\,\\mathrm{d}\\tau$")
    plt.yticks([0., 1.])
    plt.xticks([t], labels=["$x$"])
    plt.axis([-1.5, 1.5, -0.2, 1.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    return fig,

def save_f():
    plt.clf()
    plt.plot(x, y_f, '-', label="$f$")
    plt.yticks([0., 1.])
    plt.axis([-1.5, 1.5, -0.2, 1.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    fig.set_size_inches(6.4, 3.2, True)
    plt.savefig("f.png", dpi=200)
    plt.close()

def save_g():
    plt.clf()
    plt.plot(x, y_g, '-', label="$g$", color='#ff7f0e')
    plt.yticks([0., 1.])
    plt.axis([-1.5, 1.5, -0.2, 1.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    fig.set_size_inches(6.4, 3.2, True)
    plt.savefig("g.png", dpi=200)
    plt.close()


fig = plt.figure()
save_g()
fig = plt.figure()
save_f()
#


fig = plt.figure()
anim = animation.FuncAnimation(fig, animate_corr, frames=frames, interval=20, blit=True)
fig.set_size_inches(6.4, 3.2, True)
anim.save('cross-correlation.mp4', fps=30, extra_args=['-vcodec', 'libx264'], dpi=200)

anim = animation.FuncAnimation(fig, animate_conv, frames=frames, interval=20, blit=True)
fig.set_size_inches(6.4, 3.2, True)
anim.save('convolution.mp4', fps=30, extra_args=['-vcodec', 'libx264'], dpi=200)
