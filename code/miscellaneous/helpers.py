import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import os
import numpy as np

def StyblinskiTangNN(X):
    return (0.5 * (torch.pow(X,4) - 16 * torch.pow(X,2) + 5 * X)).sum(axis=-1)

class StyblinskiTangNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return (0.5 * (torch.pow(X,4) - 16 * torch.pow(X,2) + 5 * X)).sum(axis=-1)

def StyblinskiTang(X):
    return (0.5 * (torch.pow(X,4) - 16 * torch.pow(X,2) + 5 * X)).sum(axis=0)


def PlotStyblinskiTang(X, Y, Z, title):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap="viridis",
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-100, 250)
    ax.zaxis.set_tick_params(pad=8)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    plt.suptitle("Styblinski Tang function")

    plt.savefig(title)
    
def plot_results(npts,nb_epochs,learning_rate,sobolev_weights, out, out_sobolev):

    predict_fn, list_loss = out[0], out[1]
    predict_fn_S, list_loss_S, list_loss_J_S = out_sobolev[0],out_sobolev[1],out_sobolev[2]

    # Create a mesh on which to evaluate pred_fn
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)

    """xx = X.ravel().reshape(-1, 1)
    yy = Y.ravel().reshape(-1, 1)

    inputs = np.concatenate((xx, yy), axis=1).astype(np.float32)
    Z = predict_fn(inputs).reshape(X.shape)
    Z_S = predict_fn_S(inputs).reshape(X.shape)"""

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2,2, wspace=0.3)

    # Plot for standard network
    ax = fig.add_subplot(gs[0], projection='3d')
    ax.plot_surface(X, Y, predict_fn, cmap="viridis",
                    linewidth=0, antialiased=False)
    ax.set_zlim(-100, 250)
    ax.zaxis.set_tick_params(pad=8)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Standard network (%s pts)" % npts, fontsize=22)

    ax = plt.subplot(gs[1])
    ax.plot(list_loss, linewidth=2, label="MSE loss")
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("MSE loss", fontsize=20)
    ax.set_ylim([1, 2E4])
    ax.set_yscale("log")
    ax.set_title("Training loss standard network", fontsize=22)
    ax.legend(loc="best", fontsize=20)

    # Plot for Sobolev network
    ax = fig.add_subplot(gs[2], projection='3d')
    ax.plot_surface(X, Y, predict_fn_S, cmap="viridis",
                    linewidth=0, antialiased=False)
    ax.set_zlim(-100, 250)
    ax.zaxis.set_tick_params(pad=8)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Sobolev network (%s pts)" % npts, fontsize=22)

    ax = plt.subplot(gs[3])
    ax.plot(list_loss_S, linewidth=2, label="MSE loss")
    ax.plot(list_loss_J_S, linewidth=2, label="Sobolev loss")
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("MSE loss ", fontsize=20)
    ax.set_ylim([1, 2E4])
    ax.set_yscale("log")
    ax.set_title("Training loss Sobolev network", fontsize=22)
    ax.legend(loc="best", fontsize=20)

    if not os.path.exists("figures"):
        os.makedirs("figures")

    fig_name = "plot_%s_epochs_%s_npts_%s_LR_%s_sobolev_weight.png" % (nb_epochs,
                                                                       npts,
                                                                       learning_rate,
                                                                       sobolev_weights)

    plt.savefig(os.path.join("figures", fig_name))
    plt.clf()
    plt.close()