import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import imageio
import os


def get_midpt_and_range(lims):
    midpt = (lims[1] + lims[0])/2.
    span  = abs(lims[1] - lims[0])
    return midpt, span

def equal_axes(ax):
    x_m, x_r = get_midpt_and_range(ax.get_xlim3d())
    y_m, y_r = get_midpt_and_range(ax.get_ylim3d())
    z_m, z_r = get_midpt_and_range(ax.get_zlim3d())

    r = max([x_r, y_r, z_r])/2.
    ax.set_xlim3d([x_m - r, x_m + r])
    ax.set_ylim3d([y_m - r, y_m + r])
    ax.set_zlim3d([z_m - r, z_m + r])


def plot_field(xyz, field, cmap="coolwarm", title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, cmap=cmap)
    equal_axes(ax)
    plt.title(title)
    #ax.view_init(10,60)
    plt.show()

def plot_fields(xyz, fields, titles = None, cmap="coolwarm"):
    if titles is not None:
        assert len(fields) == len(titles)
    N = len(fields)

    vmin = min([min(f) for f in fields])
    vmax = max([max(f) for f in fields])

    fig = plt.figure(figsize=(3.5*N,4),dpi=300)
    for i, field in enumerate(fields):
        ax = fig.add_subplot(1, N, i+1, projection='3d')
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, cmap=cmap, vmin = vmin, vmax = vmax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        equal_axes(ax)
        plt.title(titles[i])

    plt.show()

import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib as mpl

br_map = ListedColormap(["blue","red"],"bluered")
mpl.colormaps.register(br_map, name="bluered")



def plot_fields_custom(xyz, fields, titles = None, cmaps="coolwarm", limgroups = None):
    if titles is not None:
        assert len(fields) == len(titles)
    N = len(fields)

    if type(cmaps) == list:
        assert N == len(cmaps)
    else:
        cmaps = [cmaps,]*N

    if limgroups == None:
        limgroups = [0,]*N
    else:
        assert N == len(limgroups)

    fig = plt.figure(figsize=(3.5*N,3),dpi=300)
    width_ratios = []
    for i in range(N):
        if i == N-1 or limgroups[i] not in limgroups[i+1:]:
            width_ratios.append(1.25)
        else:
            width_ratios.append(1.)
    G = gridspec.GridSpec(1,N, width_ratios = width_ratios)


    for i, field in enumerate(fields):

        vmin = min([min(fields[j]) for j in range(len(fields)) if limgroups[j] == limgroups[i]])
        vmax = max([max(fields[j]) for j in range(len(fields)) if limgroups[j] == limgroups[i]])

        ax = fig.add_subplot(G[-1,i], projection='3d')
        scatter = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2], c=field, cmap=cmaps[i], vmin = vmin, vmax = vmax)
        if width_ratios[i] > 1.1 and cmaps[i] != "bluered":
            cbar = fig.colorbar(scatter, format="% .2f", shrink=0.8)
        elif width_ratios[i] > 1.1 and cmaps[i] == "bluered":
            cbar = fig.colorbar(scatter, ticks=[0,1], shrink=0.8)
            cbar.ax.set_yticklabels(['No', 'Yes'])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        equal_axes(ax)
        plt.title(titles[i])


    plt.show()


def plot_field_frames(verts, fields, cmap="coolwarm", title=""):
    N = len(fields)
    fig = plt.figure(figsize=(N*7,6), dpi=300)

    zmin = np.min(np.concatenate([v[:,2] for v in verts]))
    zmax = np.max(np.concatenate([v[:,2] for v in verts]))

    for i in range(N):
        xyz = verts[i//2 + 1 + 1 *(i == N-1)]
        ax = fig.add_subplot(1,N,i+1, projection='3d')
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=fields[i], cmap=cmap)
        equal_axes(ax)
        if i < N-1:
            ax.set_zlim3d([zmin, zmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.title(f"t = {i+1}")

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

class Gif:
    """A list of images that can be turned into a gif   

    Methods:
    - add_frame()
    - add_frame_from_file(filename)
    - add_frame_from_figure(figure)
    - clear()
    - export(filename, fps)

    """
    def __init__(self):
        self.frames = []

    def add_frame(self):
        """Append a frame from the current figure"""
        filename = "temp_gif_frame_image.png"
        plt.savefig(filename, bbox_inches='tight')
        image = imageio.imread(filename)
        os.remove(filename)
        self.frames.append(image)
    
    def add_frame_from_file(self, filename):
        """Append a frame from an image file"""
        image = imageio.imread(filename)
        self.frames.append(image)

    def add_frame_from_figure(self, figure):
        """Append a frame from a given figure/figure number"""
        plt.figure(figure)
        self.add_frame()
    
    def clear(self):
        """Clear all frames within a gif"""
        self.frames.clear()
    
    def export(self, filename, fps=10):
        """Export a gif to a file (at a specified fps)"""
        duration = len(self.frames) / fps
        imageio.mimsave(filename, self.frames, duration=duration)