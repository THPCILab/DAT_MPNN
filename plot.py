import scipy.io as sio
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as color
import copy
import os


def plot_cm(confusion, dir):
    blue = np.array([114, 191, 68]) / 255
    blue_list = np.linspace([1., 1., 1], blue, 8)
    blues = color.ListedColormap(blue_list, 'blues')

    confusion_plot = copy.deepcopy(confusion)
    confusion_plot[confusion >= 350] = 350
    plt.figure()
    plt.imshow(confusion_plot, cmap=blues)
    plt.xticks(np.arange(10), str(np.arange(10)).replace('[', '').replace(']', '').split(' '))
    plt.yticks(np.arange(10), str(np.arange(10)).replace('[', '').replace(']', '').split(' '))
    for i, j in itertools.product(range(10), range(10)):
        plt.text(j, i, int(confusion[i, j]), horizontalalignment='center')
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title('Confusion Matrix')
    save_dir = os.path.join(dir, 'confusion.pdf')
    plt.savefig(save_dir)


def plot_ed(energe_distribution, dir):
    yellow = np.array([248, 153, 28]) / 255
    yellow_list = np.linspace([1., 1., 1], yellow, 8)
    yellows = color.ListedColormap(yellow_list, 'yellow')
    for i in range(10):
        energe_distribution[:, i] = energe_distribution[:, i] / np.sum(energe_distribution[:, i]) * 100

    plt.figure()
    plt.imshow(energe_distribution, cmap=yellows)
    plt.xticks(np.arange(10), str(np.arange(10)).replace('[', '').replace(']', '').split(' '))
    plt.yticks(np.arange(10), str(np.arange(10)).replace('[', '').replace(']', '').split(' '))
    for i, j in itertools.product(range(10), range(10)):
        plt.text(j, i, int(energe_distribution[i, j]), horizontalalignment='center')
    plt.xlabel('Input Digits')
    plt.ylabel('Detector Regions')
    plt.title('Energe Distribution')
    save_dir = os.path.join(dir, 'energe_distribution.pdf')
    plt.savefig(save_dir)


def plot_phase(phase_shifter, name, dir):
    parula_list = sio.loadmat('parula.mat')['parula']
    parula = color.ListedColormap(parula_list, 'parula')
    
    plt.imshow(phase_shifter, cmap=parula)
    plt.axis('off')
    plt.colorbar()
    save_dir = os.path.join(dir, name + '.pdf')
    plt.savefig(save_dir)


def plot_intensity(intensity, loca_num, dir):
    parula_list = sio.loadmat('parula.mat')['parula']
    parula = color.ListedColormap(parula_list, 'parula')
    
    plt.figure()
    plt.imshow(intensity / intensity.max(), cmap=parula)
    plt.colorbar()
    plt.axis('off')
    save_dir = os.path.join(dir, 'Inten_atM'+str(loca_num)+'.pdf')
    plt.savefig(save_dir)
