import torch
import numpy as np


def cnormsq(comp: torch.Tensor):
    return comp[0] ** 2 + comp[1] ** 2


def rc_mul(real: torch.Tensor, comp: torch.Tensor):
    return real.unsqueeze(dim=0) * comp


def cc_mul(comp1: torch.Tensor, comp2: torch.Tensor) -> torch.Tensor:
    real = comp1[0] * comp2[0] - comp1[1] * comp2[1]
    comp = comp1[0] * comp2[1] + comp1[1] * comp2[0]
    return torch.stack((real, comp), dim=0)


def phasor(real: torch.Tensor):
    return torch.stack((real.cos(), real.sin()), dim=0)


def norm_inputs(inputs, feature_axis=1):
    if feature_axis == 1:
        n_features, n_examples = inputs.shape
    elif feature_axis == 0:
        n_examples, n_features = inputs.shape
    for i in range(n_features):
        l1_norm = np.mean(np.abs(inputs[i, :]))
        inputs[i, :] /= l1_norm
    return inputs


def write_txt (dir, data_save):
    if isinstance(data_save, str):
        print(data_save)
    with open(dir, 'a') as data:
        data.write(str(data_save) + '\n')


def correct(energe, label):
    corr = (energe.argmax(dim=-1) == label.argmax(dim=-1)).sum().item()
    corr /= energe.size(0)
    return corr