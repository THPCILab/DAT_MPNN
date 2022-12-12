import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import normalize
from function import cnormsq, rc_mul, cc_mul, phasor
#################################################
from neurophoxTorch.torch import RMTorch
# The function 'RMTorch' is modified to add 
# phase shift errors based on the origin version.
#################################################

class ElectroopticNonlinearity(nn.Module):
    def __init__(self, alpha: float=0.1, g: float=0.05 * np.pi, phi_b: float=np.pi):
        super(ElectroopticNonlinearity, self).__init__()
        self.alpha = alpha
        self.g = g
        self.phi_b = phi_b

    def forward(self, inputs):
        phase = 0.5 * self.g * cnormsq(inputs) + 0.5 * self.phi_b
        return np.sqrt(1 - self.alpha) * cc_mul(rc_mul(phase.cos(), phasor(-phase)), inputs)


class CNormSq(nn.Module):
    def __init__(self, normed=True):
        super(CNormSq, self).__init__()
        self.normed = normed

    def forward(self, inputs):
        return normalize(cnormsq(inputs), dim=1) if self.normed else cnormsq(inputs)


################################################################
############### For MNIST/FMNIST CLASSIFICATION ################
################################################################

class Pure_MZI_Net(nn.Module):
    """
    Input  -> | N*N matrix using MZIs -> Activation | -> N*N matrix -> Output
               -----  Repeat (IterN - 1) times -----
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.N      = config.waveguide_dims
        self.bsE    = config.bs_error
        self.phaseE = config.phase_error
        self.model  = 'PureMZINet'
        self.IterN  = config.iter_num
        self.error  = config.error_type

        if config.data_name == 'MNIST' or 'FMNIST': 
            self.index_end = 10
        else:
            raise ValueError('This dataset has not been prepared!')

        for i in range(self.IterN):
            bs_error_file = ("./MZI_Error/{}/N{}/bsE{}/Iter{}".format(self.model, self.N, self.bsE, i))
            phase_error_file = ("./MZI_Error/{}/N{}/phaseE{}/Iter{}".format(self.model, self.N, self.phaseE, i))
            setattr(self, f'RM_block{i}', RMTorch(units=self.N, bs_error=self.bsE, bs_error_files=bs_error_file,
                                                    phase_error=self.phaseE, phase_error_files=phase_error_file))
            if i != self.IterN - 1: 
                setattr(self, f'Act{i}', ElectroopticNonlinearity())
            else:
                setattr(self, f'OutUnit', CNormSq())

    def forward(self, x):
        # forward propagation in an error-free system
        for i in range(self.IterN):
            x, _, _, _ = getattr(self, f'RM_block{i}')(x)
            if i != self.IterN - 1:
                x = getattr(self, f'Act{i}')(x)
            else:
                self.at_sensor = x
                x = getattr(self, f'OutUnit')(x)
        self.at_sensor_intensity = x
        return self.at_sensor_intensity[:, :self.index_end]

    def phy_forward(self, x):
        # forward propagation in the physical system with systematic errors
        self.in_outs_phy = []
        with torch.no_grad():
            for i in range(self.IterN):
                self.in_outs_phy.append(x)
                if self.error == 'bsE':
                    _, x, _, _ = getattr(self, f'RM_block{i}')(x)
                elif self.error == 'phaseE':
                    _, _, x, _ = getattr(self, f'RM_block{i}')(x)
                elif self.error == 'bothE':
                    _, _, _, x = getattr(self, f'RM_block{i}')(x)
                elif self.error == 'noE':
                    x, _, _, _ = getattr(self, f'RM_block{i}')(x)
                else:
                    raise ValueError('This error has not been defined.')
                setattr(self, f'at_mask_phy{i+1}', x)
                setattr(self, f'at_mask_intensity_phy{i+1}', CNormSq()(x))
                if i != self.IterN - 1:
                    x = getattr(self, f'Act{i}')(x)
                else:
                    self.at_sensor_phy = x
                    x = getattr(self, f'OutUnit')(x)
            self.at_sensor_intensity_phy = x
            return self.at_sensor_intensity_phy[:, :self.index_end]

    def phy_replace_sim(self):
        # PAT: replace the output
        with torch.no_grad():
            self.at_sensor_intensity.data.copy_(self.at_sensor_intensity_phy.data)


class Pure_MZI_Net_With_FSP_Compensation(Pure_MZI_Net):
    def __init__(self, config, cns):
        super().__init__(config)

        for i in range(1, config.iter_num + 1):
            setattr(self, f'cn{i}', cns[i - 1])
    
    def forward(self, x, cn_weight=1.):
        # forward propagation in the simulation model with complex-valued SEPNs
        for i in range(self.IterN):
            x, _, _, _ = getattr(self, f'RM_block{i}')(x)
            x    = getattr(self, f'cn{i+1}')(x) * cn_weight + x
            setattr(self, f'at_mask{i+1}', x)
            setattr(self, f'at_mask_intensity{i+1}', CNormSq()(x))
            if i != self.IterN - 1:
                x = getattr(self, f'Act{i}')(x)
            else:
                self.at_sensor = x
                x = getattr(self, 'OutUnit')(x)
        self.at_sensor_intensity = x
        return self.at_sensor_intensity[:, :self.index_end]
    
    def phy_forward(self, inp):
        return super().phy_forward(inp)

    def forward_for_training_cn(self, inp, num):
        # training in the separable mode when measuring internal states
        x_sim, _, _, _ = getattr(self, f'RM_block{num}')(inp)
        x_sim    = getattr(self, f'cn{num+1}')(x_sim) + x_sim
        x_sim = getattr(self, 'OutUnit')(x_sim)

        if self.error == 'bsE':
            _, x_phy, _, _ = getattr(self, f'RM_block{num}')(inp)
        elif self.error == 'phaseE':
            _, _, x_phy, _ = getattr(self, f'RM_block{num}')(inp)
        elif self.error == 'bothE':
            _, _, _, x_phy = getattr(self, f'RM_block{num}')(inp)
        x_phy = getattr(self, 'OutUnit')(x_phy)

        return x_sim, x_phy
    
    def phy_replace_sim(self):
        # DAT: state fusion
        with torch.no_grad():
            angle = torch.angle(torch.complex(self.at_sensor[0, :, :], self.at_sensor[1, :, :]))
            amp = torch.abs(torch.complex(self.at_sensor_phy[0, :, :], self.at_sensor_phy[1, :, :]))
            new_data = amp * torch.exp(1j * angle)
            data_transfer = torch.stack([torch.real(new_data), torch.imag(new_data)], dim=0)
            self.at_sensor.data.copy_(data_transfer.data)
            self.at_sensor_intensity.data.copy_(self.at_sensor_intensity_phy.data)

            if self.config.meas_IS_unitary or self.config.meas_IS_separable:
                for i in range(1, self.IterN):
                    angle = torch.angle(torch.complex(getattr(self, f'at_mask{i}')[0, :, :], 
                                                        getattr(self, f'at_mask{i}')[1, :, :]))
                    amp = torch.abs(torch.complex(getattr(self, f'at_mask_phy{i}')[0, :, :], 
                                                        getattr(self, f'at_mask_phy{i}')[1, :, :]))
                    new_data = amp * torch.exp(1j * angle)
                    data_transfer = torch.stack([torch.real(new_data), torch.imag(new_data)], dim=0)
                    getattr(self, f'at_mask{i}').data.copy_(data_transfer.data)
                    getattr(self, f'at_mask_intensity{i}').data.copy_(getattr(self, f'at_mask_intensity_phy{i}').data)
