'''
Author: Ziyang Zheng
Date: 2022-03-18 20:36:45
LastEditors: Please set LastEditors
LastEditTime: 2022-04-13 17:35:20
Description: Testing pnn.
'''

import os
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import model
import scipy.io as sio
import numpy as np
import function
from function import write_txt, norm_inputs
import collections
import plot

class Testing():
    """
    Testing models.
    """
    def __init__(self, config):
        self.conf = config

    def prepare_data(self):
        if self.conf.data_name == 'MNIST':
            Mnist_train = torchvision.datasets.MNIST(root='./data',
                                                    train=True, download = True,
                                                    transform = transforms.ToTensor())
            Mnist_test = torchvision.datasets.MNIST(root='./data',
                                                    train=False, download = True,
                                                    transform = transforms.ToTensor())
            train_loader = Data.DataLoader(Mnist_train, batch_size=self.conf.train_batch_size, shuffle=True)
            test_loader = Data.DataLoader(Mnist_test, batch_size=self.conf.train_batch_size) 
            return train_loader, test_loader

        elif self.conf.data_name == 'FMNIST':
            Fmnist_train = torchvision.datasets.FashionMNIST(root='./data',
                                                            train=True, download = True,
                                                            transform = transforms.ToTensor())
            Fmnist_test = torchvision.datasets.FashionMNIST(root='./data',
                                                            train=False, download = True,
                                                            transform = transforms.ToTensor())
            train_loader = Data.DataLoader(Fmnist_train, batch_size=self.conf.train_batch_size, shuffle=True)
            test_loader = Data.DataLoader(Fmnist_test, batch_size=self.conf.train_batch_size)         
            return train_loader, test_loader                               

        else:
            raise ValueError("This data is not prepared.")
         
         
    def prepare_model(self):
        if self.conf.pnn_name == 'PureMZINet':
            pnn_net = model.Pure_MZI_Net(self.conf)
        else:
            raise ValueError("This model is not defined.")

        return pnn_net.cuda()
    
    
    def do_testing(self):
        """
        Testing with physical system
        """
        ### define parameters saving path and log path ###
        if self.conf.train_method == 'ideal':
            pth_path = os.path.join(self.conf.load_path, 'train.pth')
        else:
            pth_path = os.path.join(self.conf.load_path, 'train_best.pth')
        log_path = os.path.join(self.conf.save_path, 'test.log')

        ### prepare dataset ###
        train_loader, test_loader = self.prepare_data()
  
        ### define model ###
        pnn_net = self.prepare_model()

        if os.path.exists(pth_path):
            state_dict = torch.load(pth_path)
            # state_dict = torch.load(pth_path, map_location='cuda:0')
            pnn_state_dict = collections.OrderedDict((key, value) for key, value in state_dict.items() if 'RM' in key)
            pnn_net.load_state_dict(pnn_state_dict, strict=True)
            if self.conf.plot:
                for plot_num in range(self.conf.iter_num):
                    theta_name = 'RM_block{}.theta'.format(plot_num)
                    phi_name   = 'RM_block{}.phi'.format(plot_num)
                    plot.plot_phase(pnn_state_dict[theta_name].cpu().numpy(), theta_name, self.conf.save_path)
                    plot.plot_phase(pnn_state_dict[phi_name].cpu().numpy(), phi_name, self.conf.save_path)
            write_txt(log_path, "Successfully loading existing model......")
        else:
            raise ValueError("No existing model can be loaded.")
        
        running_acc_sim  = []
        running_acc_phy = []

        crop_range = int(np.sqrt(self.conf.waveguide_dims)) // 2
        base_index = self.conf.ori_size // 2

        confusion_matrix = torch.zeros(10, 10).cuda()
        energe_distribution = torch.zeros(10, 10).cuda()

        for i, data in enumerate(test_loader, 0):
            inputs, labels = data[0], data[1]
            inputs = inputs.squeeze()
            labels = torch.tensor(np.eye(10)[labels]).cuda()
            inputs_fre = np.fft.fftshift(np.fft.fft2(inputs.numpy()), axes=(1, 2))
            inputs_crop = inputs_fre[:, base_index - crop_range : base_index + crop_range, 
                                        base_index - crop_range : base_index + crop_range]
            inputs_norm = norm_inputs(inputs_crop.reshape((-1, self.conf.waveguide_dims))).astype(np.complex64)

            outputs_sim = pnn_net(inputs_norm)
            correct_sim = function.correct(outputs_sim, labels)
            with torch.no_grad():
                outputs_phy = pnn_net.phy_forward(inputs_norm)
                correct_phy = function.correct(outputs_phy, labels)

            # record information
            running_acc_sim.append(correct_sim)
            running_acc_phy.append(correct_phy)

            predict_phy = outputs_phy.argmax(dim=-1)
            ground_truth = labels.argmax(dim=-1)

            for n in range(self.conf.train_batch_size):
                pred = predict_phy[n]
                gt = ground_truth[n]
                confusion_matrix[pred, gt] += 1
                energe_distribution[:, gt] += outputs_phy[n, :]
        
            # print statistics
            if (i+1) % self.conf.log_batch_num == 0:
                content = ( f'| step = {i + 1:5d} ' + 
                            f'| acc_sim = {np.mean(running_acc_sim):.3f} ' +
                            f'| acc_phy = {np.mean(running_acc_phy):.3f} ')
                write_txt(log_path, content)

            if self.conf.plot:
                if i == 0:
                    plot.plot_intensity(inputs[0, :, :].cpu().numpy(), 'product', self.conf.save_path)
                    plot.plot_intensity(abs(inputs_fre[0, :, :]), 'fre', self.conf.save_path)
                    plot.plot_intensity(np.expand_dims(abs(inputs_norm[0, :]), -1), 'center', self.conf.save_path)
                    for iter in range(1, self.conf.iter_num):
                        plot.plot_intensity(np.expand_dims(getattr(pnn_net, f'at_mask_intensity_phy{iter}')[0, :].cpu().numpy(), -1), iter, self.conf.save_path)
                    plot.plot_intensity(np.expand_dims(pnn_net.at_sensor_intensity_phy[0, :10].cpu().numpy(), -1), self.conf.iter_num, self.conf.save_path)

        if self.conf.plot:
            plot.plot_cm(confusion_matrix.cpu().numpy(), self.conf.save_path)
            plot.plot_ed(energe_distribution.cpu().numpy(), self.conf.save_path)

        content = ( f'Accuracy in an error-free system on 10000 test images: {np.mean(running_acc_sim):.3f}\n' + 
                    f'Accuracy in the physical system with systematic errors on 10000 test images: {np.mean(running_acc_phy):.3f}\n')
        write_txt(log_path, content)