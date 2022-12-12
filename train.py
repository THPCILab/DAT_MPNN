'''
Author: Zhengyang Duan, Ziyang Zheng
Date: 2022-03-18 20:36:45
LastEditors: Ziyang Zheng
LastEditTime: 2022-12-09 16:47:55
Description: Training pnn.
'''
import time
import os
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import model
import numpy as np
from function import write_txt, norm_inputs
import function
import net


class Training():
    def __init__(self, config):
        self.conf = config
        self._H = config.H
        self._W = config.W

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
            if self.conf.train_method != 'dat':
                pnn_net = model.Pure_MZI_Net(self.conf)
            else:
                HW_size = int(np.sqrt(self.conf.waveguide_dims))
                for i in range(1, self.conf.iter_num + 1):
                    if self.conf.cn_model == 'ResNet':
                        setattr(self, f'cn{i}', net.ComplexResNet([HW_size, HW_size], self.conf.kernel_size, False, self.conf.CB_layers, self.conf.FM_num))
                    elif self.conf.cn_model == 'UNet':
                        setattr(self, f'cn{i}', net.ComplexUNet([HW_size, HW_size], self.conf.kernel_size, False, self.conf.CB_layers, self.conf.FM_num))
                    else:
                        raise ValueError("The compensation model is not supported.")
                pnn_net = model.Pure_MZI_Net_With_FSP_Compensation(self.conf, [getattr(self, f'cn{i}') for i in range(1, self.conf.iter_num + 1)])
        else:
            raise ValueError("This model is not defined.")

        pnn_net = pnn_net.cuda()
        return pnn_net


    def net_training_process(self):
        """
        Training with numerical model
        """
        ### define parameters saving path and log path ###
        pth_path = os.path.join(self.conf.save_path, 'train.pth')
        log_path = os.path.join(self.conf.save_path, 'train.log')
        
        ### prepare dataset ###
        train_loader, test_loader = self.prepare_data()
  
        ### define model ###
        pnn_net = self.prepare_model()

        if self.conf.load_existing_model:
            best_path = os.path.join(self.conf.save_path, 'train_best.pth')
            if os.path.exists(best_path):
                pnn_net.load_state_dict(torch.load(best_path), strict=True)
                write_txt(log_path, "Successfully loading existing model......")
            else:
                raise ValueError("Not supported already.")
            '''
            TODO: Implement training after loading existing model. Also modify config.py to support.
            '''
        else:
            write_txt(log_path, "Training new model from scratch......")

        ### define loss & optimizer
        criterion_pnn = nn.CrossEntropyLoss()
        params_pnn = [ p for n, p in pnn_net.named_parameters() if "RM_block" in n ]
        optimizer_pnn = optim.Adam(params_pnn, lr=self.conf.init_lr)  
        sched_pnn = optim.lr_scheduler.StepLR(optimizer_pnn, step_size=10, gamma=0.5)

        if self.conf.train_method == 'dat':
            criterion_cn  = nn.MSELoss()
            params_cn = [ p for n, p in pnn_net.named_parameters() if "cn" in n ]
            optimizer_cn  = optim.Adam(params_cn, lr=self.conf.init_lr_cn)
            sched_cn = optim.lr_scheduler.StepLR(optimizer_cn, step_size=10, gamma=0.5)
        
        write_txt(log_path, '--------------------------\nTraining with SEPN\n--------------------------')
        start_time = time.time()

        best_test_result = 0.

        for epoch in range(self.conf.epoch):  # loop over the dataset multiple times

            current_lr_pnn= optimizer_pnn.param_groups[0]['lr']
            if self.conf.train_method == 'dat':
                current_lr_cn = optimizer_cn.param_groups[0]['lr']

            write_txt ( log_path, 
                        f'\nCurrent training epoch: {epoch + 1}, '+
                        f'learning rate for pnn: {current_lr_pnn}, ' + 
                        (f'learning rate for SEPN: {current_lr_cn}' if self.conf.train_method == 'dat' else ''))
            
            running_loss_pnn = []
            running_loss_cn  = []

            cn_weight = 0. if epoch < 20 else 1. 
            # cn_weight =  1. 

            running_acc_sim  = []
            running_acc_phy = []

            crop_range = int(np.sqrt(self.conf.waveguide_dims)) // 2
            base_index = self.conf.ori_size // 2

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0], data[1]
                inputs = inputs.squeeze()
                labels = torch.tensor(np.eye(10)[labels]).cuda()
                inputs = np.fft.fftshift(np.fft.fft2(inputs.numpy()), axes=(1, 2))
                inputs = inputs[:, base_index-crop_range: base_index+crop_range, 
                                base_index-crop_range: base_index+crop_range]
                inputs = norm_inputs(inputs.reshape((-1, self.conf.waveguide_dims))).astype(np.complex64)

                optimizer_pnn.zero_grad()
                loss_pnn = torch.zeros(1)
                loss_cn = torch.zeros(1)
                
                if self.conf.train_method == 'dat':
                    optimizer_cn.zero_grad()
                    outputs_sim = pnn_net(inputs)
                    outputs_phy = pnn_net.phy_forward(inputs)
                    in_outs_phys_ = pnn_net.in_outs_phy
                    
                    if self.conf.meas_IS_separable:
                        for num in range(self.conf.iter_num):
                            optimizer_cn.zero_grad()
                            outp_unit, outp_unit_phy = pnn_net.forward_for_training_cn(in_outs_phys_[num], num)
                            loss_cn_unit = criterion_cn(outp_unit, outp_unit_phy)
                            loss_cn_unit.backward()
                            optimizer_cn.step()
                        if self.conf.meas_IS_unitary:
                            outputs_sim = pnn_net(inputs)
                            outputs_phy = pnn_net.phy_forward(inputs)

                    if self.conf.meas_IS_unitary:
                        loss_cn = criterion_cn(outputs_sim, outputs_phy) + \
                                  criterion_cn(pnn_net.at_mask_intensity1, pnn_net.at_mask_intensity_phy1) + \
                                  criterion_cn(pnn_net.at_mask_intensity2, pnn_net.at_mask_intensity_phy2)
                    else:
                        loss_cn = criterion_cn(outputs_sim, outputs_phy)
                    loss_cn.backward()
                    optimizer_cn.step()
                
                if self.conf.train_method == 'dat':
                    outputs_sim = pnn_net(inputs, cn_weight) 
                else:
                    outputs_sim = pnn_net(inputs) 
                correct_sim = function.correct(outputs_sim, labels)
                with torch.no_grad():
                    outputs_phy = pnn_net.phy_forward(inputs)
                    correct_phy = function.correct(outputs_phy, labels)
                    if not self.conf.train_method == 'ideal':
                        pnn_net.phy_replace_sim()
                        outputs_sim.data.copy_(outputs_phy.data)
                loss_pnn = criterion_pnn(outputs_sim, labels)
                loss_pnn.backward()
                optimizer_pnn.step()

                # record information
                running_loss_pnn.append(loss_pnn.item())
                running_loss_cn.append(loss_cn.item())
                running_acc_sim.append(correct_sim)
                running_acc_phy.append(correct_phy)
            
                # print statistics
                if (i+1) % self.conf.log_batch_num == 0:
                    content = ( f'| epoch = {epoch + 1} ' + 
                            f'| step = {i + 1:5d} ' + 
                            f'| loss_pnn = {np.mean(running_loss_pnn):.3f} ' +
                            f'| loss_cn = {np.mean(running_loss_cn):.8f} ' +
                            f'| acc_sim = {np.mean(running_acc_sim):.3f} ' +
                            f'| acc_phy = {np.mean(running_acc_phy):.3f} ')
                    write_txt(log_path, content)
            
            sched_pnn.step()
            if self.conf.train_method == 'dat':
                sched_cn.step()

            if (epoch + 1) % self.conf.save_epoch == 0 and epoch + 1 < self.conf.epoch:
                mid_pth_path = os.path.join(self.conf.save_path, f'train_ep{epoch + 1}.pth')
                torch.save(pnn_net.state_dict(), mid_pth_path)

            testing_acc_phy = []
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data[0], data[1]
                inputs = inputs.squeeze()
                labels = torch.tensor(np.eye(10)[labels]).cuda()
                inputs = np.fft.fftshift(np.fft.fft2(inputs.numpy()), axes=(1, 2))
                inputs = inputs[:, base_index-crop_range:base_index+crop_range, base_index-crop_range:base_index+crop_range]
                inputs = norm_inputs(inputs.reshape((-1, self.conf.waveguide_dims))).astype(np.complex64)

                with torch.no_grad():
                    outputs_phy = pnn_net.phy_forward(inputs)
                    correct_phy = function.correct(outputs_phy, labels)

                # record information
                testing_acc_phy.append(correct_phy)

            current_test_result = np.mean(testing_acc_phy)
            content = ( f'Phy accuracy of the network on 10000 test images: {current_test_result:.3f}\n')
            write_txt(log_path, content)

            if current_test_result > best_test_result:
                best_test_result = current_test_result
                best_pth_path = os.path.join(self.conf.save_path, 'train_best.pth')
                torch.save(pnn_net.state_dict(), best_pth_path)
            
        torch.save(pnn_net.state_dict(), pth_path)
            
        end_time = time.time()
        elapsed = end_time - start_time
        content = (f"Finished training. Elapsed time of training = {str(timedelta(seconds=elapsed))}.\n" +\
                   f"Best accuracy on test images: {best_test_result}.")
        write_txt(log_path, content)
