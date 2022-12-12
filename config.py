'''
Author: Ziyang Zheng
Date: 2022-03-18 21:17:09
LastEditors: Ziyang Zheng
LastEditTime: 2022-12-09 17:17:19
Description: Configuration.
'''

import argparse
import shutil
import os
from function import write_txt

parser = argparse.ArgumentParser()

"""
MZI model.
"""
pnn_model_arg = parser.add_argument_group('pnn_model')
pnn_model_arg.add_argument(
    '-pnn', '--pnn_name', type=str, default='PureMZINet',
    help="The name of models.")
pnn_model_arg.add_argument(
    '-iterN', '--iter_num', type=int, default=3,
    help="How many photonic meshes are used to construct the model.")


"""
Data pre-processing.
"""
data_arg = parser.add_argument_group('data')
data_arg.add_argument(
    '-dataN', '--data_name', type=str, default="MNIST",
    help="Name of training data.")
data_arg.add_argument(
    '-ors', '--ori_size', type=int, default=28,
    help="Size of original input. Default setting corresponds to MNIST dataset.")

"""
Complex-valued systematic error prediction networks.
"""
pnn_cn_arg = parser.add_argument_group('pnn_cn')
pnn_cn_arg.add_argument(
    '-cn', '--cn_model', type=str, choices=['UNet', 'ResNet'], default='Unet',
    help="The name of SEPN.")
pnn_cn_arg.add_argument(
    '-ks', '--kernel_size', type=int, default=3,
    help="Kernel size of convolution layers.")
pnn_cn_arg.add_argument(
    '-bnflag', '--BN_flag', type=bool, default=False,
    help="Whether adopt Batch Normalization or not.")
pnn_cn_arg.add_argument(
    '-CB_layers', '--CB_layers', nargs=3, type=int, default=[3, 3, 3],
    help="Please refer to 'ComplexNet' in 'net.py' for more details.")
pnn_cn_arg.add_argument(
    '-FM_num', '--FM_num', nargs=3, type=int, default=[4, 8, 16],
    help="Feature numbers. Please refer to 'ComplexNet' in 'net.py' for more details.")
pnn_cn_arg.add_argument(
    '-misu', '--meas_IS_unitary', action='store_true', default=False,
    help="Optimize SEPN in unitary mode.")
pnn_cn_arg.add_argument(
    '-miss', '--meas_IS_separable', action='store_true', default=False,
    help="Optimize SEPN in separable mode.")


"""
Experimental Setups.
"""
setup_arg = parser.add_argument_group('setup')
setup_arg.add_argument(
    '-train', '--train', action='store_true',
    help="Train model if true.")
setup_arg.add_argument(
    '-test', '--test', action='store_true',
    help="Test model if true.")
setup_arg.add_argument(
    '-method', '--train_method', type=str, choices=['ideal', 'pat', 'dat'], default='ideal',
    help="The method to train the model. Choices include [ideal, pat, dat]\n" + 
            "ideal: in silico training in an error-free system\n" + 
            "pat: physics-aware training\n" + 
            "dat: dual adaptive training.")
setup_arg.add_argument(
    '-id', '--exp_id', type=int, default=0,
    help="ID of the experiment/model.")
setup_arg.add_argument(
    '-g', '--gpu', type=str, default='1',
    help="ID's of allocated GPUs.")
setup_arg.add_argument(
    '-load_pt', '--load_path', type=str, default="./logs",
    help="Where to load models.")
setup_arg.add_argument(
    '-save_pt', '--save_path', type=str, default="./logs",
    help="Where to save models.")
setup_arg.add_argument(
    '-p', '--plot', action='store_true',
    help='Save phase, confusion matrix and energe distribution during testing phase.'
)

"""
Training arguments.
"""
train_arg = parser.add_argument_group('train')
train_arg.add_argument(
    '-train_load', '--load_existing_model', action='store_true',
    help="Whether we load existing models for training or not.")
train_arg.add_argument(
    '-bs', '--train_batch_size', type=int, default=500,
    help="Training batch size.")
train_arg.add_argument(
    '-di', '--log_batch_num', type=int, default=10,
    help="The iteration number for displaying training information.")
train_arg.add_argument(
    '-epoch', '--epoch', type=int, default=50,
    help="Training epoches.")
train_arg.add_argument(
    '-save_ep', '--save_epoch', type=int, default=10,
    help="When to save models.")
train_arg.add_argument(
    '-lr', '--init_lr', type=float, default=1e-3,
    help="Initial learning rate.")
train_arg.add_argument(
    '-lrcn', '--init_lr_cn', type=float, default=1e-3,
    help="Initial learning rate for SEPNs.")


"""
Arguments for MZI-based pnns.
"""
mzi_arg = parser.add_argument_group('mzi')
mzi_arg.add_argument(
    '-wave_dims', '--waveguide_dims', type=int, default=64,
    help="The input dimensions.")
mzi_arg.add_argument(
    '-bsE', '--bs_error', type=float, default=0.1,
    help="Std of beamspliter errors.")
mzi_arg.add_argument(
    '-phaseE', '--phase_error', type=float, default=0.1,
    help="Std of phase shifter errors.")
mzi_arg.add_argument(
    '-Etype', '--error_type', type=str, 
    choices=['noE', 'phaseE', 'bsE', 'bothE'], default="noE",
    help="The type of errors.")


def get_config():
    config, unparsed = parser.parse_known_args()

    # check the settings
    # assert config.lr_decay[1] < config.lr_decay[0] < 1.0
    assert config.train ^ config.test

    setattr(config, 'H', config.ori_size)
    setattr(config, 'W', config.ori_size)
    
    # record model information
    kernel_size_info = str(config.CB_layers[0]) + str(config.CB_layers[1]) + str(config.CB_layers[2])
    feature_map_info = str(config.FM_num[0]) + str(config.FM_num[1]) + str(config.FM_num[2])
    model_data_info  = (f'{config.pnn_name}' + 
                        f'_N{config.iter_num}' + 
                        f'_{config.data_name}' +
                        f'_L{config.waveguide_dims}')
    
    if config.train_method == 'dat':
        model_data_info +=  (f'_CnK{kernel_size_info}' + f'_F{feature_map_info}')
    if config.meas_IS_unitary and not config.meas_IS_separable:  
        model_data_info +=  (f'_RecordInterState_U')
    elif config.meas_IS_separable and not config.meas_IS_unitary:
        model_data_info +=  (f'_RecordInterState_S')
    elif config.meas_IS_separable and config.meas_IS_unitary:
        model_data_info +=  (f'_RecordInterState_US')
        
    exp_error_info   =  f'ExpID_{config.exp_id}'
    if config.error_type == 'noE':
        assert config.train_method == 'ideal'
        exp_error_info  +=  f'_noError' 
        config.bs_error = 0.0
        config.phase_error = 0.0
    if config.error_type == 'bsE' or config.error_type == 'bothE':
        exp_error_info  +=  f'_BS{config.bs_error}' 
    if config.error_type == 'phaseE' or config.error_type == 'bothE':
        exp_error_info  +=  f'_PS{config.phase_error}'
        
    config.load_path = os.path.join(config.load_path, model_data_info, 'Base') if (config.train_method == 'ideal' and config.test) else \
                       os.path.join(config.load_path, model_data_info, exp_error_info, config.train_method)
    config.save_path = os.path.join(config.save_path, model_data_info, 'Base') if (config.train_method == 'ideal' and config.train) else \
                       os.path.join(config.save_path, model_data_info, exp_error_info, config.train_method)

    if config.test:
        if not os.path.exists(config.load_path):
            print(config.load_path)
            raise ValueError('No model state can be loaded.')
        if os.path.exists(config.save_path):
            if os.path.exists(os.path.join(config.save_path, 'test.log')):
                os.remove(os.path.join(config.save_path, 'test.log'))
        else:
            os.makedirs(config.save_path)

    if config.train and not config.load_existing_model:
        if os.path.exists(config.save_path):
            if input("Path already existed. Input 'Y' to delete and re-train, others to exit.\n") == 'Y':
                shutil.rmtree(config.save_path)
            else:
                exit()
        os.makedirs(config.save_path)

    if config.train and not config.load_existing_model:
        ArgInfo_path = os.path.join(config.save_path, 'ArgInfo.txt')
        # Arguments from command line and default values
        write_txt(ArgInfo_path, '---Argument Information---')
        for name, value in vars(parser.parse_args()).items():
            write_txt(ArgInfo_path, f'{name}: {value}')
        write_txt(ArgInfo_path, '--------------------------')

    return config, unparsed