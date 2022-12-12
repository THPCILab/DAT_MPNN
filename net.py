import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, \
    ComplexLinear, ComplexConvTranspose2d, ComplexReLU


class ComplexConvLayer(nn.Sequential):
    def __init__(self, complex_conv, in_channels, out_channels, kernel_size, stride, 
                 padding=None, dilation=1, complex_bn=None, complex_act=None, bias=True):
        super(ComplexConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        if complex_conv == 'conv':
            self.add_module('ComplexConv2d', ComplexConv2d(in_channels, out_channels, kernel_size, 
                                                            stride, padding, dilation=dilation, bias=bias))
        elif complex_conv == 'deconv':
            self.add_module('ComplexConvT2d', \
                ComplexConvTranspose2d(in_channels, out_channels, kernel_size, 
                                        stride, padding, output_padding=1, dilation=dilation, bias=bias))
        else:
            raise ValueError('No such conv name.')
            
        if complex_bn is not None:
            self.add_module('ComplexBN', complex_bn(out_channels))
        if complex_act is not None:
            self.add_module('ComplexAct', complex_act)


class ComplexConvBlock(nn.Sequential):
    def __init__(self, in_channels, channels, complex_bn=ComplexBatchNorm2d, k=3, 
                 s=1, complex_act=ComplexReLU(), num_layer=4):
        super(ComplexConvBlock, self).__init__()
        self.add_module('ComplexConv-0', ComplexConvLayer(
                'conv', in_channels, channels, k, s, 
                padding=None, dilation=1, complex_bn=complex_bn, 
                complex_act=complex_act))

        for i in range(1, num_layer):
            self.add_module('ComplexConv-{}'.format(i), ComplexConvLayer(
                'conv', channels, channels, k, s, 
                padding=None, dilation=1, complex_bn=complex_bn, 
                complex_act=complex_act))


class ComplexUNet(nn.Module):
    """
    UNet architecture is adopted as follows.
    
    'Input'                                       'Output'
      | FFT                                          ^
      | Dot Multiply a Matrix                        | IFFT(x4 + x0)
      v                                              | -conv-> 'x4'
    'x0' ---------Conv Block_1 --------> 'x1' -> '{x1, Up[x2, Up(x3)]}'  
      |DownSampling_1                                ^ Concat two branches
      |conv-(BN)-ReLU                                |UpSampling_2
      v                                              |conv-(BN)-ReLU
    'Scaled 0.5x' ----Conv Block_2 ----> 'x2' -> '[x2, Up(x3)]'
      |DownSampling_2                                ^ Concat two branches
      |conv-(BN)-ReLU                                |UpSampling_1
      v                                              |conv-(BN)-ReLU
    'Scaled 0.25x' ---------Conv Block_3 ---------> 'x3'  
    
    where 'Conv Block_i' for 'i=1,2,3' consists of architecture as '...conv-(BN)-ReLU...'.
    
    All 'conv', 'BN' and 'ReLU' are set in complex mode.
    """
    
    def __init__(self, size, kernel_size=3, bn_flag=False,
                 CB_layers=[3, 3, 3], FM_num=[4, 8, 16]):
        """
        CB_layers: Number of conv layers in Conv Block. 
                   Should be a list of [Num_B1, Num_B2, Num_B2].
        FM_num   : Number of feature maps in Conv Block.
                   should be a list of [Num_B1, Num_B2, Num_B3].
        """
        super(ComplexUNet, self).__init__()
        self.H, self.W = size
        # self.DotMat = nn.Parameter(torch.randn(self.H, self.W) + 1j * torch.randn(self.H, self.W))
        
        # Define Conv Blocks
        In_channels = [ 1, FM_num[0], FM_num[1] ]
        bn = ComplexBatchNorm2d if bn_flag else None
        for i in range(3):
            self.add_module(f"ConvB{i+1}", \
                ComplexConvBlock( in_channels  = In_channels[i], 
                                  channels     = FM_num[i], 
                                  complex_bn   = bn,
                                  k            = kernel_size, 
                                  complex_act  = ComplexReLU(), 
                                  num_layer    = CB_layers[i] ) )
            
        # Define convs for DownSampling and UpSampling
        for i in range(2):
            self.add_module(f"DownConv{i+1}", \
                ComplexConvLayer( complex_conv = 'conv', 
                                  in_channels  = In_channels[i], 
                                  out_channels = FM_num[i], 
                                  kernel_size  = kernel_size, 
                                  stride       = 2, 
                                  complex_bn   = bn, 
                                  complex_act  = ComplexReLU() ) )
            
        for i in range(2):
            in_channel_Up  = FM_num[-1] if i == 0 else FM_num[-2] * 2
            out_channel_Up = FM_num[-2] if i == 0 else FM_num[-3] 
            self.add_module(f"UpConv{i+1}", \
                ComplexConvLayer( complex_conv = 'deconv', 
                                  in_channels  = in_channel_Up, 
                                  out_channels = out_channel_Up, 
                                  kernel_size  = kernel_size, 
                                  stride       = 2, 
                                  complex_bn   = bn, 
                                  complex_act  = ComplexReLU() ) )
            
        # Define final conv
        self.OutConv = \
                ComplexConvLayer( complex_conv = 'conv', 
                                  in_channels  = FM_num[-3] * 2, 
                                  out_channels = 1, 
                                  kernel_size  = kernel_size, 
                                  stride       = 1 )
        
    def forward(self, x):
        x         = torch.complex(x[0, :, :].float(), x[1, :, :].float())
        x0        = x.view( -1, 1, self.H, self.W )
        x1        = self.ConvB1   ( x0 )
        x0_down2x = self.DownConv1( x0 )
        x2        = self.ConvB2   ( x0_down2x )
        x0_down4x = self.DownConv2( x0_down2x )
        x3        = self.ConvB3   ( x0_down4x )
        x_up1     = self.UpConv1  ( x3 )
        x_up2     = self.UpConv2  ( torch.cat( (x2, x_up1), 1 ) )
        x4        = self.OutConv  ( torch.cat( (x1, x_up2), 1 ) )
        x_out = x4.view(-1, self.H * self.W)
        x_out = torch.stack([torch.real(x_out).double(), torch.imag(x_out).double()], dim=0)
        return x_out



class ComplexResNet(nn.Module):
    def __init__(self, size, kernel_size=3, bn_flag=False,
                 CB_layers=[4, 4, 4], FM_num=[4, 4, 4]):
        super(ComplexResNet, self).__init__()
        self.H, self.W = size
        
        # Define Conv Blocks
        bn = ComplexBatchNorm2d if bn_flag else None
        
        self.add_module("ConvB", \
            ComplexConvBlock( in_channels  = 1, 
                              channels     = FM_num[0], 
                              complex_bn   = bn,
                              k            = kernel_size, 
                              complex_act  = ComplexReLU(), 
                              num_layer    = CB_layers[0] ) )    
            
        # Define final conv
        self.OutConv = \
                ComplexConvLayer( complex_conv = 'conv', 
                                  in_channels  = FM_num[0], 
                                  out_channels = 1, 
                                  kernel_size  = kernel_size, 
                                  stride       = 1 )
            
    def forward(self, x):
        x     = torch.complex(x[0, :, :].float(), x[1, :, :].float())
        x0    = x.view(-1, 1, self.H, self.W)
        x1    = self.ConvB  ( x0 ) + x0
        x2    = self.OutConv ( x1 ) 
        x_out = x2.view(-1, self.H * self.W)
        x_out = torch.stack([torch.real(x_out).double(), torch.imag(x_out).double()], dim=0)
        return x_out
