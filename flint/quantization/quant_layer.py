import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from flint.quantization.quant_func_build_helper import Quantizer
from flint.quantization.quant_func.quant_func import IntQuantizer, FpQuantizer
import re
import os

##########
##  Layer
##########
class QuantizedConv2d(nn.Module):
    """ 
    A convolutional layer with its weight tensor and input tensor quantized. 
    """
    def __init__(self, wbits=0, wfmt='int', wnorm=2, abits=0, afmt='int',anorm=2, \
        cali_num=200,w_scale_method='max',a_scale_method='max',quant_pos='out', 
        enable_int=True,path="", mo4w=False, limited=False, fast=False, cali_mode='global'):
        super(QuantizedConv2d, self).__init__()
        self.wbits = wbits
        self.abits = abits
        self.use_weight_quant = True
        self.use_act_quant = True
        self.path = path
        self.mo4w = mo4w
        self.enable_int=enable_int
        self.fast = fast
        if not mo4w:
            self.quantize_w = Quantizer(wbits,wfmt,wnorm,0,w_scale_method,enable_int=self.enable_int,
                                        model_path=path+'w/', fast=fast, cali_mode=cali_mode)
            self.quantize_a = Quantizer(abits,afmt,anorm,cali_num,a_scale_method,enable_int=self.enable_int,
                                        model_path=path+'a/', fast=fast, cali_mode=cali_mode)
        self.wfmt = wfmt
        self.wnorm = wnorm
        self.cali_num = cali_num
        self.w_scale_method = w_scale_method
        self.quant_pos = quant_pos
        self.limited = limited
        self.cali_mode=cali_mode
       
    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data)
        try:
            self.bias = nn.Parameter(conv.bias.data)
        except AttributeError:
            self.bias = None

        if self.mo4w:
            self.quantize_aw = Quantizer(self.wbits,self.wfmt,self.wnorm,self.cali_num,self.w_scale_method,
                                            [self.weight, self.stride, self.padding, self.dilation, self.groups], 
                                            enable_int=self.enable_int, model_path=self.path+'w/', fast=self.fast,
                                            limited=self.limited, cali_mode=self.cali_mode)
        
    def forward(self, input):
        weight = self.weight
        try:
            in_shape = input.shape
        except:
            in_shape = input.tensors.shape
        if self.mo4w:
            if self.quantize_aw.quantize.inited:
                input, weight = self.quantize_aw.quantize.quant(input, weight)
            else:
                input, weight = self.quantize_aw.quantize.cali(input, self.use_weight_quant, self.use_act_quant)
            return F.conv2d(input, weight,
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)

        if self.quant_pos == 'out':
            if self.use_weight_quant:
                    weight = self.quantize_w(weight)
            if self.use_act_quant:
                return self.quantize_a(F.conv2d(input, weight, 
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups))
            else:
                return F.conv2d(input, weight, 
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)

        if self.use_weight_quant:
            weight = self.quantize_w(weight)
        if self.use_act_quant:
            input = self.quantize_a(input)
        return F.conv2d(input, weight, 
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
                
class QuantizedLinear(nn.Module):
    def __init__(self, wbits=0, wfmt='int',wnorm=2,abits=0, afmt='int',anorm=2, \
        cali_num=200,w_scale_method='max',a_scale_method='max',quant_pos='out', 
        enable_int=True, path="", mo4w=False, limited=False, fast=False, cali_mode='global'):
        super(QuantizedLinear, self).__init__()#in_features, out_features, bias)
        self.path = path
        self.enable_int=enable_int
        self.wbits = wbits
        self.abits = abits
        self.use_weight_quant = True
        self.use_act_quant = True
        self.fast = fast
        self.mo4w = mo4w
        self.enable_int = enable_int
        if not mo4w:
            self.quantize_w = Quantizer(wbits,wfmt,wnorm,0,w_scale_method,enable_int=self.enable_int,
                                        model_path=path+'w/',fast=fast, cali_mode=cali_mode)
            self.quantize_a = Quantizer(abits,afmt,anorm,cali_num,a_scale_method,enable_int=self.enable_int,
                                        model_path=path+'a/',fast=fast, cali_mode=cali_mode)
        self.wfmt = wfmt
        self.wnorm = wnorm
        self.cali_num = cali_num
        self.w_scale_method = w_scale_method
        self.quant_pos = quant_pos
        self.limited = limited
        self.cali_mode=cali_mode
    
    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data)
        # self.weight = nn.Parameter(linear.weight.data.clone())
        try:
            self.bias = nn.Parameter(linear.bias.data)
            # self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

        if self.mo4w:
            self.quantize_aw = Quantizer(self.wbits,self.wfmt,self.wnorm,self.cali_num,self.w_scale_method,
                        [self.weight],enable_int=self.enable_int,model_path=self.path+'w/', fast=self.fast,
                        limited=self.limited, cali_mode=self.cali_mode)

        
    def forward(self, input):
        
        weight = self.weight
        if self.mo4w:
            if self.quantize_aw.quantize.inited:
                input, weight = self.quantize_aw.quantize.quant(input, weight)
            else:
                input, weight = self.quantize_aw.quantize.cali(input, self.use_weight_quant, self.use_act_quant)
            return F.linear(input, weight, self.bias)

        
        if self.quant_pos == 'out':
            if self.use_weight_quant:
                weight = self.quantize_w(weight)

            if self.use_act_quant:
                return self.quantize_a(F.linear(input, weight, self.bias))
            else:
                return F.linear(input, weight, self.bias)
                    
        if self.use_weight_quant:
            weight = self.quantize_w(weight)
        if self.use_act_quant:
            input = self.quantize_a(input)
        return F.linear(input, weight, self.bias)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class QuantLayer(nn.Module):
    def __init__(self, abits=0,afmt='int',anorm=2,layer=None,
                 cali_num=200, scale_method='max',quant_pos='out', 
                 enable_int=True,path="",fast=False, cali_mode='global'):
        super(QuantLayer, self).__init__()
        self.path = path
        self.enable_int=enable_int
        self.layer = layer
        self.abits = abits
        self.use_act_quant = True
        self.quant_pos = quant_pos
        if quant_pos == 'out':
            if hasattr(layer, 'out_num'):
                self.out_num = layer.out_num
                assert self.out_num == 2
            else:
                self.out_num = 1
            self.quantize_a = nn.ModuleList([Quantizer(abits,afmt,anorm,cali_num,scale_method,
                                                       enable_int=self.enable_int,model_path=path+'a/',
                                                       fast=fast, cali_mode=cali_mode) for _ in range(self.out_num)])
        else:
            if hasattr(layer, 'in_num'):
                self.in_num = layer.in_num
            else:
                self.in_num = 1
            self.quantize_a = nn.ModuleList([Quantizer(abits,afmt,anorm,cali_num,scale_method,
                                                       enable_int=self.enable_int,model_path=path+'a/',
                                                       fast=fast, cali_mode=cali_mode) for _ in range(self.in_num)])

    def forward(self, *input):
        if self.quant_pos == 'out':
            output = self.layer(*input)
            if isinstance(output, tuple):
                output = list(output)
                if self.use_act_quant:
                    for i in range(len(output)):
                        output[i] = self.quantize_a[i](output[i])
                return tuple(output)
            else:
                if self.use_act_quant:
                    output = self.quantize_a[0](output)
                return output
        else:
            if isinstance(input, tuple):
                input = list(input)
                for i in range(len(input)):
                    if self.use_act_quant:
                        input[i] = self.quantize_a[i](input[i])
                input = tuple(input)
            else:
                if self.use_act_quant:
                    input = self.quantize_a[0](input)
            output = self.layer(*input)
            return output

    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant

class QuantLast(nn.Module):
    def __init__(self, abits=0,afmt='int',anorm=2, cali_num=200, scale_method='max',out_num=1,
                 enable_int=True, fast=False, cali_mode='global'):
        super(QuantLast, self).__init__()
        self.use_act_quant = True
        self.quantize_o = nn.ModuleList([Quantizer(abits,afmt,anorm,cali_num,scale_method,
                                                   enable_int=enable_int,fast=fast,cali_mode=cali_mode)] * out_num) 
        
    def forward(self, *x):
        if self.use_act_quant:
            if isinstance(x, tuple):
                x = list(x)
                for i in range(len(x)):
                    x[i] = self.quantize_o[i](x[i])
                x = tuple(x)
                if len(x) == 1:
                    return x[0]
            else:
                x = self.quantize_o[0](x)
        return x
    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant

class QuantFirst(nn.Module):
    def __init__(self, block, abits=0, afmt='int', anorm=2, cali_num=200, scale_method='max', in_num=1,
                 enable_int=True, fast=False, cali_mode='global'):
        super(QuantFirst, self).__init__()
        self.quantize_i = nn.ModuleList([Quantizer(abits,afmt,anorm,cali_num,scale_method,
                                                   enable_int=enable_int,fast=fast,cali_mode=cali_mode)] * in_num)
        self.use_act_quant = True 
        self.block = block
    def forward(self, *x):
        if self.use_act_quant:
            if isinstance(x, tuple):
                x = list(x)
                for i in range(len(x)):
                    x[i] = self.quantize_i[i](x[i])
                x = tuple(x)
            else:
                x = self.quantize_i[0](x)

        x = self.block(*x)
        return x
    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant