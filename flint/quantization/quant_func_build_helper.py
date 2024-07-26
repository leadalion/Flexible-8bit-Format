import torch.nn as nn
import torch.nn.functional as F
from flint.quantization.quant_func.quant_func import *
from flint.quantization.quant_func.FCQuantizer import FCWeightQuantizer
from flint.quantization.quant_func.ConvQuantizer import ConvWeightQuantizer

class Quantizer(nn.Module):
    def __init__(self, bits=8, fmt='int', norm=2, 
                 cali_num=200, scale_method='max', conv_param=None, enable_int=True, 
                 model_path="", fast=False, limited=False, cali_mode="global"):
        super(Quantizer, self).__init__()
        if bits == 0:
            self.quantize = nn.Identity()
        else:
            if isinstance(conv_param, list):
                if len(conv_param)>1:
                    self.quantize = ConvWeightQuantizer(bits, fmt, norm, cali_num, scale_method, conv_param, 
                                                        enable_int=enable_int, save_name=model_path, fast=fast,
                                                        limited=limited, cali_mode=cali_mode)
                else:
                    self.quantize = FCWeightQuantizer(bits, fmt, norm, cali_num,scale_method, conv_param, 
                                                      enable_int=enable_int, save_name=model_path, fast=fast,
                                                      limited=limited, cali_mode=cali_mode)
            else:
                self.quantize = FloatQuantizer(bits, fmt, norm, cali_num, scale_method, enable_int=enable_int, 
                                                   save_name=model_path, fast=fast, cali_mode=cali_mode)             

    def forward(self, input):
        return self.quantize(input)
