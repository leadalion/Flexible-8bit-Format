import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qtorch import _backend

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if isinstance(pred, list):
        if reduction == 'none':
            return [(_-t).abs().pow(p) for _,t in zip(pred,tgt)]
        else:
            return torch.stack([(_-t).abs().pow(p).mean() for _,t in zip(pred,tgt)]).mean()
    if reduction == 'none':
        return (pred-tgt).abs().pow(p)#.sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def cos_dis(a, b):
    a = a.flatten()
    b = b.flatten()
    u = torch.sum(a * b)
    d = torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
    cos = u/d
    if d == 0:
        return 1
    else:
        return 1-cos

def kl_divergence(x, y):
    return F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
         
class IntQuantizer(nn.Module):
    def __init__(self, bit):
        super().__init__() 
        self.bit = bit
        self.max = self.get_fp_max()
    def get_fp_max(self):
        return 2 ** (self.bit - 1) - 1

    def forward(self, x):
        zero_point = 2 ** (self.bit-1) - 1 
        x_int = round_ste(x) + zero_point
        x_quant = torch.clamp(x_int, 0, 2 ** self.bit - 2)
        return x_quant - zero_point

class FpQuantizer(nn.Module):
    def __init__(self, exp, man, mode='stpu'):
        super().__init__() 
        self.exp = exp
        self.man = man
        self.mode = mode
        self.max = self.get_fp_max()
        self.quant_func = self.forward_quant()
    
    def get_fp_max(self):
        if self.mode == 'stpu_extend':
            if self.exp == 4 and self.man == 3:
                return 448
            elif self.exp == 2 and self.man == 1:
                return 6
            elif self.exp == 1 and self.man == 2:
                return 3.5
            elif self.exp == 3 and self.man == 0:
                return 16
            else:
                assert(0 == 1)
        else:
            if self.exp == 2 and self.man == 5:
                return 3.9375
            elif self.exp == 3 and self.man == 4:
                return 15.5
            elif self.exp == 4 and self.man == 3:
                return 240
            elif self.exp == 5 and self.man == 2:
                return 57344
            elif self.exp == 2 and self.man == 3:
                return 3.75
            elif self.exp == 3 and self.man == 2:
                return 14
            elif self.exp == 2 and self.man == 1:
                return 3
            elif self.exp == 1 and self.man == 2:
                return 1.5
            elif self.exp == 3 and self.man == 0:
                return 8
            
    def forward_quant(self):
        if self.mode == "nearest":
            return lambda x, quant_module: quant_module.float_quantize_nearest(
                x, self.man, self.exp)
        elif self.mode == "extend":
            return lambda x, quant_module: quant_module.float_quantize_extend(
                x, self.man, self.exp)
        elif self.mode == "stpu":
            return lambda x, quant_module: quant_module.float_quantize_stpu(
                x, self.man, self.exp)
        elif self.mode == "stpu_normal":
            return lambda x, quant_module: quant_module.float_quantize_stpu_normal(
                x, self.man, self.exp)
        elif self.mode == "stpu_extend":
            return lambda x, quant_module: quant_module.float_quantize_stpu_extend(
                x, self.man, self.exp)
        elif self.mode == "nodenormal":
            return lambda x, quant_module: quant_module.float_quantize_nodenormal(
                x, self.man, self.exp)
   
    def forward(self, x):
        out, _ = self.quant_func(x.contiguous(), _backend)
        return out

class FloatQuantizer(nn.Module):
    def __init__(self, bit=8, fmt='stpu', norm=2, cali_num=200, scale_method='max', 
                 conv_param=None, enable_int=True, save_name="", 
                 fast=False, limited=False, cali_mode="global"):
        super().__init__()        
        assert cali_mode in ["global", "batch"]
        self.mode = cali_mode
        self.inited = False
  
        self.cali_num = cali_num
        self.norm = norm
        self.forward_mode = fmt#'stpu'#'extend'#"nearest"#'nodenormal'
        self.numeric = 'FP32'
        self.fast = fast
        self.limited = limited
        self.conv_param = conv_param
        self.accum_path = save_name
        
        if bit == 8:
            self.bit = [(2,5),(3,4),(4,3),(5,2)]
        elif bit == 6:
            self.bit = [(2,3),(3,2)]
        elif bit == 4:
            self.bit = [(1,2),(2,1),(3,0)]
        else:
            raise NotImplementedError()
        if fmt == 'int':
            self.bit = []
            assert enable_int
        self.nbit = bit
        self.format_info = [f'e{self.bit[i][0]}m{self.bit[i][1]}' for i in range(len(self.bit))]
        self.enable_int = enable_int
        assert (len(self.bit) or self.enable_int)
        
        for i in range(len(self.bit)):
            self.__dict__['quantizer' + str(i)] = FpQuantizer(
                exp=self.bit[i][0], man=self.bit[i][1],
                mode=self.forward_mode)
        if enable_int:
            self.__dict__['quantizer' + str(len(self.bit))] = IntQuantizer(
                bit=bit)
            self.format_info.append(f'int{bit}')
        
        self.scale_method = scale_method
        self.quantizer = None

    def forward(self, x):
        if self.inited is False:
            # global mode: 1. accum data 2. calculate scale  
            if self.mode == 'global':    
                self.accum_data(x)
                if len(self.accum) < self.cali_num:
                    return x 
                scale = self.init_quantization_scale(self.accum) 
                del self.accum 
                
            # batch mode: 1. cal of a batch .... 2. calculate mean 
            elif self.mode == "batch":
                self.accum_batch(x)
                if len(self.accum_score)*len(x) < self.cali_num:
                    return x
                scale = self.static_scale() 
            self.inited = True
            self.scale = scale
            return x # important
        
        x_q = self.quantizer(x/self.scale)*self.scale
        return x_q
    
    def accum_data(self, x):
        if not hasattr(self, 'accum'):
            self.accum = []
        if self.scale_method in ['mse', 'meanmax', 'max']:
            if self.cali_num == 0:
                self.accum = [x]
            for i in range(x.shape[0]):
                self.accum.append(x[i])
                if len(self.accum) == self.cali_num:
                    break
        else:
            print(self.scale_method)
            raise NotImplementedError

    def test_quant(self, x, xmax, quantizer):
        max_range = quantizer.max
        # scale = xmax / max_range
        scale = xmax.clamp_(min=1e-5, max=1e20) / max_range
        # fast compute score
        if self.fast:
            if hasattr(quantizer, 'exp'):
                log_x = x.abs().log2()
                e, m = quantizer.exp, quantizer.man
                b = 2 ** (e - 1) - 1
                h_b = b - math.log(scale, 2)
                ss = log_x.add_(h_b).floor().sub_(h_b).sub_(m).clamp_(min=1-h_b-m)
                score = ss.sum()
            else:
                score = scale.log2() * x.numel()
        else:          
            x_q = quantizer(x/scale)*scale
            score = lp_loss(x, x_q, p=self.norm, reduction='all')
        return score, scale
    
    def find_max(self, x):
        x = torch.where(torch.isinf(x), torch.full_like(x, 0), x)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        if self.scale_method in ['max', 'mse']:
            x_absmax = x.abs().max()
            
        if self.scale_method == 'meanmax':
            x_ = x.flatten(start_dim=1, end_dim=-1)
            x_absmax = x_.abs().max(dim=-1)[0]
            x_absmax = x_absmax.mean()
        # print(self.accum_path, x_absmax)
        return x_absmax
    
    def init_quantization_scale(self, x):
        best_score = 1000
        # print([_.shape for _ in x],self.accum_path)
        x = x[0] if self.cali_num == 0 else torch.stack(x)
        x_absmax = self.find_max(x)
        
        for i in range(len(self.format_info)):
            score, scale = self.test_quant(x, x_absmax, 
                            self.__dict__['quantizer' + str(i)])
            if self.scale_method == 'mse':
                for n in range(1, 80):
                    new_max = x_absmax * (1.0 - (n * 0.01))
                    curr_score, curr_scale = self.test_quant(x, new_max, 
                                self.__dict__['quantizer' + str(i)])
                    if curr_score < score:
                        score = curr_score
                        scale = curr_scale
   
            if score < best_score:
                best_score = score
                s = scale
                self.quantizer = self.__dict__['quantizer' + str(i)]
                self.numeric = self.format_info[i]
        return s
        
    def init_batch_scale(self, x):
        x_absmax = self.find_max(x)
        a = []
        o = []
        for i in range(len(self.format_info)):
            score, scale = self.test_quant(x, x_absmax, 
                            self.__dict__['quantizer' + str(i)])
            if self.scale_method == 'mse':
                for n in range(1, 80):
                    new_max = x_absmax * (1.0 - (n * 0.01))
                    curr_score, curr_scale = self.test_quant(x, new_max, 
                                self.__dict__['quantizer' + str(i)])
                    if curr_score < score:
                        score = curr_score
                        scale = curr_scale
                
            a.append(scale)
            o.append(score)            
        return a, o

    def accum_batch(self, x):
        if not hasattr(self, 'accum_scale'):
            self.accum_scale = []
        if not hasattr(self, 'accum_score'):
            self.accum_score = []
        
        best_scale, best_score = self.init_batch_scale(x)
        assert len(best_scale)>0, "Skip module: {} for unkown reason".format(self.accum_path)
        self.accum_scale.append(best_scale)
        self.accum_score.append(best_score)

    def static_scale(self):
        # [batch_idx][bit_idx]
        batch_len = len(self.accum_score)
        bit_len = len(self.format_info)
        bit_score = np.zeros(bit_len)
        bit_scale = np.zeros(bit_len)
        for batch_score, batch_scale in zip(self.accum_score, self.accum_scale):
            for i in range(len(self.format_info)):
                bit_score[i] += batch_score[i]
                bit_scale[i] = max(bit_scale[i], batch_scale[i])
        
        best_choice = bit_score.argmin()
        self.quantizer = self.__dict__['quantizer' + str(best_choice)]
        self.numeric = self.format_info[best_choice]
        # logger.info("{}:{}".format(self.accum_path, best_choice))
        return bit_scale[best_choice]

    def reinit(self):
        assert self.inited == True
        self.inited = False
        # del self.accum
        self.accum = []

    def clean(self):
        self.accum = []
    
    def extra_repr(self):
        return self.numeric