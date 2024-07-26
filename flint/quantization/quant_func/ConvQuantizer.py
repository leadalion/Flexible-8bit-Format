from functools import reduce
import numpy as np
import torch
import torch.nn.functional as F

from flint.quantization.quant_func.quant_func import *

class ConvWeightQuantizer(FloatQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inited = False
        self.weight = self.conv_param[0]
        self.stride =  self.conv_param[1]
        self.padding = self.conv_param[2]
        self.dilation = self.conv_param[3] 
        self.groups = self.conv_param[4]

        self.quantizer_a = None
        self.quantizer_w = None
        
        self.format_info = [f'e{self.bit[i][0]}m{self.bit[i][1]}' for i in range(len(self.bit))]
        for i in range(len(self.bit)):
            self.__dict__['quantizer_w' + str(i)] = FpQuantizer(
                exp=self.bit[i][0], man=self.bit[i][1],
                mode=self.forward_mode)
        if self.enable_int:
            self.__dict__['quantizer_w' + str(len(self.bit))] = IntQuantizer(
                bit=self.nbit)
            self.format_info.append(f'int{self.nbit}')

    def cali(self, x, use_weight_quant, use_act_quant):
        self.use_weight_quant = use_weight_quant
        self.use_act_quant = use_act_quant

        if self.inited is False:
            if self.mode == 'global':    
                self.accum_data(x)
                if len(self.accum) < self.cali_num:
                    return x, self.weight
                self.x_scale, self.w_scale = self.init_quantization_scale(self.accum)
                
            elif self.mode == "batch":
                self.accum_batch(x)
                if len(self.accum_score)*len(x) < self.cali_num:
                    return x, self.weight
                self.x_scale, self.w_scale = self.static_scale()
            else:
                assert False, "wrong mode for {} ".format(self)
            self.inited = True
        else:
            print("Weight: {} has been inited but still calibrate!".format(self.accum_path))
        return x, self.weight

    def test_quant(self, x, xmax, quantizer, quantizer_w):
        max_range = quantizer.max
        x_scale = xmax.clamp_(min=1e-5, max=1e20) / max_range
        
        max_range = quantizer_w.max
        w_scale = self.w_absmax.clamp_(min=1e-5, max=1e20) / max_range
        # fast compute score
      
        if self.fast:
            if hasattr(quantizer, 'exp'):
                log_x = x.abs().log2()
                e, m = quantizer.exp, quantizer.man
                b = 2 ** (e - 1) - 1
                h_b = b - math.log(x_scale, 2)
                ss = log_x.add_(h_b).floor().sub_(h_b).sub_(m).clamp_(min=1-h_b-m)
                x_score = ss.sum()
            else:
                x_score = x_scale.log2() * x.numel()
            if hasattr(quantizer_w, 'exp'):
                log_x = self.weight.abs().log2()
                e, m = quantizer_w.exp, quantizer_w.man
                b = 2 ** (e - 1) - 1
                h_b = b - math.log(w_scale, 2)
                ss = log_x.add_(h_b).floor().sub_(h_b).sub_(m).clamp_(min=1-h_b-m)
                w_score = ss.sum()
            else:
                w_score = w_scale.log2() * self.weight.numel()
            score = x_score + w_score
        else:        
            x_q = quantizer(x/x_scale)*x_scale
            w_q = quantizer_w(self.weight/w_scale)*w_scale
            score = self.score_cal(x, x_q, self.weight, w_q)
                
        return score, x_scale, w_scale
    
    def init_quantization_scale(self, x):
        x = x[0] if self.cali_num == 0 else torch.stack(x)
        x_absmax = self.find_max(x)
        
        if not hasattr(self, 'w_absmax'):
            self.w_absmax = self.find_max(self.weight)
        best_score = np.inf
        best_scale = [0,0]
        for n in range(len(self.format_info)):
            for w in range(len(self.format_info)):  
                score, x_scale, w_scale = self.test_quant(x, x_absmax, self.__dict__['quantizer' + str(n)], 
                                                          self.__dict__['quantizer_w' + str(w)])
                if self.limited and self.enable_int:
                    if (n==len(self.bit)) ^ (w==len(self.bit)):
                        score = np.inf 
                if score < best_score:
                    best_scale[0] = x_scale
                    best_scale[1] = w_scale
                    best_score = score
                    self.quantizer_a = self.__dict__['quantizer' + str(n)]
                    a_numeric = self.format_info[n]
                    self.quantizer_w = self.__dict__['quantizer_w' + str(w)]
                    w_numeric = self.format_info[w]
                    self.numeric = f'w {w_numeric} a {a_numeric}'
        return best_scale[0], best_scale[1]
    
    def quant(self, x, w):
        if self.use_act_quant:
            x_q = self.quantizer_a(x/self.x_scale)*self.x_scale
        else:
            x_q = x
        if self.use_weight_quant:
            w_q = self.quantizer_w(w/self.w_scale)*self.w_scale
        else:
            w_q = w
        return x_q, w_q
    
    
    def init_batch_scale(self, x):
        x_absmax = self.find_max(x)
        a = []
        o = []
        
        for n in range(len(self.format_info)):
            for m in range(len(self.format_info)):
                score, x_scale, w_scale = self.test_quant(x, x_absmax, self.__dict__['quantizer' + str(n)], 
                                                          self.__dict__['quantizer_w' + str(m)])
                if self.limited and self.enable_int:
                    if (n==len(self.bit)) ^ (m==len(self.bit)):
                        score = np.inf
                        
                o.append(score)
                a.append([x_scale, w_scale])
        return a, o

    def score_cal(self, x, x_q, w, w_q):
        out = F.conv2d(x, w, None,self.stride, self.padding, self.dilation, self.groups)
        out_q = F.conv2d(x_q, w_q, None,self.stride, self.padding, self.dilation, self.groups)
        score = lp_loss(out, out_q, p=self.norm, reduction='all')
        return score
    
    def accum_batch(self, x):
        if not hasattr(self, 'accum_scale'):
            self.accum_scale = []
        if not hasattr(self, 'accum_score'):
            self.accum_score = []
            
        if not hasattr(self, 'w_absmax'):
            self.w_absmax = self.find_max(self.weight)
        # print(len(self.accum_scale))
        best_scale, best_score = self.init_batch_scale(x)
        # print(best_scale)
        assert len(best_scale)>0, "Skip module: {} for unkown reason".format(self.accum_path)
        self.accum_scale.append(best_scale)
        self.accum_score.append(best_score)
        

    def static_scale(self):
        b_len = len(self.format_info)
        bit_len = b_len**2
        bit_score = np.zeros(bit_len)
        bit_scale = np.zeros((bit_len, 2))
        for batch_score, batch_scale in zip(self.accum_score, self.accum_scale):
            for i in range(bit_len):
                bit_score[i] += batch_score[i]
                bit_scale[i][0] = max(bit_scale[i][0], batch_scale[i][0])
                bit_scale[i][1] = max(bit_scale[i][1], batch_scale[i][1])
        
        best_choice = bit_score.argmin()
        best_choice_2dim = np.unravel_index(best_choice, (b_len, b_len))
        assert best_choice_2dim[0] * b_len +  best_choice_2dim[1] == best_choice
        self.quantizer_a = self.__dict__['quantizer' + str(best_choice_2dim[0])]
        self.quantizer_w = self.__dict__['quantizer' + str(best_choice_2dim[1])]
        a_numeric = self.format_info[best_choice_2dim[0]]
        w_numeric = self.format_info[best_choice_2dim[1]]
        self.numeric = f'w {w_numeric} a {a_numeric}'
        # logger.info("{}:{}".format(self.accum_path, best_choice))
        return bit_scale[best_choice][0], bit_scale[best_choice][1]
        
    