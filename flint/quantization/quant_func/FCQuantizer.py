import torch.nn.functional as F

from flint.quantization.quant_func.quant_func import *
from flint.quantization.quant_func.ConvQuantizer import ConvWeightQuantizer

class FCWeightQuantizer(ConvWeightQuantizer):
    def __init__(self, *args, **kwargs):
        super(ConvWeightQuantizer, self).__init__(*args, **kwargs)
        self.inited = False
        self.weight = self.conv_param[0]
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

    def score_cal(self, x, x_q, w, w_q):
        out = F.linear(x, w)
        out_q = F.linear(x_q, w_q)
        score = lp_loss(out, out_q, p=self.norm, reduction='all')
        return score

