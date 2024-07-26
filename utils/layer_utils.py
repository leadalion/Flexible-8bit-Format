import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class Eltwise(nn.Module):
    def __init__(self, mode, coef1=1,coef2=1,in_num=2):
        super(Eltwise, self).__init__()
        self.mode = mode
        self.in_num = in_num
        self.coef1 = coef1
        self.coef2 = coef2

    def forward(self, x1,x2):
        if self.mode == 'Add':
            return x1*self.coef1+x2*self.coef2
        elif self.mode == 'PROD':
            return x1*x2
        else:
            print('not support')

class Matmul(nn.Module):
    def __init__(self, in_num=2):
        super(Matmul, self).__init__()
        self.in_num = in_num

    def forward(self, x1,x2):
        return torch.matmul(x1, x2)
    
class Concat(nn.Module):
    def __init__(self, axis, in_num=2):
        super(Concat, self).__init__()
        self.axis = axis
        self.in_num = in_num

    def forward(self, *x):
        # print(type(x))
        return torch.cat(x,self.axis)

class Slice(nn.Module):
    def __init__(self, slice_point, axis,out_num=2):
        super(Slice, self).__init__()
        self.slice_point = slice_point
        self.axis = axis
        self.out_num = out_num

    def forward(self, x):
        if self.axis == 1 and len(self.slice_point)==1:
        # x1, x2 = torch.split(x,self.slice_point,dim=self.axis)
            x1 = x[:,:self.slice_point[0],:,:]
            x2 = x[:,self.slice_point[0]:,:,:]
        return x1, x2

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Adap_avg_pool(nn.Module):
    def forward(self, input):
        return F.adaptive_avg_pool2d(input, 1)

class Interp(nn.Module):
    def __init__(self, size, mode, align_corners=True):
        super(Interp, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if self.mode == 'bilinear':
            return F.interpolate(x,size=self.size,mode="bilinear",align_corners=self.align_corners)
        elif self.mode == 'nearest':
            return F.interpolate(x,size=self.size,mode="nearest")
        else:
            print('not support!')


class V3Sigmoid(nn.Module):
    def forward(self, x): 
        # m = x*0.1<0?0:(x*0.1>1.0?1.0:x*0.1);
        x = x*0.1
        return torch.clamp(x, 0, 1)