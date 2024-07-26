import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from flint.quantization.fold_bn import search_fold_and_remove_bn
from flint.quantization.quant_layer import *
from flint.quantization.quant_func.quant_func import *

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, quantlist, args):
        super().__init__()
        if args.fold_bn:
            search_fold_and_remove_bn(model)
        self.model = model
        self.quant_list = quantlist
        self.quant(self.model, args)
        if args.quant_all:
            if args.quant_pos == 'out':
                self.quant_first(args)
            else:
                self.quant_last(args)

    def quant(self, model, args, model_path=""):
        if type(model) in self.quant_list:
            if isinstance(model, nn.Conv2d):
                quant_mod = QuantizedConv2d(wbits=args.wbit,wfmt=args.wfmt,wnorm=args.wnorm,abits=args.abit,afmt=args.afmt,anorm=args.anorm,
                                cali_num=args.cali_num, w_scale_method=args.w_scale_method, a_scale_method=args.a_scale_method, 
                                quant_pos=args.quant_pos, enable_int=args.enable_int, path=model_path, mo4w=args.mo4w,
                                limited=args.limited, fast=args.fast, cali_mode=args.cali_mode)
                quant_mod.set_param(model)
                return quant_mod
            elif isinstance(model, nn.Linear):
                quant_mod = QuantizedLinear(wbits=args.wbit,wfmt=args.wfmt,wnorm=args.wnorm,abits=args.abit,afmt=args.afmt,anorm=args.anorm, 
                                cali_num=args.cali_num, w_scale_method=args.w_scale_method, a_scale_method=args.a_scale_method, 
                                quant_pos=args.quant_pos, enable_int=args.enable_int, path=model_path, mo4w=args.mo4w,
                                limited=args.limited, fast=args.fast, cali_mode=args.cali_mode)
                quant_mod.set_param(model)
                return quant_mod
            else:
                quant_mod = QuantLayer(abits=args.abit, afmt=args.afmt, anorm=args.anorm, 
                            layer=model, cali_num=args.cali_num, scale_method=args.a_scale_method, 
                            quant_pos=args.quant_pos, enable_int=args.enable_int, path=model_path,
                            fast=args.fast, cali_mode=args.cali_mode)
                return quant_mod
        elif isinstance(model, (nn.Sequential, nn.ModuleList)):
            for n, m in model.named_children():
                new_submodel = self.quant(m, args, model_path+n+'/')
                new_submodel.path = model_path+n+'/'
                setattr(model, n, new_submodel)
            return model

        else:
            for attr in dir(model):
                try:
                    mod = getattr(model, attr)
                except:
                    continue
                if isinstance(mod, nn.Module):  # and 'norm' not in attr:
                    if attr in ['roi_head', 'base_model']:
                        continue
                    new_submodel = self.quant(mod, args, model_path+attr+'/')
                    new_submodel.path = model_path+attr+'/'
                    setattr(model, attr, new_submodel)
            return model

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, (QuantLayer)):
                m.set_quant_state(act_quant)

    def forward(self,  *args, **kwargs):
        return self.model( *args, **kwargs)

    def quant_first(self, args):
        if hasattr(self.model,"features"):
            block_first = self.model.features[0]
            first_block = QuantFirst(block_first, abits=args.abit,afmt=args.afmt, anorm=args.anorm, 
                                 cali_num=args.cali_num, scale_method=args.a_scale_method, enable_int=args.enable_int, 
                                fast=args.fast, cali_mode=args.cali_mode)
            self.model.features[0] = first_block

    def quant_last(self, args):
        last_block = QuantLast(abits=args.abit,afmt=args.afmt,anorm=args.anorm,  
                               cali_num=args.cali_num, scale_method=args.a_scale_method, out_num=1, enable_int=args.enable_int, 
                                fast=args.fast, cali_mode=args.cali_mode)
        # [!] model out
        self.model = nn.Sequential(self.model, last_block)
        
    def init_scale(self, train_loader, cali_num, bz):
        itr = math.ceil(cali_num / bz)
        # torch.manual_seed(2021) 
        print("start init scaling")
        for i, data in enumerate(train_loader):

            if i < itr:
                print("init activation ({}/{})".format(i, itr))
                if isinstance(data, tuple) or isinstance(data, list):
                    data = data[0].to('cuda')
                elif isinstance(data, dict):
                    if 'image' in data:
                        data['image'] = data['image'].to('cuda')
                    elif 'Q' in data:
                        data = data['Q']
                else:
                    data = data.to('cuda')
                with torch.no_grad():
                    _ = self.model(data)  # .detach()
            else:
                break
