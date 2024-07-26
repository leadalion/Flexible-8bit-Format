import torch.nn as nn
import torch.nn.functional as F

from utils.metric import test
from flint.quantization import * 
from flint.quantization.quant_func.quant_func import FloatQuantizer
from utils.layer_utils import *
import copy
from utils.parser import *

def prepare_net(args):

    Quantlist = [nn.Conv2d,
            nn.Linear, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Sigmoid, nn.Softmax, 
            ChannelShuffle, Eltwise, Concat, Slice, Flatten, Adap_avg_pool, Interp, V3Sigmoid]
    
    if not args.skip_fptest:
        accuracy1 = test(args.cnn, args.test_loader, net2=None, test_mode=args.test_mode, net_flag=args.net, logger=args.logger, args=args)
        args.logger.write('CNN-Origin-Accuracy',accuracy1)
        
    qnn = args.cnn
    qnn = QuantModel(qnn, Quantlist, args)
    args.logger.info(f"start init quant scale in {args.cali_mode} mode")
    qnn.set_quant_state(True, True)
    qnn.to('cuda').eval()
    
    qnn.init_scale(args.train_loader, args.cali_num, args.batch_size)
    # args.logger.write('QNN structure', qnn)
    # stop()

    if args.wbit == 8:
        res = {'int8':0, 'e5m2':0, 'e4m3':0,'e3m4':0,'e2m5':0}
    elif args.wbit == 6:
        res = {'int6':0, 'e3m2':0, 'e2m3':0}
    elif args.wbit == 4:
        res = {'int4':0, 'e1m2':0, 'e2m1':0, 'e3m0':0}
    else:
        res = {f'int{args.wbit%32}':0}
    for module in qnn.modules():
        # print(module)
        if isinstance(module, FloatQuantizer):
            if not module.inited:
                args.logger.info("{} is not inited".format(module.accum_path))
            # print(module, module.numeric)
            num = module.numeric.split()
            for n in num:
                if n in res:
                    res[n] += 1
            # res[module.numeric] += 1

    args.logger.write('QNN bit dist', res)
    
    accuracy2 = test(qnn, args.test_loader, net2=qnn, test_mode=args.test_mode, net_flag=args.net, logger=args.logger, args=args)
    args.logger.write('QNN-Origin-Accuracy',accuracy2)
    # if args.test_mode == 'cos':
    #     return accuracy1, res
    return accuracy2, res

    


def run_train(args):
    # parse
    
    if args.model_zoo in ['mmcls','mmdet','mmseg']:
        try:
            from model_zoo.mmzoo_helper import mmdecorator
        except:
            raise ImportError("mmdet framework is not installed, so {} file won't be transfered".format(args.net))
        args.cnn = mmdecorator(args.net, args.model_zoo)
        args.cnn.to('cuda').eval()

    elif args.model_zoo=='None':
        parse_net(args)
    
    parse_dataset(args)
    assert not(args.cali_num % args.batch_size)
    acc = prepare_net(args)

    return acc
