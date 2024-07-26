import os
import argparse
import torch
import numpy as np
import random
import time
from flint.utils.logger import Logger

# from run_hook import hook

def parse():
    parser = argparse.ArgumentParser()
    
    # common param
    parser.add_argument('--model_zoo', type=str, default='None', choices=['None', 'mmdet', 'mmcls', 'mmseg'], help='random seed')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--logger', action='store_true', help='log')

    # data param
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNet',
                        'mmdet','mmcls','mmseg'], help='dataset name (default: CIFAR10)')
    parser.add_argument('--data_path', type=str, default='data/', help='path for store traning data')
    parser.add_argument('--test_path', type=str, default='data/', help='path for store testing data')

    # net param
    parser.add_argument('--ckpt', type=str, default='', help='path for teacher model')
    parser.add_argument('--net', type=str, default='', help='path for teacher model')
    parser.add_argument('--skip_fptest', action='store_true', help='skip float model acc test')

    # run param
    parser.add_argument('--test_mode', type=str, default='acc', choices=['acc', 'cos', 'both'], help='test mode')
    parser.add_argument('--batch_size', type=int, default='32', help='batchsize')
    parser.add_argument('--fold_bn', action='store_true', help='fold bn layers')
    
    #  quant param
    parser.add_argument('--wbit', type=int, default='4', help='bit width for weight')
    parser.add_argument('--abit', type=int, default='4', help='bit width for activation')
    parser.add_argument('--wfmt', type=str, default='int', help='format for weight')
    parser.add_argument('--afmt', type=str, default='int', help='format for activation')
    parser.add_argument('--enable_int', action='store_true', help='mo4w')
    
    parser.add_argument('--quant_pos', type=str, default='out', choices=['out','in'], help='quant function')
    parser.add_argument('--quant_all', action='store_true', help='quant all layers')
    
    parser.add_argument('--mo4w', action='store_true', help='measure out for weight quant')
    parser.add_argument('--limited', action='store_true', help='measure out for weight quant')
    parser.add_argument('--fast', action='store_true', help='measure out for weight quant')
    
    # cali param
    parser.add_argument('--cali_num', type=int, default='200', help='data num for calibration')
    parser.add_argument('--cali_mode', type=str, default='batch', choices=['global','batch'], help='quant function')
    parser.add_argument('--w_scale_method', type=str, default='max', choices=['max','mse','update_max'], help='scale method')
    parser.add_argument('--a_scale_method', type=str, default='mse', choices=['max','meanmax','mse','ema'], help='scale method')
    parser.add_argument('--wnorm', type=float, default='2', help='norm for activation')
    parser.add_argument('--anorm', type=float, default='2', help='norm for activation')
    
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    global args
    args = parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.makedirs('log', exist_ok=True)
    args.model_label = '{}-{}'.format(args.net, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    print(args.model_label)

    logger = Logger('log/{}.txt'.format(args.model_label), state=args.logger)
    logger.write('config', args)
    args.logger = logger
    
    # args limit
    if args.mo4w:
        logger.info("Use no4w mode")
        args.quant_pos = "in"
    else:
        logger.info("Not use no4w mode")
    if args.limited:
        assert args.mo4w 
 
    from test import run_train

    setup_seed(args.seed)

    run_train(args)
    
    args.logger.close()
