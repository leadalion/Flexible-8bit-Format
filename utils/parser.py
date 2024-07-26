import torch
import collections
import numpy as np
from utils.metric import cos_dis
import os

from dataset.dataset_utils import get_dataloader

def parse_dataset(args):
    args.train_loader = get_dataloader(args, train_flag=True)
    args.test_loader = get_dataloader(args, train_flag=False)
                            
def parse_net(args):
    
    def resnet18_cifar100(args):
        from network.resnet import resnet18
        args.cnn = resnet18()
        params_t = torch.load(args.ckpt)
        args.cnn.load_state_dict(params_t,strict=True)

    def mbv2_cifar100(args):
        from network.mobilenetv2_cifar100 import mobilenetv2
        args.cnn = mobilenetv2()
        params_t = torch.load(args.ckpt)
        args.cnn.load_state_dict(params_t,strict=True)

    def mbv2_imnt(args):
        from network.mobilenetv2_imagenet import mobilenet_v2
        args.cnn = mobilenet_v2()
        params = torch.load(args.ckpt)
        args.cnn.load_state_dict(params,strict=True)
    
    def mbv3_small(args):
        import torchvision
        args.cnn = torchvision.models.mobilenet_v3_large(pretrained=True)
        
    def efficientNet_b3(args):
        args.cnn = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b3', pretrained=True)
        
    def regnet600m_imnt(args):
        from network.regnet import regnetx_600m
        args.cnn = regnetx_600m(
            pretrained=False,
            num_classes=1000,
            task='classification',
        )
        params = torch.load(args.ckpt)
        args.cnn.load_state_dict(params,strict=True)

    def regnet3200m_imnt(args):
        from network.regnet import regnetx_3200m
        args.cnn = regnetx_3200m(
            pretrained=True,
            num_classes=1000,
            task='classification',
        )

    def vit_imnt(args): 
        from network.vit import VisionTransformer
        args.cnn = VisionTransformer(
                 image_size=args.input_shape[-2:],
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1)

        state_dict = torch.load(args.ckpt)['state_dict']
        args.cnn.load_state_dict(state_dict)


    config_dict = { 
      ('resnet18',      'CIFAR100'): [resnet18_cifar100, (3,32,32)],
      ('mobilenetv2',   'CIFAR100'): [mbv2_cifar100,     (3,32,32)], #[1, 1, 1, 1, 1, 1, 1, 1, 1]
      ('mobilenetv2',   'ImageNet'): [mbv2_imnt,         (3,224,224), 32], #[1, 1, 2, 3, 4, 3, 3, 2, 1]
      ('resnet18',      'CIFAR100'): [resnet18_cifar100, (3,32,32)],
      ('resnet18',      'CIFAR100'): [resnet18_cifar100, (3,32,32)],
      ('vit',           'ImageNet'): [vit_imnt,          (3,384,384), ],
      ('mbv3_small',    'ImageNet'): [mbv3_small,        (3,384,384), ],
      ('regnet600m',    'ImageNet'): [regnet600m_imnt,   (3,224,224), 32],
      ('regnet3200m',   'ImageNet'): [regnet3200m_imnt,  (3,224,224), 32],
      ('efficientNet_b3','ImageNet'): [efficientNet_b3,  (3,300,300), 32],
      }

    config = config_dict[(args.net, args.dataset)]
    args.input_shape = config[1]
    args.crop_len = 0
    if len(config) > 2:
        args.crop_len = config[2]
    config[0](args)
    args.cnn.cuda().eval()