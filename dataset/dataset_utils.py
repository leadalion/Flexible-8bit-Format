import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import torch
from PIL import Image

def get_normalizer(data_set, net=None, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    elif data_set == 'ImageNet':
        if net == 'vit':
            MEAN = (0.5, 0.5, 0.5)
            STD = (0.5, 0.5, 0.5)
        else:
            MEAN = (0.485, 0.456, 0.406)
            STD = (0.229, 0.224, 0.225)
    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return transforms.Normalize(MEAN, STD)


def get_transformer(data_set, net=None, imsize=None, cropsize=None,
                    crop_padding=None, hflip=None, rrc_size=None, center_crop=None):
    transformers = []
    if imsize:
        if net == 'efficientNet_b3':
            transformers.append(transforms.Resize(imsize, interpolation=Image.BICUBIC))
        else:
            transformers.append(transforms.Resize(imsize))
    if cropsize:
        transformers.append(
            transforms.RandomCrop(cropsize, padding=crop_padding))
    if rrc_size:
        transformers.append(transforms.RandomResizedCrop(rrc_size))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip())
    if center_crop:
        transformers.append(transforms.CenterCrop(center_crop))
    
    
    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set, net))

    return transforms.Compose(transformers)

def get_dataloader(args, train_flag=True):

    if args.dataset == 'ImageNet':
        from dataset.imagenet import ImageNetFew, ImageNet
        args.num_class = 1000
        args.num_per_class = 1
        assert args.input_shape[0] == 3 and args.input_shape[1] == args.input_shape[2]

        if train_flag:
            data_loader = DataLoader(
            ImageNetFew(args.data_path, args.num_per_class,
                        transform=get_transformer(args.dataset, args.net,
                                                imsize=args.input_shape[1] + args.crop_len,
                                                center_crop=args.input_shape[1])),
                                                # rrc_size=args.input_shape[-1],
                                                # hflip=True)),
            batch_size=args.batch_size, num_workers=4, shuffle=True)
        else:
            data_loader = DataLoader(
                ImageNet(root=args.data_path, transform=get_transformer(args.dataset, args.net, 
                            imsize=args.input_shape[1] + args.crop_len,#args.input_shape[-2:],
                            center_crop=args.input_shape[1])),#args.input_shape[-2:])),
                            batch_size=args.batch_size, num_workers=4, shuffle=False)

    elif args.dataset in ['mmcls', 'mmdet', 'mmseg']:
        data_loader = get_MMCV_dataloader(args, train_flag)
    
    else:
        raise NotImplementedError

    return data_loader


def get_MMCV_dataloader(args, train_flag):
    try:
        import mmcv
    except:
        raise ImportError("Not install package mmcv")
    cfg = mmcv.Config.fromfile(args.net)
    cfg.data.train['pipeline'] = cfg.data.val['pipeline']
    args.batch_size = cfg.data.samples_per_gpu
    test_dataloader_default_args = dict(
        samples_per_gpu=cfg.data.samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
    if args.dataset == 'mmcls':
        from mmcls.datasets import build_dataloader
        from mmcls.datasets import build_dataset
        if not 'ann_file' in cfg.data.train:
            cfg.data.train['ann_file'] = cfg.data.val['ann_file'].replace('val.txt', 'train.txt')
        extra_kwargs = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }
    elif args.dataset == 'mmdet':
        from mmdet.datasets import build_dataloader
        from mmdet.datasets import build_dataset
        extra_kwargs = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }
    elif args.dataset == 'mmseg':
        from mmseg.datasets import build_dataloader
        from mmseg.datasets import build_dataset
        extra_kwargs = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }
    if train_flag:
        dataset = build_dataset(cfg.data.train)
    else:
        dataset = build_dataset(cfg.data.val)
    # print(extra_kwargs)
    # stop()
    data_loader = build_dataloader(
                    dataset,
                    **extra_kwargs
                    )
    return data_loader
