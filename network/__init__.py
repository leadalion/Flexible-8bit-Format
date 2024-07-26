from .resnet import *
from .mobilenetv2_cifar100 import *
from .mobilenetv2_imagenet import *

def model_entry(net):
    return globals()[net]()