import torch.nn.functional as F
import torch
import numpy as np
import math
from flint.quantization import * 

def test(net, test_loader, net2=None, test_mode=None, net_flag=None, logger=None, args=None):
    net.eval()
    if test_mode == 'cos':
        if net2:
            net2.eval()
        else:
            net2 = copy.deepcopy(net)
            net2.reset_scion()
        cos = [0]
        for i, data in enumerate(test_loader):
            image_info = []
            if isinstance(data, list):
                image_info = data[-1]
                data = data[0]
            data = data.cuda()
            with torch.no_grad():
                output1 = net([data, image_info])
                output2 = net2([data, image_info])
            if isinstance(output1, tuple):
                multi = len(output1)
                if i == 0:
                    cos = [0]*multi
                for j in range(multi):
                    for b in range(output1[0].shape[0]):
                        cos[j] += cos_dis(output1[j][b].detach().cpu().numpy(),output2[j][b].detach().cpu().numpy())
            else:
                out1 = output1.detach().cpu().numpy()
                out2 = output2.detach().cpu().numpy()
                for b in range(out1.shape[0]):
                    cos[0] += cos_dis(out1[b],out2[b])
        for n in range(len(cos)):
            cos[n] = cos[n] / len(test_loader.dataset) 
        if logger:
            logger.write('Cos-Similarity',cos)
        
        accuracy = sum(cos) / len(cos)
        return accuracy

    else:
        if args.dataset in ['mmcls', 'mmdet', 'mmseg']:
            if isinstance(net, QuantModel):
                acc = net.model.evaluate(test_loader)
            else:
                acc = net.evaluate(test_loader)
            return acc
        else:
            correct = 0
            count = 0
            net.eval()
            for i, data in enumerate(test_loader):
                if isinstance(data, dict):
                    data, target = data['image'].cuda(), data['gt_bboxes'].cuda()
                else:
                    data, target = data[0].cuda(), data[1].cuda()
                with torch.no_grad():
                    output = net(data)
                if isinstance(output, dict):
                    output = output['logits']
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += output.size(0)
            acc = correct / count
            return acc

def cos_dis(a, b):
    a = a.flatten()
    b = b.flatten()
    u = np.sum(a * b)
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    cos = u/d
    if d == 0:
        return 0
    else:
        return cos
