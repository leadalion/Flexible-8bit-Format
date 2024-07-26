import torch.nn as nn
import torch
from mmcv import Config
from mmcv.ops import RoIPool
from mmcv.parallel import scatter, MMDataParallel
import numpy as np



class mmdecorator(nn.Module):
    def __init__(self, config, zoo):
        super().__init__()
        self.config = Config.fromfile(config)
        self.zoo = zoo
        if zoo == 'mmdet':
            from mmdet.models import build_detector
            from mmdet.apis import single_gpu_test
            self.model = build_detector(self.config.model)
            self.single_gpu_test = single_gpu_test
        elif zoo == 'mmcls':
            from mmcls.models import build_classifier
            from mmcls.apis import single_gpu_test
            self.model = build_classifier(self.config.model)
            self.single_gpu_test = single_gpu_test
        elif zoo == 'mmseg':
            from mmseg.models import build_segmentor
            from mmseg.models import build_segmentor
            self.model = build_segmentor(self.config.model)
            self.single_gpu_test = single_gpu_test
        self.load_ckpt()
        # import pdb;pdb.set_trace()
        self.model = MMDataParallel(self.model)

    def load_ckpt(self):
        if not getattr(self.config, 'resume', None) is None:
            self.model.load_state_dict(torch.load(self.config.resume)['state_dict'], strict=True)
        elif not getattr(self.config, 'load_from', None) is None:
            self.model.load_state_dict(torch.load(self.config.load_from)['state_dict'], strict=True)
        else:
            print("Not load any trained ckpt")
    
    def forward(self, data, **kwargs):
        if self.zoo in ['mmdet', 'mmseg']:
            data['rescale']=True
        with torch.no_grad():
            result = self.model(return_loss=False, **data)
        return result

    @torch.no_grad()
    def evaluate(self, data_loader):
        eval_kwargs = self.config.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
        ]:
            eval_kwargs.pop(key, None)
        outputs = self.single_gpu_test(self.model, data_loader)
        results = data_loader.dataset.evaluate(outputs, **eval_kwargs)
        return results