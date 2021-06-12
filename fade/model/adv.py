import torch
from torch import nn

from typing import Union, Dict
from .split import SplitEncoder
from .utils import GradientReversalFunction
from .shot import init_weights


class SplitAdvNet(SplitEncoder):
    """Split Adversarial Net with classifier and discriminator heads above the encoder.

    Model structure refer to Peng et al. ICLR 2020, Fed Adv Domain Adaptation.

    Args:
        backbone:
            - `alexnet`: Use AlexNet as feature_extractor.
            - `resnet`: Use ResNet101 as feature_extractor.
            We load the model pretrained on ImageNet from torch's rep. The module
            code can be found in https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        mid_dim: Split dimension size. NOTE this is useless for backbone=lenet5a.
        n_task: If n_task > 0, construct a `task_decoder` and predict task.
        rev_lambda: The param for Reversal Gradient layer in task decoder.
        in_channel: Num of input channels.
        disable_bn_stat: If set True, the batch norm layers will not update running mean and std
            both in train and eval mode.
    ```
    """
    def train(self, mode: bool = True):
        super(SplitAdvNet, self).train(mode)

        if self.disable_bn_stat:
            def stop_bn_stat(m):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
            self.apply(stop_bn_stat)
        return self

    def __deepcopy__(self, memodict={}):
        new_model = self.__class__(mid_dim=self.mid_dim, n_class=self.n_class,
                                   n_task=self.n_task,
                                   rev_lambda_scale=self.rev_lambda_scale,
                                   freeze_backbone=self.freeze_backbone,
                                   freeze_decoder=self.freeze_decoder,
                                   disable_bn_stat=self.disable_bn_stat,
                                   ).to('cuda')
        if hasattr(self, 'feature_extractor'):
            new_model.feature_extractor.load_state_dict(self.feature_extractor.state_dict())
        new_model.encoder.load_state_dict(self.encoder.state_dict())
        new_model.decoder.load_state_dict(self.decoder.state_dict())
        new_model.task_decoder.load_state_dict(self.task_decoder.state_dict())
        return new_model

    def load_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                        strict: bool = True):
        # Remove task_encoder keys
        state_dict = dict((k, v) for k, v in state_dict.items() if not k.startswith('task_decoder'))
        ret = super().load_state_dict(state_dict, strict=False)
        # missing_keys, unexpected_keys
        if strict:
            assert len(ret.unexpected_keys) == 0, f"Got unexpected_keys: {ret.unexpected_keys}"
            unexpected_missing_keys = []
            for k in ret.missing_keys:
                if not k.startswith('task_decoder'):
                    unexpected_missing_keys.append(k)
            if len(unexpected_missing_keys) > 0:
                raise RuntimeError(f"Got unexpected missing keys: {unexpected_missing_keys}."
                                   f" Only allow task encoder keys to be missing.")
        return ret

    def __init__(self, mid_dim=512, n_class=10, n_task=0,
                 rev_lambda_scale=1., freeze_backbone=True, freeze_decoder=False,
                 disable_bn_stat=False):
        super().__init__()
        self.mid_dim = mid_dim
        # self.n_class = n_class
        self.n_class = n_class
        self.n_task = n_task
        self.rev_lambda_scale = rev_lambda_scale
        self.freeze_backbone = freeze_backbone
        self.freeze_decoder = freeze_decoder
        self.disable_bn_stat = disable_bn_stat

    def can_predict_task(self):
        return hasattr(self, 'task_decoder')

    def encode(self, x: torch.Tensor, a=0.5):
        if hasattr(self, 'feature_extractor'):
            x = self.feature_extractor(x)
        return self.encoder(x)

    def decode(self, z, a=0.5) -> torch.Tensor:
        return self.decoder(z)

    def predict_task(self, z, rev_lambda) -> torch.Tensor:
        z = GradientReversalFunction.apply(z, rev_lambda * self.rev_lambda_scale)
        # print("### z", z.shape)
        # print("### task_decoder", self.task_decoder)
        return self.task_decoder(z)

    def get_shared_submodule(self):
        return [self.encoder, self.task_decoder]

    def get_private_submodule(self):
        return self.decoder

    def get_param_group_with_lr(self, lr, param_group=[], **kwargs):
        for k, v in self.feature_extractor.named_parameters():
            param_group += [{'params': v, 'lr': lr * 0.1, **kwargs}]
        for k, v in self.encoder.named_parameters():
            param_group += [{'params': v, 'lr': lr, **kwargs}]
        for k, v in self.decoder.named_parameters():
            param_group += [{'params': v, 'lr': lr, **kwargs}]
        for k, v in self.task_decoder.named_parameters():
            param_group += [{'params': v, 'lr': lr, **kwargs}]
        return param_group

    def reset_task_decoder(self):
        """Reset task decoder by re-init."""
        self.task_decoder.apply(init_weights)
