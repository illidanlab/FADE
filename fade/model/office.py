"""For dataset. office+Caltech10

Model structure refer to Peng et al. ICLR 2020, Fed Adv Domain Adaptation.
"""
import torch
from torch import nn
from .utils import GradientReversalFunction, freeze_model
from .shot import init_weights
from .adv import SplitAdvNet


class OfficeCnnSplit(SplitAdvNet):
    """CNN split network for Office + Caltech10 datasets for domain adaptation.

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
    """
    def __deepcopy__(self, memodict={}):
        new_model = self.__class__(backbone=self.backbone, mid_dim=self.mid_dim,
                                   n_class=self.n_class,
                                   n_task=self.n_task, in_channel=self.in_channel,
                                   rev_lambda_scale=self.rev_lambda_scale,
                                   freeze_backbone=self.freeze_backbone,
                                   freeze_decoder=self.freeze_decoder,
                                   disable_bn_stat=self.disable_bn_stat,
                                   bottleneck_type=self.bottleneck_type,
                                   CDAN_task=self.CDAN_task,
                                   pretrained=self.pretrained
                                   ).to('cuda')
        if hasattr(self, 'feature_extractor'):
            new_model.feature_extractor.load_state_dict(self.feature_extractor.state_dict())
        new_model.encoder.load_state_dict(self.encoder.state_dict())
        new_model.decoder.load_state_dict(self.decoder.state_dict())
        new_model.task_decoder.load_state_dict(self.task_decoder.state_dict())
        return new_model

    def __init__(self, backbone="alexnet", mid_dim=512, n_class=10, n_task=0,
                 in_channel=3, rev_lambda_scale=1., freeze_backbone=True, freeze_decoder=False,
                 disable_bn_stat=False, bottleneck_type='bn',
                 CDAN_task=False, pretrained=True):
        super().__init__(mid_dim=mid_dim, n_class=n_class, n_task=n_task,
                         rev_lambda_scale=rev_lambda_scale, freeze_backbone=freeze_backbone,
                         freeze_decoder=freeze_decoder, disable_bn_stat=disable_bn_stat)
        self.backbone = backbone
        self.in_channel = in_channel
        self.bottleneck_type = bottleneck_type
        self.CDAN_task = CDAN_task
        self.pretrained = pretrained

        if self.CDAN_task:
            task_fea_dim = mid_dim * self.n_class
        else:
            task_fea_dim = mid_dim

        if backbone.lower().startswith('resnet'):
            # from torchvision.models import resnet50
            from .shot import ResBase, feat_bootleneck, feat_classifier
            if backbone.startswith('resnet'):
                base = ResBase(res_name=backbone, pretrained=self.pretrained)
            else:
                raise ValueError(f"Invalid backbone: {backbone}")
            bottleneck = feat_bootleneck(type=self.bottleneck_type, feature_dim=base.in_features,
                                         bottleneck_dim=mid_dim)

            freeze_model(base, self.freeze_backbone)
            self.feature_extractor = base
            self.encoder = bottleneck

            self.decoder = feat_classifier(type='wn', class_num=n_class,
                                           bottleneck_dim=mid_dim)
            freeze_model(self.decoder, self.freeze_decoder)

            if n_task > 0:
                if n_task > 1:
                    self.task_decoder = nn.Linear(task_fea_dim, n_task)
                else:
                    self.task_decoder = nn.Sequential(
                        nn.Linear(task_fea_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(256, n_task),
                    )
                    self.task_decoder.apply(init_weights)
                self.shared += [p for p in self.task_decoder.parameters(recurse=True)]
        else:
            raise NotImplementedError(f"backbone {backbone}")
        self.shared += [p for p in self.encoder.parameters(recurse=True)]
        self.private += [p for p in self.decoder.parameters(recurse=True)]

    def encode(self, x: torch.Tensor, a=0.5):
        if hasattr(self, 'feature_extractor'):
            x = self.feature_extractor(x)
        return self.encoder(x)

    def decode(self, z, a=0.5) -> torch.Tensor:
        return self.decoder(z)

    def predict_task(self, z, rev_lambda) -> torch.Tensor:
        z = GradientReversalFunction.apply(z, rev_lambda * self.rev_lambda_scale)
        return self.task_decoder(z)

    def get_current_module_norm(self, mode="grad"):
        """Get submodules' (grad) norms.

        Args:
            mode: 'grad' or 'weight'

        Returns:
            all_norms, a dict
        """
        all_modules = {"enc": self.encoder, "task_dec": self.task_decoder, "dec": self.decoder}
        all_norms = {"enc": 0., "task_dec": 0., "dec": 0.}
        for name, module in all_modules.items():
            for np, p in module.named_parameters():
                if p.grad is not None:
                    if mode == "grad":
                        norm = p.grad.data.norm().item()
                        all_norms[name] += p.grad.data.norm().item()
                    elif mode == "weight":
                        norm = p.data.norm().item()
                    else:
                        raise ValueError(f"mode: {mode}")
                    all_norms[name] += norm
            #         print(f"   ### > {np} norm... {norm}")
            # print(f"   ### model {mode} norm... ", name, all_norms[name])
        all_norms = dict((k+f"_{mode}_norm", v) for k, v in all_norms.items())
        return all_norms

