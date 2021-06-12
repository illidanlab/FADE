"""Models for mnist-class datasets."""
import torch
from torch import nn

from .adv import SplitAdvNet
from .utils import GradientReversalFunction, freeze_model


class MnistCnnSplit(SplitAdvNet):
    """CNN split network for Mnist.
    Old name: MnistCNN_pEnc

    Args:
        backbone: One of 'lenet5a' (default), 'lenet5b',
            - `lenet5a` is "modified" version of LeNet5 using ReLU.
            - `lenet5b` is similar to "lenet5a" where we split at the last layer.
        mid_dim: Split dimension size. NOTE this is useless for backbone=lenet5a.
        n_task: If n_task > 0, construct a `task_decoder` and predict task.
        rev_lambda: The param for Reversal Gradient layer in task decoder.
        in_channel: Num of input channels.
    """

    def __deepcopy__(self, memodict={}):
        new_model = self.__class__(backbone=self.backbone, mid_dim=self.mid_dim, n_task=self.n_task,
                                   in_channel=self.in_channel, rev_lambda_scale=self.rev_lambda_scale,
                                   n_class=self.n_class, bottleneck_type=self.bottleneck_type,
                                   freeze_decoder=self.freeze_decoder,
                                   CDAN_task=self.CDAN_task,
                                   disable_bn_stat=self.disable_bn_stat,
                                   ).to('cuda')
        if hasattr(self, 'feature_extractor'):
            new_model.feature_extractor.load_state_dict(self.feature_extractor.state_dict())
        new_model.encoder.load_state_dict(self.encoder.state_dict())
        new_model.decoder.load_state_dict(self.decoder.state_dict())
        new_model.task_decoder.load_state_dict(self.task_decoder.state_dict())
        return new_model

    def __init__(self, backbone="lenet5a", mid_dim=100, n_task=0,
                 in_channel=3, rev_lambda_scale=1., n_class=10,
                 bottleneck_type='bn', freeze_decoder=False,
                 CDAN_task=False, disable_bn_stat=False):
        super().__init__(mid_dim=mid_dim, n_class=n_class, n_task=n_task,
                         rev_lambda_scale=rev_lambda_scale, freeze_backbone=False,
                         freeze_decoder=freeze_decoder,
                         disable_bn_stat=disable_bn_stat)
        self.backbone = backbone
        self.in_channel = in_channel
        self.bottleneck_type = bottleneck_type
        self.CDAN_task = CDAN_task

        if self.CDAN_task:
            task_fea_dim = mid_dim * self.n_class
        else:
            task_fea_dim = mid_dim
        if backbone.lower() == "lenet5c":
            from .shot_digit import LeNetBase, feat_bootleneck, feat_classifier
            base = LeNetBase(self.in_channel)  # NOTE for s2m, use DTN
            # self.feature_extractor = base  # may cause error when load

            netB = feat_bootleneck(type=self.bottleneck_type, feature_dim=base.in_features,
                                   bottleneck_dim=mid_dim)
            self.encoder = nn.Sequential(
                base, netB
            )
            self.decoder = feat_classifier(type='wn', class_num=self.n_class,
                                           bottleneck_dim=mid_dim)
            if freeze_decoder:
                freeze_model(self.decoder)
            if n_task > 0:
                self.task_decoder = nn.Sequential(
                    # GradientReversal(lambda_=rev_lambda),
                    nn.Linear(task_fea_dim, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, n_task)
                )
                self.shared += [p for p in self.task_decoder.parameters(recurse=True)]
        elif backbone.lower() == "dtn":  # used for SVHN
            from .shot_digit import DTNBase, feat_bootleneck, feat_classifier
            base = DTNBase(self.in_channel)
            netB = feat_bootleneck(type=self.bottleneck_type, feature_dim=base.in_features,
                                   bottleneck_dim=mid_dim)
            # self.feature_extractor = base
            self.encoder = nn.Sequential(
                base, netB
            )
            self.decoder = feat_classifier(type='wn', class_num=self.n_class,
                                           bottleneck_dim=mid_dim)
            if freeze_decoder:
                freeze_model(self.decoder)
            if n_task > 0:
                self.task_decoder = nn.Sequential(
                    # GradientReversal(lambda_=rev_lambda),
                    nn.Linear(task_fea_dim, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, n_task)
                )
                self.shared += [p for p in self.task_decoder.parameters(recurse=True)]
        else:
            raise ValueError(f"backbone {backbone}")
        self.shared += [p for p in self.encoder.parameters(recurse=True)]
        self.private += [p for p in self.decoder.parameters(recurse=True)]

        # self.task_clf = nn.Parameter(torch.randn((mid_dim, 1)))

    def can_predict_task(self):
        return hasattr(self, 'task_decoder')

    def encode(self, x: torch.Tensor, a=0.5):
        if len(x.shape) < 4:
            x = torch.reshape(x, (x.shape[0], self.in_channel, 28, 28))
        # print(f"### x shape: {x.shape}")
        return self.encoder(x)

    def decode(self, z, a=0.5) -> torch.Tensor:
        # print(f"### z shape: {z.shape}")
        if self.backbone.lower() == "se":
            z = self.se(z)
            z = z.view(z.shape[0], -1)
            z = self.decoder0(z)
            return self.decoder(z)
        elif self.backbone.lower() in ["lenet5a", "dann15"]:
            z = z.view(z.shape[0], -1)
            return self.decoder(z)
        else:
            return self.decoder(z)
        # F.log_softmax(x, dim=1)

    def predict_task(self, z, rev_lambda):
        # return torch.sum((z - torch.reshape(self.task_clf, (1, -1))) ** 2, dim=1) / z.shape[1]
        if self.backbone.lower() in ["lenet5c", "dtn"]:
            z = GradientReversalFunction.apply(z, rev_lambda * self.rev_lambda_scale)
            return self.task_decoder(z)
        else:
            raise NotImplementedError()
            # return self.task_decoder(F.relu(z))

    def get_shared_submodule(self):
        return self.encoder

    def get_private_submodule(self):
        return self.decoder

    def get_param_group_with_lr(self, lr, param_group=[], **kwargs):
        return [{'params': self.parameters(), 'lr': lr, **kwargs}]
