import torch

from .adv import SplitAdvNet
from .shot import init_weights
from .utils import freeze_model


class AdultDNNSplit(SplitAdvNet):
    """
    Three branch network with GRL.
    """
    def __deepcopy__(self, memodict={}):
        new_model = self.__class__(mid_dim=self.mid_dim,
                                   n_class=self.n_class,
                                   n_task=self.n_task, in_channel=self.in_channel,
                                   rev_lambda_scale=self.rev_lambda_scale,
                                   freeze_backbone=self.freeze_backbone,
                                   freeze_decoder=self.freeze_decoder,
                                   disable_bn_stat=self.disable_bn_stat,
                                   CDAN_task=self.CDAN_task,
                                   ).to('cuda')
        if hasattr(self, 'feature_extractor'):
            new_model.feature_extractor.load_state_dict(self.feature_extractor.state_dict())
        # # NOTE this may ignore some class args.
        new_model.encoder.load_state_dict(self.encoder.state_dict())
        new_model.decoder.load_state_dict(self.decoder.state_dict())
        new_model.task_decoder.load_state_dict(self.task_decoder.state_dict())
        return new_model

    def __init__(self, mid_dim=512, n_class=10, n_task=0,
                 in_channel=3, rev_lambda_scale=1., freeze_backbone=True, freeze_decoder=False,
                 disable_bn_stat=False, CDAN_task=False):
        '''
        Args:
            alpha: L = L_utility - alpha * L_adversarial
        '''
        super().__init__(mid_dim=mid_dim, n_class=n_class, n_task=n_task,
                         rev_lambda_scale=rev_lambda_scale, freeze_backbone=freeze_backbone,
                         freeze_decoder=freeze_decoder, disable_bn_stat=disable_bn_stat)
        self.in_channel = in_channel
        self.CDAN_task = CDAN_task

        if self.CDAN_task:
            task_fea_dim = mid_dim * self.n_class
        else:
            task_fea_dim = mid_dim

        # f: shared feature extractor
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(110, 100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
        )
        freeze_model(self.feature_extractor, self.freeze_backbone)
        self.encoder = torch.nn.Linear(100, self.mid_dim)

        # g: utility task classifier:
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.mid_dim, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(32, 2)
        )
        freeze_model(self.decoder, self.freeze_decoder)

        if self.n_task > 0:
            # h: adversarial/privacy task classifier:
            self.task_decoder = torch.nn.Sequential(
                torch.nn.Linear(task_fea_dim, 32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=0.25),
                torch.nn.Linear(32, self.n_task)
            )
            self.task_decoder.apply(init_weights)
            self.shared += [p for p in self.task_decoder.parameters(recurse=True)]
        self.shared += [p for p in self.encoder.parameters(recurse=True)]
        self.private += [p for p in self.decoder.parameters(recurse=True)]
