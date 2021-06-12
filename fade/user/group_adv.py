import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .base import User
from fade.model.split import SplitEncoder

from .shot_digit_loss import Entropy, cluster_estimate_label
from ..data.utils import update_dataset_targets


class GroupAdvUser(User):
    """Implementation for FedAvg clients"""
    def __init__(self, *args, adv_lambda=1., group_loss='bce', relabel_coef=0.,
                 cluster_threshold=10., negative_coef=1.0, group_loss_q=1,
                 group_loss_dro_reg=0., loss_reshape='none',
                 loss_reshape_q=1, clamp_grad=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_lambda = adv_lambda

        self.is_privacy_budget_out = False
        self.group_loss_q = group_loss_q  # used with sq_bce

        self.group_loss = group_loss
        self.relabel_coef = relabel_coef
        self.current_steps = 0
        self.cluster_threshold = cluster_threshold  # use 10 for PDA, 0 for DA.
        self.negative_coef = negative_coef

        # params for DRO or fair resource allocation
        self.group_loss_dro_reg = group_loss_dro_reg
        self.loss_reshape = loss_reshape
        self.loss_reshape_q = loss_reshape_q
        self.clamp_grad = clamp_grad

    def can_join_for_train(self):
        return not self.is_privacy_budget_out

    def has_sharable_model(self):
        return not self.is_privacy_budget_out and super().has_sharable_model()

    def compute_loss(self, X, y, rev_lambda=1., model=None):
        if model is None:
            model = self.model
        losses = {}
        if isinstance(model, SplitEncoder):
            Z = model.encode(X)

            wandb.log({f"{self.id} {self.group} Z mean": torch.mean(Z, dim=0).data.cpu().numpy()}, commit=False)
            wandb.log({f"{self.id} {self.group} Z std": torch.std(Z, dim=0).data.cpu().numpy()}, commit=False)

            assert hasattr(model, "predict_task")
            if self.group_loss in ('cdan', 'sq_cdan'):
                from .cdan_loss import CDAN_predict_task
                output = model.decode(Z)
                # NOTE do not detach softmax_out s.t. we can BP.
                softmax_out = F.softmax(output, dim=1)
                pred_group = CDAN_predict_task(Z, softmax_out, model,
                                               alpha=rev_lambda)
            else:
                pred_group = model.predict_task(Z, rev_lambda=rev_lambda)

            if model.n_task == 1:
                group_label = torch.ones(pred_group.shape[0], dtype=torch.long).fill_(
                    self.group).to(self.device)
                group_acc = torch.mean(((pred_group > 0.).int() == group_label).float()).item()
                wandb.log({f"{self.id} {self.group} group_acc": group_acc}, commit=False)
                # binary classification
                if self.group_loss == 'bce':
                    assert 0 <= self.group < 2
                    assert pred_group.shape[1] == 1, f"pred_group.shape={pred_group.shape}"
                    group_loss = F.binary_cross_entropy_with_logits(pred_group.view(-1,), group_label.float())
                elif self.group_loss == 'sq_bce':
                    assert 0 <= self.group < 2
                    assert pred_group.shape[1] == 1, f"pred_group.shape={pred_group.shape}"
                    group_loss = F.binary_cross_entropy_with_logits(pred_group.view(-1,), group_label.float())
                    # FIXME ad-hoc, the 1/2 is not used previously.
                    group_loss = group_loss ** (self.group_loss_q + 1.) / (1 + self.group_loss_q)
                elif self.group_loss == 'xent':
                    assert pred_group.shape[1] > 1, f"pred_group.shape={pred_group.shape}"
                    group_loss = F.cross_entropy(pred_group, group_label)
                elif self.group_loss in ('cdan', 'sq_cdan'):
                    from .cdan_loss import CDAN
                    group_loss = F.binary_cross_entropy_with_logits(pred_group.view(-1,), group_label.float())
                    group_loss = CDAN(group_loss, softmax_out, group_label.float(),
                                      compute_ent_weights=True, alpha=rev_lambda)
                    if self.group_loss == 'sq_cdan':
                        group_loss = group_loss ** 2 / 2.
                elif self.group_loss == 'none':
                    pass
                else:
                    raise ValueError(f"Invalid group_loss: {self.group_loss} for "
                                     f"{model.n_task} tasks.")
            else:
                group_label = torch.ones(pred_group.shape[0], dtype=torch.long).fill_(
                                                 self.group).to(self.device)
                group_acc = torch.mean((torch.argmax(pred_group) == group_label).float()).item()
                wandb.log({f"{self.id} {self.group} group_acc": group_acc}, commit=False)
                if self.group_loss == 'bce':
                    group_loss = F.cross_entropy(pred_group, group_label)
                    # ic(self.id, group_loss, pred_group, group_label)
                elif self.group_loss == 'sq_bce':
                    group_loss = F.cross_entropy(pred_group, group_label)
                    # ic(self.id, group_loss, pred_group, group_label)
                    # FIXME ad-hoc, the 1/2 is not used previously.
                    group_loss = group_loss ** (self.group_loss_q + 1.) / (1 + self.group_loss_q)
                elif self.group_loss in ('cdan', 'sq_cdan'):
                    from .cdan_loss import CDAN
                    group_loss = F.cross_entropy(pred_group, group_label)
                    group_loss = CDAN(group_loss, softmax_out, group_label.float(),
                                      compute_ent_weights=True, alpha=rev_lambda)
                    if self.group_loss == 'sq_cdan':
                        group_loss = group_loss ** 2 / 2.
                elif self.group_loss == 'none':
                    pass
                else:
                    raise ValueError(f"Invalid group_loss: {self.group_loss} for "
                                     f"{model.n_task} tasks.")
            if self.group_loss != 'none' and self.adv_lambda > 0:
                # # FIXME not used
                if self.group_loss_dro_reg > 0.:
                    losses["group_loss"] = (torch.abs(group_loss - self.group_loss_dro_reg), self.adv_lambda)
                else:
                    # loss = loss + self.adv_lambda * group_loss
                    losses["group_loss"] = (group_loss, self.adv_lambda)

            if self.label_mode == "supervised":
                output = model.decode(Z)
                if isinstance(self.loss, nn.MSELoss):
                    output = output.view_as(y)
                if isinstance(self.loss, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    y = y.float()
                # ic(self.loss(output, y))
                # ic(output, y)

                if self.loss_reshape.lower() == 'dro':  # distributionally robust opt
                    _loss_reduction = self.loss.reduction
                    self.loss.reduction = 'none'
                    task_loss = self.loss(output, y)
                    self.loss.reduction = _loss_reduction
                    task_loss = torch.mean(torch.maximum(task_loss - self.loss_reshape_q,
                                                         torch.zeros_like(task_loss)) ** 2)
                elif self.loss_reshape.lower() == 'fra':  # fair resource allocation
                    assert self.loss_reshape_q >= 0
                    task_loss = self.loss(output, y)
                    task_loss = task_loss ** (self.loss_reshape_q + 1) / (self.loss_reshape_q + 1)
                else:
                    task_loss = self.loss(output, y)

                if model.n_class <= 2:
                    losses["task_loss"] = (task_loss, self.negative_coef if 0 in y else 1.)
                else:
                    losses["task_loss"] = (task_loss, 1.)
            elif self.label_mode == "unsupervised":
                pass
            elif self.label_mode == "self_supervised":  # using Info-Max loss
                output = model.decode(Z)
                out_softmax = F.softmax(output, dim=1)
                # assert isinstance(self.loss, nn.CrossEntropyLoss)
                # losses["im_loss"] = (torch.mean(Entropy(out_softmax)), 1.)
                msoftmax = out_softmax.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                losses["im_loss"] = (torch.mean(Entropy(out_softmax)) - gentropy_loss, 0.)  #  1.) FIXME ad-hoc set as 1.

                if self.relabel_coef > 0:
                    losses["relabel_loss"] = (F.cross_entropy(output, y), self.relabel_coef)
                    # print(f"#### y: {y}")
            else:
                raise ValueError(f"label_mode: {self.label_mode}")
        else:
            raise NotImplementedError(f"Model type is {type(model)}")
        return losses

    def train(self, mode="train", rev_lambda=1.):
        LOSS = []
        if not self.no_local_model:
            self.load_model_parameters(self.local_model_params)

        if self.label_mode == "self_supervised" and self.relabel_coef > 0. and self.current_steps % self.relabel_interval == 0:
            # assert hasattr(self, 'nonshuffle_testloader')
            assert hasattr(self.model, 'n_class')
            self.model.eval()
            labels = cluster_estimate_label(self.static_trainloader, self.model,
                                            class_num=self.model.n_class,
                                            threshold=self.cluster_threshold)
            print(f"### relabel train set for user {self.id}")
            update_dataset_targets(self.train_data, labels)
            self.iter_trainloader = iter(self.trainloader)


        self.model.train()
        flag_large_group_loss = False
        for epoch in range(1, self.local_epochs + 1):
            self.current_steps += 1
            self.model.train()
            X, y = self.get_next_train_batch()
            if len(y) <= 1:
                # raise ValueError(f"len y <=1: {len(y)}")
                # 1 sample will result in error for BN layer.
                print(f"{self.id} Only one sample is in the batch.")
                continue
            self.optimizer.zero_grad()

            if flag_large_group_loss and hasattr(self.model, 'reset_task_decoder'):
                print(f"!! Reset task decoder.")
                self.model.reset_task_decoder()

            losses = self.compute_loss(X, y, rev_lambda=0. if flag_large_group_loss else rev_lambda)
            pre_flag_large_group_loss = flag_large_group_loss
            flag_large_group_loss = ("group_loss" in losses) and (losses["group_loss"][0] > 10)
            LOSS.append(losses)
            loss = 0
            # print(f"## {self.id} {self.group}:", end=" ")
            for k, (value, coef) in losses.items():
                # FIXME When local_epochs > 1, this will result in multiple records in one global wandb step.
                wandb.log({f"{self.id} {self.group} " + k: value}, commit=False)
                print(f"### {self.id} {self.group} " + k, value.item())
                if mode == "pretrain":
                    if k == "group_loss":
                        print(f"### PRETRAIN: Ignore group_loss")
                        continue
                loss = loss + value * coef
                # print(f" {k}: {value} * {coef}", end="; ")
            # print()
            if not isinstance(loss, torch.Tensor):
                print(f"### No loss. Skip backward")
                continue

            self.optimizer.zero_grad()
            loss.backward()

            if self.clamp_grad is not None:
                assert self.clamp_grad > 0
                nn.utils.clip_grad_value_(self.model.task_decoder.parameters(),
                                          clip_value=self.clamp_grad)

            # Log the grad/weight norms of submodules.
            if hasattr(self.model, "get_current_module_norm"):
                for mode in ("grad", "weight"):
                    wandb.log(dict((f"{self.id} {self.group} {k}", v)
                                   for k, v in
                                   self.model.get_current_module_norm(mode=mode).items()),
                              commit=False)

            self.optimizer.step()
            if hasattr(self, "sch"):
                self.sch.step()
                wandb.log({f"{self.id} {self.group} lr": self.sch.get_last_lr()[0]}, commit=False)

            try:
                self.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                # try another call
                self.optimizer.zero_grad()

            if not self.no_local_model:
                self.clone_model_paramenter(self.model.parameters(), self.local_model_params)
        return LOSS

