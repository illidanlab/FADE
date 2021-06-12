import torch
import torch.nn as nn
import wandb

from .base import User
from fade.model.split import SplitEncoder


class GenericUser(User):
    """Implementation for FedAvg clients"""
    def __init__(self, *args, negative_coef=1., **kwargs):
        super().__init__(*args, **kwargs)
        # self.personalized_model_params = deepcopy(list(self.model.parameters()))
        # self.personal_model = deepcopy(self.model)
        self.negative_coef = negative_coef

        self.is_privacy_budget_out = False

    def can_join_for_train(self):
        return not self.is_privacy_budget_out

    def has_sharable_model(self):
        return not self.is_privacy_budget_out and super().has_sharable_model()

    def compute_loss(self, X, y, model=None):
        if model is None:
            model = self.model
        losses = {}
        if isinstance(model, SplitEncoder):
            Z = model.encode(X)
            output = model.decode(Z)
        else:
            output = model(X)
        if isinstance(self.loss, nn.MSELoss):
            output = output.view_as(y)
        if isinstance(self.loss, (nn.BCELoss, nn.BCEWithLogitsLoss)):
            y = y.float()
        task_loss = self.loss(output, y)
        if model.n_class <= 2:
            losses["task_loss"] = (task_loss, self.negative_coef if 0 in y else 1.)
        else:
            losses["task_loss"] = (task_loss, 1.)
        return losses

    def train(self):
        LOSS = []
        if not self.no_local_model:
            self.load_model_parameters(self.local_model_params)
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            if len(y) <= 1:  # will cause error for BN layer.
                continue

            losses = self.compute_loss(X, y)
            LOSS.append(losses)
            loss = 0
            for k, (value, coef) in losses.items():
                wandb.log({f"{self.id} {self.group} " + k: value}, commit=False)
                loss = loss + value * coef

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'sch'):
                self.sch.step()
                wandb.log({f"{self.id} {self.group} " + "lr": self.sch.get_lr()[0]},
                          commit=False)
        if not self.no_local_model:
            self.clone_model_paramenter(self.model.parameters(), self.local_model_params)
        return LOSS

