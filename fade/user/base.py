import torch
import os
import sys
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from typing import List, Iterable

from fade.data.utils import make_static_dataset_copy, get_dataloader_options


class UserAgent(object):
    def __init__(self, id: str, group=None, batch_size=0, learning_rate=0, beta=0,
                 lamda=0, local_epochs=0, optimizer="sgd", K=None, personal_learning_rate=None, loss="xent", label_mode="supervised"):
        self.id = id  # integer
        self.group, self.group_name = group  # integer, str
        self.label_mode = label_mode
        self.batch_size = batch_size
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        if isinstance(optimizer, DictConfig):
            self.optimizer_name = optimizer.name
            self.learning_rate = optimizer.learning_rate
            self.personal_learning_rate = optimizer.personal_learning_rate  # personal learning
            self.optimizer_config = optimizer
        else:
            self.optimizer_name = optimizer
            self.learning_rate = learning_rate
            self.personal_learning_rate = personal_learning_rate  # personal learning

        # params for local adaptation
        self.K = K  # #iteration of local adaptation. rate.

        # to fill by suclass
        self.model = None

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))


def get_loss(name, num_classes=10):
    if name == "xent":
        return nn.CrossEntropyLoss()
    elif name == "bce":  # for binary classification or Mult-task binary classification.
        return nn.BCEWithLogitsLoss()
    elif name == "mse":
        return nn.MSELoss()
    elif name == "sxent":
        from .shot_digit_loss import CrossEntropyLabelSmooth
        return CrossEntropyLabelSmooth(num_classes, epsilon=0.1)
    else:
        raise ValueError(name)


class User(UserAgent):
    """
    Base class for users in federated learning.
    """
    def train(self):
        """Train model locally."""
        raise NotImplementedError()

    def __init__(self, id, train_data, test_data, model, batch_size=0, learning_rate=0, beta=0,
                 lamda=0, local_epochs=0, optimizer="sgd", K=None, personal_learning_rate=None, loss="xent",
                 group=None, label_mode="supervised",
                 num_glob_iters=None,
                 # hydra config compact
                 name="", privacy={}, total_local_epochs=-1, disable_opt=False,
                 no_local_model=False, data_num_workers=0, drop_last=True,
                 ):
        super().__init__(id, group, batch_size, learning_rate, beta,
                         lamda, local_epochs, optimizer, K, personal_learning_rate, loss, label_mode)
        # This is the model container. The param will be dynamically replaced case by case.
        self.model = model[0] if no_local_model else copy.deepcopy(model[0])  # type: nn.Module
        self.device = next(self.model.parameters()).device
        self.model_name = model[1]

        self.train_samples = len(train_data)
        assert self.train_samples > 0, f"self.train_samples: {self.train_samples}"
        self.test_samples = len(test_data)
        assert self.test_samples > 0, f"self.test_samples: {self.test_samples}"
        self.batch_size = min(self.batch_size, self.train_samples)
        # assert self.batch_size < self.train_samples, f"self.batch_size > self.train_samples: {self.batch_size} < {self.train_samples}"
        self.no_local_model = no_local_model

        if loss == "sxent":
            assert hasattr(self.model, 'n_class')
            self.loss = get_loss(loss, num_classes=self.model.n_class)
        else:
            self.loss = get_loss(loss)

        # drop_last = len(train_data) > self.batch_size
        self.batch_size = self.batch_size if self.batch_size > 0 else self.train_samples
        self.train_data = train_data
        train_dataloader_options = get_dataloader_options(train_data)
        # FIXME ad-hoc I use drop_last=True for the DBM dataest only.
        self.trainloader = DataLoader(train_data, self.batch_size, drop_last=drop_last, shuffle=True,
                                      num_workers=data_num_workers, pin_memory=True,
                                      **train_dataloader_options)
        # FIXME testloader is not used.
        self.batch_size = min(self.test_samples, self.batch_size)
        # assert self.batch_size <= self.test_samples, f"self.batch_size > self.test_samples: {self.batch_size} <= {self.test_samples}"
        # if self.test_samples % self.batch_size > 0:
        # FIXME this is not used because `get_next_testbatch` was not used.
        self.test_data = test_data
        test_dataloader_options = get_dataloader_options(test_data)
        self.testloader = DataLoader(test_data, self.batch_size if self.batch_size > 0 else self.test_samples, drop_last=False, shuffle=True,
                                      num_workers=data_num_workers, pin_memory=True,
                                     **test_dataloader_options)

        self.static_train_data = make_static_dataset_copy(self.train_data)
        self.static_trainloader = DataLoader(self.static_train_data, self.batch_size*3, drop_last=False, shuffle=False,
                                      num_workers=data_num_workers, pin_memory=True)

        # Limit the memory usage. If larger model is used, the limit should ba adjusted down.
        max_memory_limit = 1e7
        # NOTE Because of batch norm, the shuffle will make difference a lot.
        self.testloaderfull = DataLoader(
            test_data, min(len(test_data), self.batch_size * 2),
            drop_last=False, shuffle=True, num_workers=data_num_workers,
            **test_dataloader_options)
        self.trainloaderfull = DataLoader(
            train_data, min(len(test_data), self.batch_size * 2),  # min(self.train_samples, int(max_memory_limit/np.prod(train_data[0][0].shape))),
            drop_last=False, shuffle=True, num_workers=data_num_workers,
            **train_dataloader_options)

        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # /// Model parameters stored for different purpose. ///
        # The local model is to store the user's local model.
        self.local_model_params = None if self.no_local_model else \
            copy.deepcopy(list(self.model.parameters()))

        # UPDATE (@jyhong): In original code, the personalized_model or *_bar are intialized
        #   but they are not used. If a User do not have a personalized model, then it should not
        #   maintain such a variable. In addition, I rename the variable to fit its meaning.
        self.personalized_model_params = None  # copy.deepcopy(list(self.model.parameters()))

        self.num_glob_iters = num_glob_iters
        self.max_iters = self.num_glob_iters / self.local_epochs
        self.relabel_interval = self.max_iters // 15

        if not disable_opt:
            self.init_optimizers()

    def init_optimizers(self):

        if self.optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            # self.sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        elif self.optimizer_name.lower() == "sgd_sch":
            opt_kwargs = OmegaConf.to_container(self.optimizer_config,
                                                resolve=True, enum_to_str=True)
            del opt_kwargs['name']
            del opt_kwargs['learning_rate']
            if 'personal_learning_rate' in opt_kwargs:
                del opt_kwargs['personal_learning_rate']
            opt_kwargs['lr'] = self.learning_rate
            opt_kwargs.setdefault('momentum', 0.9)
            opt_kwargs.setdefault('nesterov', True)
            opt_kwargs.setdefault('weight_decay', .001)
            if hasattr(self.model, 'get_param_group_with_lr'):
                param_group = self.model.get_param_group_with_lr(**opt_kwargs)

                self.optimizer = torch.optim.SGD(param_group)
            else:
                param_group = self.model.parameters()
                self.optimizer = torch.optim.SGD(param_group, **opt_kwargs)
            assert self.num_glob_iters is not None, "num_glob_iters is required for sgd scheduler."
            # print(f"#### self.num_glob_iters {self.num_glob_iters} "
            #       f"self.local_epochs {self.local_epochs}")
            self.sch = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda ep: 1. / (
                        1 + 10. * ep / self.num_glob_iters / self.local_epochs) ** 0.75)
            # self.sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        elif self.optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "adam_sch":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
            self.sch = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda ep: 1. / (
                        1 + 10. * ep / self.num_glob_iters / self.local_epochs) ** 0.75)
        elif self.optimizer_name.lower() == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f"optimizer: {self.optimizer_name}")

    def save_model(self, root_path):
        if not self.no_local_model:
            # model_path = os.path.join("models", self.dataset)
            self.load_model_parameters(self.local_model_params)
            torch.save(self.model.state_dict(), os.path.join(root_path, self.id + ".pt"))

    def load_model(self, root_path):
        """Load model from file."""
        if not self.no_local_model:
            self.model.load_state_dict(torch.load(os.path.join(root_path, self.id + ".pt")))
            self.clone_model_paramenter(self.model.parameters(), self.local_model_params)

    def can_join_for_train(self):
        """Return true if the user can join in for training."""
        return True

    def set_parameters(self, model: nn.Module):
        self.clone_model_paramenter(model.parameters(), self.model.parameters())
        if not self.no_local_model:
            self.clone_model_paramenter(model.parameters(), self.local_model_params)

    def set_shared_parameters(self, model):
        if self.no_local_model:
            raise RuntimeError(f"Try to only set shared parameters when no local model is "
                               f"predefined.")
        # make sure the *private param* in local model is not override by the
        # self.model (at copy from self.model->local later).
        self.clone_model_paramenter(self.local_model_params, self.model.parameters())
        self.clone_model_paramenter(model.get_shared_parameters(),
                                    self.model.get_shared_parameters())
        # NOTE: this will also copy the private params.
        self.clone_model_paramenter(self.model.parameters(), self.local_model_params)

    def get_parameters(self):
        return self.model.parameters()

    def clone_model_paramenter(self, src_param: Iterable[nn.Parameter], dst_param: Iterable[nn.Parameter]):
        for src_p, dst_p in zip(src_param, dst_param):
            assert dst_p.data.shape == src_p.data.shape, \
                f"Cannot copy because shape not matching. Source param shape is: {src_p.shape}. " \
                f"Target param shape is {dst_p.shape}"
            # dst_p.data = src_p.data.clone()
            dst_p.data.copy_(src_p)
        return dst_param

    def load_model_parameters(self, new_params: Iterable[nn.Parameter]):
        """Load `new_params` into the `self.model`."""
        assert new_params is not None, "Try to load new_params which is None."
        for param, new_param in zip(self.model.parameters(), new_params):
            # Correctness verification please see `src/ipynb/Verify_model_copy.ipynb`
            param.data.copy_(new_param)

    def _compute_extra_forward_info(self, extra_info, full_info, x, return_logits=False, model=None,
                                    true_y=None):
        if model is None:
            model = self.model
        if hasattr(self, "sep_coef"):
            opt = {"a": self.sep_coef}
        else:
            opt = {}

        from fade.model.split import SplitEncoder
        if full_info and (isinstance(model, SplitEncoder) or (hasattr(model, "encode") and hasattr(model, "decode"))):
            z = model.encode(x, **opt)
            extra_info.setdefault('z', []).append(z.data.cpu().numpy())
            output = model.decode(z, **opt)

            if hasattr(model, 'predict_task'):
                if hasattr(self, 'group_loss') and self.group_loss in ('cdan', 'sq_cdan'):
                    from .cdan_loss import CDAN_predict_task
                    softmax_out = F.softmax(output, dim=1)
                    task_output = CDAN_predict_task(z, softmax_out, model,
                                                   alpha=1.)
                else:
                    task_output = model.predict_task(z, 1.)
                if task_output.shape[1] > 1:
                    pred_group = torch.argmax(task_output, dim=1)
                else:
                    # FIXME Only valid for BCELossWithLogits
                    pred_group = (task_output > 0.).int().view((-1,))
                extra_info.setdefault('pred_group', []).append(pred_group.data.cpu().numpy())
        else:
            output = model(x, **opt)
        pred_y = torch.argmax(output, dim=1)
        if full_info:
            extra_info.setdefault('pred_y', []).append(pred_y.data.cpu().numpy())
            # Because train set will be shuffled, we save the true y for consistence.
            extra_info.setdefault('true_y', []).append(true_y.data.cpu().numpy())
        if return_logits:
            return output, extra_info
        else:
            return pred_y, extra_info

    def _test(self, full_info=False, model=None, verbose=1):
        """Run test on the current `self.model`. Load the wanted params before call this.s"""
        if model is None:
            model = self.model
        with torch.no_grad():
            model.eval()
            # self.load_model_parameters(self.local_model_params)
            n_correct = 0
            n_sample = 0
            extra_info = {}
            for x, y in (
                    tqdm(self.testloaderfull, desc=f"{self.id} eval test", file=sys.stdout) if verbose > 0 else
                    self.testloaderfull
            ):
            # for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output, extra_info = self._compute_extra_forward_info(
                    extra_info, full_info, x, model=model, return_logits=True, true_y=y)
                if isinstance(self.loss, nn.MSELoss):
                    n_correct += nn.MSELoss(reduction='sum')(output.view_as(y), y).item()
                else:
                    if isinstance(self.loss, nn.BCEWithLogitsLoss):
                        pred_y = (output > 0.).int()
                    elif isinstance(self.loss, nn.BCELoss):
                        pred_y = (output > 0.5).int()
                    else:
                        pred_y = torch.argmax(output, dim=1)
                    if model.n_class == 2 or len(y.shape) > 1:
                        for k, v in self._compute_class_rel(pred_y, y).items():
                            extra_info.setdefault(k, 0)
                            extra_info[k] += v

                    if len(y.shape) > 1:
                        n_correct += (torch.sum(torch.mean((pred_y == y).float(), dim=1))).item()
                    else:
                        n_correct += (torch.sum((pred_y == y).int())).item()
                n_sample += y.shape[0]
                # @loss += self.loss(output, y)
                # print(self.id + ", Test Accuracy:", n_correct / y.shape[0] )
                # print(self.id + ", Test Loss:", loss)
            for k in extra_info:
                if isinstance(extra_info[k], list):
                    extra_info[k] = np.concatenate(extra_info[k])
            return n_correct, n_sample, extra_info

    def compute_loss(self, X, y):
        raise NotImplementedError()
        # losses = {}
        # output = self.model(X)
        # if isinstance(self.loss, nn.MSELoss):
        #     output = output.view_as(y)
        # losses["task_loss"] = (self.loss(output, y).item(), 1.)
        # return losses

    def _train_error_and_loss(self, full_info=False, model=None, verbose=1):
        if model is None:
            model = self.model
        with torch.no_grad():
            model.eval()
            n_correct = 0
            loss_sum = 0
            extra_info = {}  # {'pred_y': [], 'z_sh': [], 'z_pr': []}
            n_sample = 0
            for x, y in (
                    tqdm(self.trainloaderfull, desc=f"{self.id} eval_tr", file=sys.stdout)
                    if verbose > 0 else self.trainloaderfull
            ):
                x, y = x.to(self.device), y.to(self.device)
                output, extra_info = self._compute_extra_forward_info(extra_info, full_info, x, return_logits=True, model=model, true_y=y)  #.append(output.data.cpu().numpy())
                if isinstance(self.loss, nn.MSELoss):
                    output = output.view_as(y)
                if isinstance(self.loss, nn.MSELoss):
                    n_correct += nn.MSELoss(reduction='sum')(output, y).item()
                else:
                    if isinstance(self.loss, nn.BCEWithLogitsLoss):
                        pred_y = (output > 0.).int()
                    elif isinstance(self.loss, nn.BCELoss):
                        pred_y = (output > 0.5).int()
                    else:
                        pred_y = torch.argmax(output, dim=1)
                    if model.n_class == 2 or len(y.shape) > 1:
                        for k, v in self._compute_class_rel(pred_y, y).items():
                            extra_info.setdefault(k, 0)
                            extra_info[k] += v
                    if len(y.shape) > 1:
                        n_correct += (torch.sum(torch.mean((pred_y == y).float(), dim=1))).item()
                    else:
                        n_correct += (torch.sum((pred_y == y).int())).item()
                    # raise ValueError(f"unknown loss: {self.loss}")
                # print(f"### y max min: {y.max().item()}, {y.min().item()} pred y {torch.max(output, dim=1)[:5]}, {y[:5]}, {torch.argmax(output, dim=1)[:5]==y[:5]}")
                if isinstance(self.loss, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                    y = y.float()
                cur_loss = self.loss(output, y).item()
                assert not np.isnan(cur_loss), f"Found nan loss at user {self.id} with y = {y} and model output {output}."
                loss_sum += cur_loss * len(y)
                n_sample += len(y)
            # if full_info:
            for k in extra_info:
                if isinstance(extra_info[k], list):
                    extra_info[k] = np.concatenate(extra_info[k])
            return n_correct, loss_sum, n_sample, extra_info
            # else:
            #     return n_correct, loss_sum, n_sample

    def _compute_class_rel(self, pred_y, y):
        ret = {
            # negative class
            'Neg': (y == 0).sum(0),
            'PredNeg': (pred_y == 0).sum(0),
            'FP': torch.logical_and(y == 0, pred_y.squeeze() == 1).sum(0),
            # positive class
            'Pos': (y == 1).sum(0),
            'PredPos': (pred_y == 1).sum(0),
            'FN': torch.logical_and(y == 1, pred_y.squeeze() == 0).sum(0)
        }
        if len(y.shape) > 1:  # multi-task
            for k in ret:
                ret[k] = ret[k].data.cpu().numpy()
        else:  # single-task
            for k in ret:
                ret[k] = ret[k].item()
        return ret

    def test(self, full_info=False, personal=False, model=None, **kwargs):
        """Test current local model."""
        if model is None and not self.no_local_model:
            self.load_model_parameters(self.personalized_model_params if personal
                                       else self.local_model_params)
        ret = self._test(full_info=full_info, model=model, **kwargs)
        if model is None:
            if personal and not self.no_local_model:
                # TODO the loading may not be necessary. But just to be safe.
                self.load_model_parameters(self.local_model_params)
        return ret

    def train_error_and_loss(self, full_info=False, personal=False, model=None, **kwargs):
        """Test current local model."""
        if model is None and not self.no_local_model:
            self.load_model_parameters(self.personalized_model_params if personal
                                       else self.local_model_params)
        ret = self._train_error_and_loss(full_info=full_info, model=model, **kwargs)
        if model is None:
            if personal and not self.no_local_model:
                self.load_model_parameters(self.local_model_params)
        return ret

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            try:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y) = next(self.iter_trainloader)
            except StopIteration as e:
                print(f"Failed to restart data loader.")
                print(f"  self.trainloader is of length: {len(self.trainloader)}")
                raise e
        return X.to(self.device), y.to(self.device)

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            try:
                # restart the generator if the previous generator is exhausted.
                self.iter_testloader = iter(self.testloader)
                (X, y) = next(self.iter_testloader)
            except StopIteration as e:
                print(f"Failed to restart data loader.")
                print(f"  self.testloader is of length: {len(self.testloader)}")
                raise e
        return X.to(self.device), y.to(self.device)

    def has_sharable_model(self):
        """Return True if local model needs to be aggregated for Fed average."""
        return True

    def num_batch(self, subset):
        """How many batches the local loader has."""
        if subset == 'train':
            return len(self.trainloader)
        elif subset == 'test':
            return len(self.testloader)
        elif subset == 'train_full':
            return len(self.trainloaderfull)
        elif subset == 'test_full':
            return len(self.testloaderfull)
        else:
            raise ValueError(f"subset: {subset}")
