import numpy as np
from fade.server.base import Server
from fade.utils import _log_time_usage


class FedAdv(Server):
    """Federated Adversarial Server assuming:
    * Each user from different adversarial groups, e.g., real vs fake images, male vs female.
    * The group indicates the adversarial group.
    """
    if_personal_local_adaptation = False

    def train(self):
        loss = []
        only_online_users = True
        glob_iter = -1
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")

            if hasattr(self.full_config.server, "rev_lambda_warmup_iter"):
                rev_lambda_warmup_iter = self.full_config.server.rev_lambda_warmup_iter
            else:
                rev_lambda_warmup_iter = 0
            if 0 < rev_lambda_warmup_iter < 1:
                rev_lambda_warmup_iter *= self.num_glob_iters
            progress = max(float(glob_iter - rev_lambda_warmup_iter) / self.num_glob_iters, 0)
            # progress = float(glob_iter) / self.num_glob_iters
            rev_lambda = 2. / (1. + np.exp(-10. * progress)) - 1  # from 0 to 1. Half-Sigmoid
            print(f"## rev_lambda: {rev_lambda}")

            # loss_ = 0
            if len(self.online_user_idxs) >= 1:
                self.send_parameters(glob_iter=glob_iter)
                print(f"Online: {len(self.online_user_idxs)}/{len(self.users)}")
            else:
                print(f"Local training.")
                only_online_users = False

            self.selected_users = self.select_users(glob_iter, self.num_users,
                                                    only_online_users=only_online_users)
            eval_users = self.selected_users if self.partial_eval else self.users

            with _log_time_usage():
                _do_save = False
                if hasattr(self.full_config, 'eval_freq'):
                    if glob_iter % self.full_config.eval_freq == 0:
                        _do_evaluation = True
                        _do_save = True
                    else:
                        _do_evaluation = False
                else:
                    _do_evaluation = True

                if _do_evaluation:
                    # Evaluate model each iteration
                    if hasattr(self.full_config, 'snapshot') and self.full_config.snapshot:
                        raise RuntimeError(f"Not support snapshot")
                    else:
                        eval_dict = self.evaluate(eval_users, reduce_users=self.partial_eval,
                                                  full_info=False, return_dict=True)
                    eval_dict = dict(("g_" + k, v) for k, v in eval_dict.items())
                    self.log(eval_dict, commit=False)
                if _do_save:
                    self.save_model()

            with _log_time_usage("train and aggregate"):
                if hasattr(self.user_cfg, 'no_local_model') and self.user_cfg.no_local_model:
                    raise RuntimeError(f"Not support no_local_model.")
                else:
                    self.train_users(rev_lambda=rev_lambda)

                if hasattr(self.full_config.server, 'sync_optimizer') and self.full_config.server.sync_optimizer:
                    assert len(self.selected_users) == 1, \
                        "For copying user's opt states, only one selected user is allowed."
                    sel_user = self.selected_users[0]
                    for user in self.users:
                        if user.id != sel_user.id:  # ignore same user.
                            user.optimizer.load_state_dict(sel_user.optimizer.state_dict())

            self.log({"global epoch": glob_iter, "rev_lambda": rev_lambda}, commit=True)
        if len(self.online_user_idxs) >= 1:
            self.send_parameters(glob_iter=glob_iter+1)
        self.save_results()
        self.save_model()

    def train_users(self, **user_train_kwargs):
        """Train users and aggregate parameters.
        If fair_update is required, then aggregation will be weighted by softmax-ed losses.
        """
        user_losses = []
        for user in self.selected_users:
            losses = user.train(**user_train_kwargs)  # * user.train_samples
            user_losses.append(losses[0])  # Only keep the first one
        if hasattr(self.full_config.server, 'fair_update') and self.full_config.server.fair_update:
            group_losses = []
            for loss_ in user_losses:
                group_losses.append(loss_['group_loss'][0].item())
            total_group_loss = np.sum(group_losses)
            weights = [gl / total_group_loss for gl in group_losses]
            self.personalized_aggregate_parameters(weights=weights)
        else:
            self.personalized_aggregate_parameters()
