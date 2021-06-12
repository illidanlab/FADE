import numpy as np
from fade.server.base import Server
from fade.utils import _log_time_usage


def softmax(logits, temp=1.):
    y = np.minimum(np.exp(logits * temp), 1e4)
    st = y / np.sum(y)
    return st


class FedAvg(Server):
    if_personal_local_adaptation = False

    def train(self):
        loss = []
        only_online_users = True
        glob_iter = -1
        probs = np.ones(len(self.users)) / len(self.users)
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0
            if len(self.online_user_idxs) >= 1:
                self.send_parameters(glob_iter=glob_iter)
                print(f"Online: {len(self.online_user_idxs)}/{len(self.users)}")
            else:
                print(f"Local training.")
                only_online_users = False

            self.selected_users = self.select_users(glob_iter, self.num_users,
                                                    only_online_users=only_online_users,
                                                    probs=probs)
            print("Select users:", [user.id for user in self.selected_users])
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
                    # FIXME Ad-hoc reduce_users=self.partial_eval
                    eval_dict = self.evaluate(eval_users, reduce_users=False, return_dict=True)
                    eval_dict = dict(("g_"+k, v) for k, v in eval_dict.items())
                    self.log(eval_dict, commit=False)
                    print("### g_train_loss", eval_dict["g_train_loss"])
                    print("### g_train_acc", eval_dict["g_train_acc"])
                if _do_save:
                    self.save_model()

            with _log_time_usage("train and aggregate"):
                if hasattr(self.user_cfg, 'no_local_model') and self.user_cfg.no_local_model:
                    self.train_users_online_aggregate()
                else:
                    self.train_users()

            self.log({"global epoch": glob_iter}, commit=True)
        if len(self.online_user_idxs) >= 1:
            self.send_parameters(glob_iter=glob_iter+1)
        self.save_results()
        self.save_model()
