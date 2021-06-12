#!/usr/bin/env python
import os
from fade.server.base import ServerAgent
import torch
from time import time
from datetime import timedelta
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import logging

os.putenv("LC_ALL", "C.UTF-8")
os.putenv("LANG", "C.UTF-8")
os.putenv("LANGUAGE", "C.UTF-8")
logger = logging.getLogger(__name__)


def train_loop_body(i, seed, server_agent: ServerAgent, device):
    """Major function to create server and run training."""
    cfg = server_agent.full_config
    if cfg.i_rep >= 0 and cfg.i_rep != i:
        return
    # change seed every time.
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"--------------- Running with seed #{i}: {seed} ------------")
    server = server_agent.create_server(i, device=device)
    if cfg.load_model.do:
        server.load_model(hash_name=cfg.load_model.hash_name,
                          to_load=cfg.load_model.load)
        if 'user' not in cfg.load_model.load:
            server.send_parameters(partial=False)

    # Run training with clients
    server.train()

    # The final eval results will not override the one runed inside train().
    # NOTE for Central server, we use use the server model for eval
    res_dict = server.evaluate(reduce_users=False, add_res_to_record=False, return_dict=True,
                               personal=False, full_info=False, model=None)
    res_dict = dict(("g_" + k, v) for k, v in res_dict.items())
    server.log(res_dict, commit=True)
    server.dump_to_file(personal=False, key="user_eval", obj=res_dict)


def eval_loop_body(i, seed, cfgs, server_agent: ServerAgent):
    """Evaluation."""
    if cfgs.i_rep >= 0 and cfgs.i_rep != i:
        return
    # change seed every time.
    torch.manual_seed(seed)
    print(f"---------------Running time: {i}, seed {seed} ------------")
    server = server_agent.create_server(i, device=cfgs.device)
    assert cfgs.load_model.do, "Model is required to be loaded when evaluating."
    print(f"Warning: Load models by specifying hash name as: {cfgs.load_model.hash_name}")
    server.load_model(hash_name=cfgs.load_model.hash_name, to_load=cfgs.load_model.load)
    # Send model to users.
    # NOTE send params will not send states e.g., running mean of BatchNorm.
    if 'user' not in cfgs.load_model.load:
        logger.warning(f"Not load users' models. Sending loaded server model params to users. "
                       f"NOTE: this will not send states of BN's, which "
                       f"may cause low acc in some cases.")
        server.send_parameters(partial=False)

    # The eval results will not override the one runed inside train().
    eval_dict = \
        server.evaluate(reduce_users=False, full_info=True, add_res_to_record=False,
                        model=None, return_dict=True,
                        personal=False)  # NOTE: only evaluate server model.
    # NOTE Remove info that can not be log, e.g., z which may cause inconsistent #sample between users.
    server.dump_to_file(personal=False, key="full_user_eval", obj=eval_dict)
    for info_group in ("train", "test"):
        # DO NOT log the vector info.
        for k in ('pred_group', 'pred_y', 'true_y', 'z'):
            for _id in eval_dict["extra_info"][info_group]:
                eval_dict["extra_info"][info_group][_id].pop(k)
    server.log(eval_dict, commit=True, is_summary=True)


def run(cfgs):
    """Handle the non-critical config options."""
    assert cfgs.i_rep < cfgs.n_rep, f"Found cfgs.i_rep ({cfgs.i_rep}) >= cfgs.n_rep ({cfgs.n_rep})"
    if cfgs.action in ["train", "eval"] and "wandb" in cfgs.logger.loggers \
            and cfgs.n_jobs == 1 and cfgs.i_rep >= 0:
        wandb.init(**cfgs.logger.wandb, reinit=True,
                   config=OmegaConf.to_container(cfgs, resolve=True, enum_to_str=True))
    else:
        cfgs.logger.loggers = [v for v in cfgs.logger.loggers if v != "wandb"]

    # server agent will spawn duplicated servers at run.
    server_agent = ServerAgent(args=cfgs, times=cfgs.n_rep)

    rng = np.random.RandomState(cfgs.seed)
    random_seeds = rng.randint(np.iinfo(np.int32).max, size=10)  # used for repetitions
    if cfgs.action in ["train", "average", "eval", "check_files"]:
        server_agent.preload_data(print_stat=True)

    # choose action
    if cfgs.action == "train":
        for i in range(cfgs.n_rep):
            train_loop_body(i, random_seeds[i], server_agent, cfgs.device)
    elif cfgs.action == "eval":
        for i in range(cfgs.n_rep):
            eval_loop_body(i, random_seeds[i], cfgs, server_agent)
    elif cfgs.action == "check_files":  # check generated files, e.g., saved models.
        server = server_agent.create_server(cfgs.n_rep)
        server.print_config()
        print("Hash name:", server.get_hash_name(include_rep=False))
        server.check_files(verbose=True, personal=False, times=range(cfgs.n_rep))

    if cfgs.action in ["train", "average"]:
        # Average data
        # NOTE: The returned metric value may not include all users if `partial_eval` is true.
        if not (cfgs.action == "train" and cfgs.i_rep >= 0):
            res_stat = server_agent.average_results_and_save(cfgs.n_rep, cfgs.num_glob_iters)
    if cfgs.action in ["train", "eval"]:
        res_stat = {}
    else:
        res_stat = None

    return server_agent.get_hash_name(), res_stat


@hydra.main(config_name="config.yaml", config_path="config")
def app_main(args: DictConfig):
    print("=" * 60)
    print("Summary of training process:")
    # print("Dataset              : {}".format(args.dataset.name))
    print("Algorithm            : {}".format(args.server.name))
    print("Local Model          : {}".format(args.model.name))
    print("Optimizer            : {}".format(args.user.optimizer.name))
    print("Loss                 : {}".format(args.user.loss))
    print("Batch size           : {}".format(args.user.batch_size))
    print("Learning rate        : {}".format(args.user.optimizer.learning_rate))
    print("Moving Average       : {}".format(args.server.beta))
    print("Subset of users      : {}".format(args.server.num_users))
    print("Num of global rounds : {}".format(args.num_glob_iters))
    print("Num of local rounds  : {}".format(args.user.local_epochs))
    print("Partial evaluate?    : {}".format(args.partial_eval))
    print("Device               : {}".format(args.device))
    print("Logging              : {}".format(args.logging))
    print("=" * 60)

    from fade.utils import set_coloredlogs_env
    import coloredlogs
    # logging.basicConfig(level=args.logging.upper())
    set_coloredlogs_env()
    coloredlogs.install(level=args.logging.upper())

    start_time = time()
    exp_name, res_stat = run(cfgs=args)
    end_time = time()
    elapsed = str(timedelta(seconds=end_time - start_time))
    print(f"--- Elapsed: {elapsed} secs ---")


def entry():
    # this function is required to allow automatic detection of the module name when running
    # from a binary script.
    # it should be called from the executable script and not the hydra.main() function directly.
    app_main()


if __name__ == '__main__':
    app_main()
