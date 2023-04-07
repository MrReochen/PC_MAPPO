import os
import sys
import json
import yaml
import torch
import shutil
import argparse
import collections
import numpy as np

from pathlib import Path
from datetime import datetime
from os.path import dirname, abspath
from types import SimpleNamespace as SN


from envs.starcraft2.smac_maps import get_map_params
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from runner.smac_runner_expand import SMACRunner as Runner

torch.autograd.set_detect_anomaly(True)
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_config(base_args):
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    with open(os.path.join(os.path.dirname(__file__), "config", "algs", "{}.yaml".format(base_args.alg)), "r") as f:
        try:
            alg_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(base_args.alg, exc)
    with open(os.path.join(os.path.dirname(__file__), "config", "envs", "sc2.yaml" if base_args.vanilla_env else "mappo_sc2.yaml"), "r") as f:
        try:
            env_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "sc2.yaml error: {}".format(exc)
    with open(os.path.join(os.path.dirname(__file__), "config", "runs", "{}.yaml".format(base_args.run)), "r") as f:
        try:
            run_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(base_args.run, exc)
            
    config_dict = recursive_dict_update(config_dict, env_dict)
    config_dict = recursive_dict_update(config_dict, alg_dict)
    config_dict = recursive_dict_update(config_dict, run_dict)

    if base_args.params != "":
        added_params = base_args.params.split("+")
        for param in added_params:
            k, v = param.split(":")
            if k in config_dict.keys():
                if type(config_dict[k]) == type(True):
                    if v == "True":
                        config_dict[k] = True
                    else:
                        config_dict[k] = False
                else:
                    config_dict[k] = type(config_dict[k])(v)
        config_dict["experiments_params"] = base_args.params

    return SN(**config_dict)

def make_train_env(Env, all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.vanilla_env:
                if all_args.env_name == "StarCraft2":
                    env = Env(**all_args.env_args, seed=all_args.seed + rank * 1000)
                else:
                    print("Can not support the " + all_args.env_name + "environment.")
                    raise NotImplementedError
            else:
                if all_args.env_name == "StarCraft2":
                    env = Env(all_args)
                else:
                    print("Can not support the " + all_args.env_name + "environment.")
                    raise NotImplementedError
                env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(Env, all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.vanilla_env:
                if all_args.env_name == "StarCraft2":
                    env = Env(**all_args.env_args, seed=all_args.seed * 50000 + rank * 10000)
                else:
                    print("Can not support the " + all_args.env_name + "environment.")
                    raise NotImplementedError
            else:
                if all_args.env_name == "StarCraft2":
                    env = Env(all_args)
                else:
                    print("Can not support the " + all_args.env_name + "environment.")
                    raise NotImplementedError
                env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def main(base_args):
    all_args = get_config(base_args) 
    all_args.time_token = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    all_args.vanilla_env = base_args.vanilla_env
    if all_args.vanilla_env:
        all_args.map_name = all_args.env_args["map_name"]
    if base_args.map_name != "":
        all_args.map_name = base_args.map_name
        if all_args.vanilla_env:
            all_args.map_name = base_args.map_name
    
    all_args.episode_length = get_map_params(all_args.map_name)["limit"]

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.purpose != "":
        run_dir = Path(os.path.dirname(__file__)) / "results" / all_args.purpose / all_args.map_name / base_args.alg / base_args.run / all_args.time_token
    else:
        run_dir = Path(os.path.dirname(__file__)) / "results" / all_args.map_name / base_args.alg / base_args.run / all_args.time_token
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.code_backup:
        if not getattr(all_args, "files", False):
            assert False, "You have to determine which files to backup"
        else:
            os.makedirs(os.path.join(run_dir, "code_backup"))
            for k, v in all_args.files.items():
                shutil.copyfile(os.path.join(dirname(abspath(__file__)), v), os.path.join(run_dir, "code_backup", "{}.{}".format(k, v.split('.')[-1])))

    with open(os.path.join(run_dir, "code_backup", "params.json"), "w") as f:
        json.dump(all_args.__dict__, f)

    # env
    if all_args.vanilla_env:
        from smac.env import StarCraft2Env
    else:
        from envs.starcraft2.StarCraft2_Env import StarCraft2Env
    envs = make_train_env(StarCraft2Env, all_args)
    eval_envs = make_eval_env(StarCraft2Env, all_args) if all_args.use_eval else None
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    all_args.num_agents = num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="qmix", help="The algorithm used in this experiment")
    parser.add_argument("--run", type=str, default="run", help="The run config used in this experiment")
    parser.add_argument("--map_name", type=str, default="", help="The running map used in this experiment")
    parser.add_argument("--vanilla_env", action="store_true", default=False, help="Whether this experiment is using smac env")
    parser.add_argument("--params", type=str, default="", help="Set params used in exp here, which will overwrite the params from files")
    base_args = parser.parse_args()

    main(base_args)
