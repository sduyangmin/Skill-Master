# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =


def _resolve_worker_tmpdir():
    """Choose a persistent tmp root for TextWorld/Fast Downward artifacts."""
    configured = os.environ.get("SKILLRL_TMPDIR") or os.environ.get("TMPDIR")
    if configured:
        return configured
    return os.path.join(os.path.expanduser("~"), "tmp", "skillrl_tmp")


def _configure_worker_tmpdir():
    """Force tempfile users onto a spacious persistent directory instead of /tmp."""
    tmp_root = _resolve_worker_tmpdir()
    os.makedirs(tmp_root, exist_ok=True)
    os.environ["TMPDIR"] = tmp_root
    os.environ["TMP"] = tmp_root
    os.environ["TEMP"] = tmp_root
    tempfile.tempdir = tmp_root
    return tmp_root


def _patch_fast_downward_load_lib(tmp_root: str):
    """
    Monkey patch fast_downward.load_lib to avoid TemporaryDirectory cleanup races.

    The upstream implementation copies libdownward.so into a TemporaryDirectory and
    deletes it immediately after dlopen(). Under concurrent ALFWorld resets this can
    intermittently fail with "Directory not empty". We instead copy the shared
    library into a worker-local persistent cache directory and load it from there.
    """
    try:
        import fast_downward
        import fast_downward.interface as fd_interface
    except Exception:
        return

    if getattr(fd_interface, "_skillrl_patched_load_lib", False):
        return

    cache_root = os.path.join(tmp_root, "fast_downward_cache", f"pid_{os.getpid()}")
    os.makedirs(cache_root, exist_ok=True)
    cached_lib_path = os.path.join(cache_root, "libdownward.so")

    def _persistent_load_lib():
        if not os.path.isfile(fd_interface.DOWNWARD_LIB_PATH):
            raise RuntimeError(f"Can't find: {fd_interface.DOWNWARD_LIB_PATH}")

        if not os.path.isfile(cached_lib_path):
            shutil.copyfile(fd_interface.DOWNWARD_LIB_PATH, cached_lib_path)

        downward_lib = fd_interface.cdll.LoadLibrary(cached_lib_path)

        downward_lib.load_sas.argtypes = [fd_interface.c_char_p]
        downward_lib.load_sas.restype = None

        downward_lib.load_sas_replan.argtypes = [fd_interface.c_char_p]
        downward_lib.load_sas_replan.restype = None

        downward_lib.cleanup.argtypes = []
        downward_lib.cleanup.restype = None

        downward_lib.get_applicable_operators_count.argtypes = []
        downward_lib.get_applicable_operators_count.restype = int
        downward_lib.get_applicable_operators.argtypes = [fd_interface.POINTER(fd_interface.Operator)]
        downward_lib.get_applicable_operators.restype = None

        downward_lib.get_state_size.argtypes = []
        downward_lib.get_state_size.restype = int
        downward_lib.get_state.argtypes = [fd_interface.POINTER(fd_interface.Atom)]
        downward_lib.get_state.restype = None

        downward_lib.apply_operator.argtypes = [fd_interface.c_int, fd_interface.POINTER(fd_interface.Atom)]
        downward_lib.apply_operator.restype = int

        downward_lib.check_goal.argtypes = []
        downward_lib.check_goal.restype = fd_interface.c_bool

        downward_lib.solve.argtypes = [fd_interface.c_bool]
        downward_lib.solve.restype = fd_interface.c_bool

        downward_lib.solve_sas.argtypes = [fd_interface.c_char_p, fd_interface.c_bool]
        downward_lib.solve_sas.restype = fd_interface.c_bool

        downward_lib.replan.argtypes = [fd_interface.c_bool]
        downward_lib.replan.restype = fd_interface.c_bool

        downward_lib.get_last_plan_length.argtypes = []
        downward_lib.get_last_plan_length.restype = int

        downward_lib.get_last_plan.argtypes = [fd_interface.POINTER(fd_interface.Operator)]
        downward_lib.get_last_plan.restype = None

        downward_lib.check_solution.argtypes = [fd_interface.c_int, fd_interface.POINTER(fd_interface.Operator)]
        downward_lib.check_solution.restype = fd_interface.c_bool

        return downward_lib

    fd_interface.load_lib = _persistent_load_lib
    fast_downward.load_lib = _persistent_load_lib
    fd_interface._skillrl_patched_load_lib = True
    fd_interface._skillrl_cached_lib_path = cached_lib_path

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.base_env = base_env
        self.seed = seed
        self.default_game_files = list(getattr(base_env, "game_files", []))
        self.current_gamefile = None
        self.env = None
        self.tmpdir = _configure_worker_tmpdir()
        _patch_fast_downward_load_lib(self.tmpdir)
        self._init_env()

    def _init_env(self, gamefile=None):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

        if gamefile:
            self.base_env.game_files = [gamefile]
            self.base_env.num_games = 1
        elif self.default_game_files:
            self.base_env.game_files = list(self.default_game_files)
            self.base_env.num_games = len(self.default_game_files)

        self.env = self.base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(self.seed)
        self.current_gamefile = gamefile
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self, gamefile=None):
        """Reset the environment"""
        if gamefile != self.current_gamefile:
            self._init_env(gamefile=gamefile)
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self, kwargs=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []
        gamefiles = self._extract_gamefiles(kwargs)

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            gamefile = gamefiles[i] if gamefiles is not None and i < len(gamefiles) else None
            future = worker.reset.remote(gamefile=gamefile)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def _extract_gamefiles(self, kwargs):
        if kwargs is None:
            return None

        gamefiles = []
        for item in kwargs:
            gamefile = None
            if isinstance(item, dict):
                gamefile = item.get("gamefile")
            elif hasattr(item, "get"):
                gamefile = item.get("gamefile")
            gamefiles.append(gamefile)

        if not any(gamefiles):
            return None
        return gamefiles

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
