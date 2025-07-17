from config import TrainConfig, LeRobotAlohaDataConfig, DataConfig
import openpi.models.pi0 as pi0
import weight_loaders
import importlib
import yaml
import os
import gymnasium as gym
import numpy as np
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from description.utils.generate_episode_instructions import generate_episode_descriptions
import openpi.transforms as transforms
from openpi.shared import image_tools
from openpi.models import tokenizer
import random

def create_pi0_base_aloha_rl_lora_config():
    """
    Create a configuration for training the Pi0 model with Aloha dataset using LoRA.
    
    Returns:
        TrainConfig: The configuration object for training.
    """
    return TrainConfig(
        name="pi0_base_aloha_robotwin_rl_lora",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),

        data=LeRobotAlohaDataConfig(
            repo_id="online_rl",  # load the norm stats from pre-trained repo
            adapt_to_pi=False,    #important for robotwin env
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        image_keys=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        discount = 0.97,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),

        num_train_steps=30000, # env step is much faster than training step
        fsdp_devices=1,  # it must be divisible by the number of GPUs


        max_step = 100000000,
        buffer_period=200,  # save the buffer every 10000 steps
        log_period=10,

        training_starts = 100, 
        cta_ratio = 20,
        steps_per_update=1,  # number of steps per update
        cheackpoint_period=200,  # save the checkpoint every 1000 steps

        replay_buffer_capacity = 200000,

    )


class init_config(gym.Env):
    def __init__(self, action_chunk = 50):
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            -np.pi, np.pi, shape=(action_chunk,14), dtype=np.float32  # just retain the last action
        )
        image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                            -np.pi, np.pi, shape=(14,), dtype =np.float32
                        ),  #joints vector
                    
                "image": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8) 
                                for key in image_keys}
                ),
                "image_mask": gym.spaces.Dict(
                    {key: gym.spaces.Discrete(2) for key in image_keys}
                ),
                "tokenized_prompt": gym.spaces.Box(
                    0, np.inf, shape = (48,), dtype=np.int32
                ),
                "tokenized_prompt_mask": gym.spaces.Box(
                    0, 1, shape = (48,), dtype=np.bool_
                ),
            }
        )

    
def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


class RoboTwinEnv():
    def __init__(self, norm_stats = None):
        self.task_name = ["stack_blocks_two", "stack_blocks_three", "stack_bowls_two", "stack_bowls_three",]
        task_config = "demo_randomized"
        config_path = f"./task_config/{task_config}.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.args['task_config'] = task_config
        
        embodiment_type = self.args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)


        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise "missing embodiment files"
            return robot_file
        
        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args
        

        if len(embodiment_type) == 1:
            self.args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            self.args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            self.args["embodiment_dis"] = embodiment_type[2]
            self.args["dual_arm_embodied"] = False
        else:
            raise "number of embodiment config parameters should be 1 or 3"

        self.args["left_embodiment_config"] = get_embodiment_config(self.args["left_robot_file"])
        self.args["right_embodiment_config"] = get_embodiment_config(self.args["right_robot_file"])

        if norm_stats is not None:
            self.input_transform = transforms.Normalize(norm_stats, use_quantiles=False)
            self.output_transform = transforms.Unnormalize(norm_stats, use_quantiles=False)
        else:
            raise ValueError("norm_stats must be provided for transforms")
        
        self.delta_action_mask = transforms.make_bool_mask(6, -1, 6, -1)
        self.tokenizer = tokenizer.PaligemmaTokenizer(48)
        

    def reset(self, task_name = None, mode = None, now_seed = random.randint(2000, 10000), max_seed = 100000):
        if task_name is None:
            task_name = np.random.choice(self.task_name)
        task = class_decorator(task_name)
        self.args['task_name'] = task_name
        self.args["save_demo"] = False
        self.args["save_freq"] = 15
        while True:
            try:
                task.setup_demo(now_ep_num=0, seed=now_seed, is_test=False, **self.args)
                episode_info = task.play_once()
                task.close_env()
            except UnStableError as e:
                task.close_env()
                now_seed += 1
                continue
            except Exception as e:
                task.close_env()
                now_seed += 1
                continue
            if not task.plan_success or not task.check_success():
                now_seed += 1
                continue
            else:
                self.task = task
                if mode == "demo":
                    self.args["save_demo"] = True
                    self.args["save_freq"] = 2
                elif mode is not None:
                    raise ValueError("mode must be 'demo' or NOT set")
                self.task.setup_demo(now_ep_num=0, seed=now_seed, is_test=False, **self.args)
                episode_info_list = [episode_info["info"]]
                results = generate_episode_descriptions(self.args["task_name"], episode_info_list, max_seed)
                instruction = np.random.choice(results[0]["unseen"])
                self.task.set_instruction(instruction)
                self.instruction = instruction
                observation = self.input(self.get_observation())

                return observation, task_name, now_seed
        
    def step(self, actions):
        actions = transforms.pad_to_dim(actions, 32)
        observation = self.input(self.get_observation())
        state = observation["state"]
        output = self.output_transform({
            "state": state,
            "actions": actions
        })
        state, actions = output["state"], output["actions"]
        actions = np.asarray(actions[:, :14])
        actions[..., :14] += np.expand_dims(np.where(self.delta_action_mask, state[..., :14], 0), axis=-2)

        for action in actions:
            self.task.take_action(action)

        next_observation = self.input(self.get_observation())
        done = True if self.task.eval_success or self.task.take_action_cnt >= self.task.step_lim else False
        reward = 1.0 if self.task.eval_success else 0.0
        if done:
            self.task.close_env()
        return next_observation, reward, done, {"success": reward}

    def get_observation(self):
        def encode_obs(obs):
            input_rgb_arr = [
                obs["observation"]["head_camera"]["rgb"],
                obs["observation"]["right_camera"]["rgb"],
                obs["observation"]["left_camera"]["rgb"],
            ]
            input_state = obs["joint_action"]["vector"]
            return input_rgb_arr, input_state
        

        obs = self.task.get_obs()
        input_rgb_arr, input_state = encode_obs(obs)
        tokens, token_masks = self.tokenizer.tokenize(self.instruction)

        observation = {
            "state": input_state,
            "image": {
                "base_0_rgb": input_rgb_arr[0],
                "left_wrist_0_rgb": input_rgb_arr[2],
                "right_wrist_0_rgb": input_rgb_arr[1],
            },
            "image_mask": {
                "base_0_rgb":True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": True,
            },
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
        }
        return observation


    def input(self, observation):
        observation["state"] = transforms.pad_to_dim(observation["state"], 32)
        if "actions" in observation:
            observation["actions"] = transforms.pad_to_dim(observation["actions"], 32)
        observation = self.input_transform(observation)
        observation["state"] = np.asarray(observation["state"][:14])  # only keep the first 14 dims

        observation["image"] = {k: image_tools.resize_with_pad(v, 224, 224) for k, v in observation["image"].items()}
        return observation
    
    

    
    

    

    

    




