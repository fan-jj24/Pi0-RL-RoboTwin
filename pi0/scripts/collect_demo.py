import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

from copy import deepcopy
import pickle as pkl
import datetime
import numpy as np
import tqdm

import sys
sys.path.insert(0, str(root_path))

from pi0.src.openpi.training.rl_cfg import RoboTwinEnv
from pi0.src.openpi.shared import normalize
norm_stats_dir = str(root_path)
norm_stats = normalize.load(norm_stats_dir)

env = RoboTwinEnv(norm_stats = norm_stats)
TASK = ["stack_blocks_two", "stack_blocks_three", "stack_bowls_two", "stack_bowls_three",]

def main():
    transitions, success_count, success_seed = [], 30, []
    print("\033[31mBegin demo collect\033[0m")
    for val in TASK:
        print(f"\033[32mTask: {val}\033[0m")
        start_seed = 999
        
        pbar = tqdm.tqdm(range(success_count), dynamic_ncols=True, desc=f"Collecting {val}")
        for _ in pbar:
            _, _, start_seed = env.reset(task_name = val, mode = "demo",now_seed = start_seed + 1, max_seed = 2000, strict=True)
            env.task.play_once()
            env.close_env(env.task)
            demo = env.task.demo_traj
            print("length: ",len(demo))
            for index, (obs, actions) in enumerate(demo):
                actions[..., :14] -= np.expand_dims(np.where(env.delta_action_mask, obs["state"][..., :14], 0), axis=-2)
                obs["actions"] = actions
                obs = env.input(obs)
                actions = obs["actions"]
                del obs["actions"]
                obs["tokenized_prompt"], obs["tokenized_prompt_mask"] = env.tokens, env.token_masks
                if index == len(demo) - 1:
                    if env.mode_flag == "success":
                        success_seed.append(start_seed)
                        done, rew = True, 1.0
                    next_obs = obs
                    while len(actions) < 50:
                        actions.append(actions[-1])
                else:
                    done, rew = False, 0.0
                    next_obs, _ = demo[index + 1]
                    next_obs = env.input(deepcopy(next_obs))
                    next_obs["tokenized_prompt"], next_obs["tokenized_prompt_mask"] = env.tokens, env.token_masks

                transitions.append(
                    deepcopy(
                        dict(
                            observations=obs,
                            actions=actions,
                            next_observations=next_obs,
                            rewards=rew,
                            dones=done,
                        )
                    )
                )
            if env.mode_flag == "success":
                print(f"\033[33mTask {val} {start_seed} success.\033[0m")
            else:
                print(f"\033[33mTask {val} {start_seed} fail.\033[0m")

    print("success_seed: ", success_seed)
    demo_data_dir = root_path / "demo_data"
    demo_data_dir.mkdir(exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = demo_data_dir / f"{success_count}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_count} demos to {file_name}")


if __name__ == "__main__":
    main()
