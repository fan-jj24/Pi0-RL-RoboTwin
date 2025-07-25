import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

from copy import deepcopy
import pickle as pkl
import datetime
import numpy as np
import sys
sys.path.insert(0, str(root_path))

from pi0.src.openpi.training.rl_cfg import RoboTwinEnv
from pi0.src.openpi.shared import normalize
norm_stats_dir = str(root_path)
norm_stats = normalize.load(norm_stats_dir)

env = RoboTwinEnv(norm_stats = norm_stats)
TASK = ["stack_blocks_two", "stack_blocks_three", "stack_bowls_two", "stack_bowls_three",]
seed2 = [1086, 1174, 1242, 1382, 1659, 1921, 1943, 2044, 2065, 2210, 1039, 1052, 1053, 1084, 1143, 1169, 1174, 1176, 1197, 1198, 1002, 1007, 1008, 1009, 1010, 1018, 1023, 1025, 1031, 1032, 1002, 1003, 1005, 1006, 1008, 1010, 1014, 1018, 1023, 1025]
seed2 = {
    "stack_blocks_two": seed2[:10],
    "stack_blocks_three": seed2[10:20],
    "stack_bowls_two": seed2[20:30],    
    "stack_bowls_three": seed2[30:40],
}

def main():
    transitions, success_count, fail_count = [], 10, 5
    success_seed, fail_seed = [], []
    print("\033[31mBegin demo collect\033[0m")

    for val in TASK:
        print(f"\033[32mTask: {val}\033[0m")
        start_seed, i, j = 2000, 0, 0
        while i < success_count or j < fail_count:
            if i< success_count:
                _, _, start_seed = env.reset(task_name = val, mode = "demo",now_seed = start_seed + 1, max_seed = 2000)
            else:
                env.reset(task_name = val, mode = "demo",now_seed = seed2[val][j + 5], max_seed = 2000)
            env.task.play_once()
            env.close_env(env.task)
            demo = env.task.demo_traj
            if env.mode_flag == "success" and i >= success_count:
                print(f"\033[33mTask {val} {seed2[val][j + 5]} success, skip.\033[0m")
                j += 1
                continue
            if env.mode_flag == "fail" and j >= fail_count:
                print(f"\033[33mTask {val} {start_seed} fail, skip.\033[0m")
                continue

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
                        i += 1
                        done, rew = True, 1.0
                    else:
                        if i < success_count:
                            fail_seed.append(start_seed)
                        else:
                            fail_seed.append(seed2[val][j + 5])
                        j += 1
                        done, rew = True, 0.0

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
    print("fail_seed: ", fail_seed)        
    if not os.path.exists("/home/anker/robotwin/Pi0-RL-RoboTwin/demo_data"):
        os.makedirs("/home/anker/robotwin/Pi0-RL-RoboTwin/demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"/home/anker/robotwin/Pi0-RL-RoboTwin/demo_data/{success_count + fail_count}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_count + fail_count} demos to {file_name}")
                    

if __name__ == "__main__":
    main()

        