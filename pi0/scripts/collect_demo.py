from openpi.training.rl_cfg import RoboTwinEnv
from copy import deepcopy
import pickle as pkl
import datetime
import os
import numpy as np

env = RoboTwinEnv()
TASK = ["stack_blocks_two", "stack_blocks_three", "stack_bowls_two", "stack_bowls_three",]

def main():
    transitions, demo_count = [], 10
    print("\033[31mBegin demo collect\033[0m")

    for val in TASK:
        print(f"\033[32mTask: {val}\033[0m")
        start_seed = 999
        for i in range(demo_count):
            _, _, start_seed = env.reset(task_name = val, mode = "demo",now_seed = start_seed + 1, max_seed = 2000)
            env.task.play_once()
            demo = env.task.actions_traj
            for index, (obs, actions) in enumerate(demo):
                actions[..., :14] -= np.expand_dims(np.where(env.delta_action_mask, obs["state"][..., :14], 0), axis=-2)
                obs["actions"] = actions
                obs = env.input(obs)
                actions = obs["actions"]
                del obs["actions"]
                if index == len(demo) - 1:
                    done, rew = True, 1.0
                    next_obs = obs
                    while len(actions) < 50:
                        actions.append(actions[-1])
                else:
                    done, rew = False, 0.0
                    next_obs, _ = demo[index + 1]
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
            print(f"\033[33mTask {val} {i} success.\033[0m")

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{demo_count}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {demo_count} demos to {file_name}")
                    

if __name__ == "__main__":
    main()

        