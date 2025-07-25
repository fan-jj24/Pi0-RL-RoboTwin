#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import glob
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
import copy
import pickle as pkl
from natsort import natsorted
from datetime import datetime
import subprocess
import yaml
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(root_path))

from agentlace.trainer import TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import make_trainer_config, make_act_agent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.agents.ActAgent import ActorAgent
from pi0.src.openpi.training.rl_cfg import rl_config, RoboTwinEnv
from pi0.src.openpi.shared import normalize

config = rl_config()
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))
def print_red(x):
    return print("\033[91m {}\033[00m".format(x))
def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))
def print_blue(x):
    return print("\033[94m {}\033[00m".format(x))


##############################################################################


def actor(agent, data_store, intvn_data_store, sampling_rng):
    output_file = open(str(root_path / "eval_results" / "policy_results.log"), "a")
    save_dir = root_path / "eval_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    camera_config_path = str(root_path / "task_config" / "_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)["D435"]
    video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
    eval_count = 0
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            f"{save_dir}/episode{eval_count}.mp4",
        ],
        stdin=subprocess.PIPE,
    )
    
    norm_stats_dir = str(root_path)
    norm_stats = normalize.load(norm_stats_dir)
    env = RoboTwinEnv(norm_stats=norm_stats)


    buffer_dir = os.path.join(FLAGS.checkpoint_path, "buffer")
    buffer_files = natsorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))
    if FLAGS.checkpoint_path and os.path.exists(buffer_dir) and buffer_files:
        start_step = int(os.path.basename(buffer_files[-1])[12:-4]) + 1
    else:
        start_step = 0

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        actor_params = {"modules_actor": params}
        agent = agent.replace(state=agent.state.replace(params=actor_params))

    client.recv_network_callback(update_params)

    transitions = []
    # demo_transitions = []
    
    obs, instruction, now_seed = env.reset(save_video=True)
    env.task.eval_video_path = save_dir
    env.task._set_eval_video_ffmpeg(ffmpeg)
    # training loop
    timer = Timer()

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                sample_rng = sampling_rng,
                observations=jax.device_put(obs),
                seed=key,
                argmax=False,
            )
            actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, info = env.step(actions)

            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                dones=done,
            )
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            '''if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))'''

            obs = next_obs
            if done:
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                client.update()
                result_msg = f"reward: {reward}, task_name: {instruction}, seed: {now_seed}"
                output_file.write(f"{datetime.now()}: {result_msg}\n")  
                output_file.flush() 
                env.task._del_eval_video_ffmpeg()
                env.close_env(env.task)
                eval_count += 1
                obs, instruction, now_seed = env.reset(save_video=True)
                env.task.eval_video_path = save_dir
                env.task._set_eval_video_ffmpeg(ffmpeg)


        if step > 0 and config.buffer_period and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            #demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            # if not os.path.exists(demo_buffer_path):
                # os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            # with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                # pkl.dump(demo_transitions, f)
                # demo_transitions = []

        timer.tock("total")
        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################

def main(_):
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)    

    sample_obs=config.observation_space.sample()
    sample_action=config.action_space.sample()
    sample_obs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_obs)
    sample_action = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_action)

    agent: ActorAgent = make_act_agent(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        target_policy_noise=config.target_policy_noise,
        noise_clip=config.noise_clip,
        pretrained_policy_path=str(root_path / "pretrained_params" / "params_30000"),
    )

    data_store = QueuedDataStore(20000)  # the queue size on the actor
    intvn_data_store = QueuedDataStore(20000)
    # actor loop
    print_green("starting actor loop")
    actor(
        agent,
        data_store,
        intvn_data_store,
        sampling_rng,
    )



if __name__ == "__main__":
    app.run(main)
