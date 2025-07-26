#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import pickle as pkl
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(root_path))

from agentlace.trainer import TrainerServer

from serl_launcher.utils.launcher import make_rl_agent_hybrid_dual_arm, make_trainer_config, make_wandb_logger
from serl_launcher.data.data_store import DynamicNextObsReplayBufferDataStore
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.agents.RLAgent_dual import RLAgent
from pi0.src.openpi.training.rl_cfg import rl_config

config = rl_config()
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
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

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)) if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) else None

    if latest_ckpt is not None:
        start_step = int(os.path.basename(latest_ckpt)[11:]) + 1
        print(f"Resuming from checkpoint at step {start_step}.")
    else:
        start_step = 0
        print("No checkpoint found. Starting from step 0.")

    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_red("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    train_critic_networks_to_update = frozenset({"critic"})
    train_actor_networks_to_update = frozenset({"actor"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):  
        # run n critic updates and 1 actor update
        # the actor update is much more expensive, we do it less frequently and dont combine it with critic updates
        with timer.context("train_critics"):
            for critic_step in range(config.cta_ratio):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )
                if critic_step % config.log_period == 0 and wandb_logger:
                    wandb_logger.log(critics_info, step = critic_step * (step + 1))

        with timer.context("train_actor"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_actor_networks_to_update,
            )

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
    
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params["modules_actor"])
            print_red(f"published network at step {step}")

        if step > 0 and step % config.checkpoint_period == 0:
            checkpoints.save_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=5)


##############################################################################

def main(_):
    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)    

    sample_obs=config.observation_space.sample()
    sample_action=config.action_space.sample()
    sample_obs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_obs)
    sample_action = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_action)

    agent: RLAgent = make_rl_agent_hybrid_dual_arm(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        discount=config.discount,
        soft_target_update_rate=config.soft_target_update_rate,
        target_policy_noise=config.target_policy_noise,
        noise_clip=config.noise_clip,
        image_keys=config.image_keys,
        augmentation_function=config.augmentation_function,
        pretrained_policy_path=str(root_path / "pretrained_params" / "params_30000"),
        reward_bias=config.reward_bias,
    )
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        print("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            print(f"Loaded previous checkpoint at step {ckpt_number}.")
        else:
            print("No checkpoint found. Starting from scratch.")
            ckpt_number = 0

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = DynamicNextObsReplayBufferDataStore(
            config.observation_space,
            config.action_space,
            capacity=config.replay_buffer_capacity,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="pi0-serl",
            description="serl pi0 in robotwin",
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger


    sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
    replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
    demo_buffer = DynamicNextObsReplayBufferDataStore(
        config.observation_space,
        config.action_space,
        capacity=config.replay_buffer_capacity,
    )

    assert FLAGS.demo_path is not None
    for path in FLAGS.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            for transition in transitions:
                demo_buffer.insert(transition)
    print_green(f"demo buffer size: {len(demo_buffer)}")
    print_green(f"online buffer size: {len(replay_buffer)}")

    if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
        for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    replay_buffer.insert(transition)
        print_green(f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

    if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
        for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    demo_buffer.insert(transition)
        print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

    # learner loop
    print_green("starting learner loop")
    learner(
        sampling_rng,
        agent,
        replay_buffer,
        demo_buffer=demo_buffer,
        wandb_logger=wandb_logger,
    )

    


if __name__ == "__main__":
    app.run(main)
