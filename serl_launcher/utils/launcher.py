# !/usr/bin/env python3
import jax
import jax.numpy as jnp
from typing import Iterable, Optional
from agentlace.trainer import TrainerConfig
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.RLAgent_dual import RLAgent
from serl_launcher.vision.data_augmentations import batched_random_crop

##############################################################################

def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000


    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations
    
    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch
    
    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl-for-your-policy",
    description: str = "serl_launcher",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger


def make_rl_agent_hybrid_dual_arm(
    seed,
    sample_obs,
    sample_action,
    discount=0.95,
    soft_target_update_rate: float = 0.005,
    target_policy_noise: float = 0.1,
    noise_clip: float = 0.1,
    image_keys: Iterable[str] = ("image",),
    augmentation_function: Optional[callable] = None,
    pretrained_policy_path: Optional[str] = None,
    reward_bias: float = 0.0,

):
    agent = RLAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        critic_ensemble_size=2,
        discount=discount,
        soft_target_update_rate=soft_target_update_rate,
        target_policy_noise=target_policy_noise,
        noise_clip=noise_clip,
        image_keys=image_keys,
        # augmentation_function=make_batch_augmentation_func(image_keys),
        augmentation_function=augmentation_function,
        pretrained_policy_path=pretrained_policy_path,
        reward_bias=reward_bias,
    )
    return agent
