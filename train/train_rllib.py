import sys
import os

# Fix path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from env.climate_env import ClimateParallelEnv


# ---------------- ENV ----------------
def env_creator(config):
    return ParallelPettingZooEnv(ClimateParallelEnv(n_agents=3))


register_env("climate_env", env_creator)

ray.init(ignore_reinit_error=True)


# ---------------- MULTI-AGENT SETUP ----------------
temp_env = ClimateParallelEnv(n_agents=3)

policies = {
    agent: (
        None,
        temp_env.observation_space(agent),
        temp_env.action_space(agent),
        {}
    )
    for agent in temp_env.agents
}


def policy_mapping_fn(agent_id, *args, **kwargs):
    return agent_id


# ---------------- PPO CONFIG ----------------
config = (
    PPOConfig()
    .environment("climate_env")
    .framework("torch")

    .env_runners(num_env_runners=1)

    .training(
        train_batch_size=4000,
        lr=2e-4,
        gamma=0.99,
        clip_param=0.2,
        entropy_coeff=0.02,
        vf_loss_coeff=1.0,
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        }
    )

    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=list(policies.keys())  # ✅ IMPORTANT
    )

    .clip_actions(True)
    .normalize_actions(True)

    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)


# ---------------- TRAIN ----------------
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 150},
    verbose=1,

    # 🔥 SAVE CHECKPOINTS (CRITICAL)
    checkpoint_freq=10,
    checkpoint_at_end=True
)