from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SimBaActorCriticCfg(RslRlPpoActorCriticCfg):
    """Config for the SimBa actor-critic backbone.
    """

    class_name: str = "SimBaActorCritic"

    # Unused by SimBa
    actor_hidden_dims: list[int] = []
    critic_hidden_dims: list[int] = []
    activation: str = "relu"

    # SimBa-specific
    actor_hidden_dim: int = 1024
    critic_hidden_dim: int = 1024
    actor_num_blocks: int = 2
    critic_num_blocks: int = 2
    expansion: int = 2


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 100
    experiment_name = "g1_inspire_hoi"
    policy = SimBaActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dim=1024,
        critic_hidden_dim=1024,
        actor_num_blocks=2,
        critic_num_blocks=2,
        expansion=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
