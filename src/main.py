import gym

from agent import AWRAgent
from models.actor import Actor
from models.critic import Critic
from utilities.debug import DebugType
from utilities.training import Training


# setting the hyper-parameters
hyper_ps = {
    'replay_size': 50000,
    'max_epochs': 150,
    'sample_mod': 10,
    'beta': .05,
    'max_advantage_weight': 50.,
    'min_log_pi': -50.,
    'discount_factor': .9,
    'alpha': 0.95,
    'c_hidden_size': 64,
    'c_hidden_layers': 1,
    'a_hidden_size': 64,
    'a_hidden_layers': 2,
    'c_learning_rate': 1e-4,
    'c_momentum': .9,
    'a_learning_rate': 5e-5,
    'a_momentum': .9,
    'critic_threshold': 3.5,
    'critic_suffices_required': 1,
    'critic_steps_start': 200,
    'critic_steps_end': 200,
    'actor_steps_start': 1000,
    'actor_steps_end': 1000,
    'batch_size': 256,
    'seed': 123456,
    'replay_fill_threshold': 1.,
    'random_exploration': True,
    'return_norm': 2.,
    'test_iterations': 15,
    'validation_epoch_mod': 3,
}

# configuring the environment
environment = gym.make('Pendulum-v0')
# environment._max_episode_steps = 600

# setting up the training components
agent = AWRAgent
actor = Actor()
critic = Critic()

# training and testing
Training.train(
    (actor, critic),
    agent,
    environment,
    hyper_ps,
    save=True,
    debug_type=DebugType.NONE
)
