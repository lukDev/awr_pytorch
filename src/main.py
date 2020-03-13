import gym

from agent import AWRAgent
from models.actor import Actor
from models.critic import Critic
from utilities.debug import DebugType
from utilities.training import Training


# setting the hyper-parameters
hyper_ps = {
    'replay_size': 20000,
    'max_epochs': 1000,
    'beta': 0.05,
    'max_advantage_weight': 50.,
    'discount_factor': .9,
    'c_hidden_size': 128,
    'c_hidden_layers': 1,
    'a_hidden_size': 128,
    'a_hidden_layers': 2,
    'c_learning_rate': 1e-4,
    'c_momentum': .9,
    'a_learning_rate': 5e-5,
    'a_momentum': .9,
    'critic_threshold': 1.,
    'critic_suffices_required': 5,
    'critic_steps': 200,
    'actor_steps': 500,
    'batch_size': 256,
    'seed': 123456,
    'lambda': .95,
}

# configuring the environment
environment = gym.make('Pendulum-v0')
environment._max_episode_steps = 600

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
