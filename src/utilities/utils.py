import copy

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def t(v, dtype=None, device=device, requires_grad=False):
    """Shortcut for tensor instantiation with device."""
    return torch.tensor(v, device=device, dtype=dtype, requires_grad=requires_grad)


def mc_values(rewards, hyper_ps):
    """
    Gives a list of trajectories for a given list of samples from an RL environment.
    The MC estimator is used for this computation.
    All non-final rewards are discounted according to the strategy specified in the given hyper-parameters.
    A sample consists of state, action distr., action, REWARD, remaining time.
    A trajectory consists of state, action distr., action, RETURN, remaining time.

    :param samples: The samples to be used to compute their corresponding trajectories.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The trajectories.
    """

    mcs = np.zeros(shape=(len(rewards),))
    discount_factor = hyper_ps['discount_factor']

    for i, reward in enumerate(rewards):
        ret = reward
        gamma = 1.

        for j in range(i + 1, len(rewards)):
            gamma *= discount_factor
            ret += gamma * rewards[j]

        mcs[i] = ret

    return mcs


def td_values(replay_buffers, state_values, hyper_ps):
    """
    Gives a list of trajectories for a given list of samples from an RL environment.
    The TD(0) estimator is used for this computation.
    A sample consists of state, action distribution, action, reward, remaining time.
    A trajectory consists of state, action distribution, action, return, remaining time.

    :param samples: The samples to be used to compute their corresponding trajectories.
    :param epoch: The epoch the agent is currently in.
    :param critic: The state-value estimator.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The trajectories.
    """

    states, rewards, dones = replay_buffers
    tds = np.zeros(shape=(len(states),))

    discount_factor = dict_with_default(hyper_ps, 'discount_factor', .9)
    alpha = dict_with_default(hyper_ps, 'alpha', .95)

    for i, state in enumerate(states):
        state_value = state_values[i]
        next_value = 0. if dones[i] else state_values[i + 1]
        ret = state_value + alpha * (rewards[i] + discount_factor * next_value - state_value)

        tds[i] = ret

    return tds


def critic_inputs(trajectories, next_states=False):
    """
    Extracts the relevant inputs for the V-critic from the given list of trajectories.

    :param trajectories: The trajectories from which the information should be taken.
    :param next_states: Extract the next-state entries from the samples instead of the current states.

    :return: The extracted information in the form of a batched tensor.
    """

    return torch.cat([(tr.next_state if next_states else tr.state).flatten().unsqueeze(0) for tr in trajectories]).to(device)


def nan_in_model(model):
    """
    Checks if the given model holds any parameters that contain NaN values.

    :param model: The model to be checked for NaN entries.

    :return: Whether the model contain NaN parameters.
    """

    for p in model.parameters():
        p_nan = torch.isnan(p.data).flatten().tolist()
        if any(p_nan):
            return True

    return False


def dict_with_default(dict, key, default):
    """
    Returns the value contained in the given dictionary for the given key, if it exists.
    Otherwise, returns the given default value.

    :param dict: The dictionary from which the value should be read.
    :param key: The key to look for in the dictionary.
    :param default: The fallback value, in case the dictionary doesn't contain the desired key.

    :return: The value read from the dictionary, if it exists. The default value otherwise.
    """

    if key in dict:
        return dict[key]
    else:
        return default


def xavier_init(m):
    """
    Xavier normal initialisation for layer m.

    :param m: The layer to have its weight and bias initialised.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def kaiming_init(m):
    """
    Kaiming normal initialisation for layer m.

    :param m: The layer to have its weight and bias initialised.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
