import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def t(v, dtype=None, device=device, requires_grad=False):
    """Shortcut for tensor instantiation with device."""
    return torch.tensor(v, device=device, dtype=dtype, requires_grad=requires_grad)


def mc_values(rewards, hyper_ps):
    """
    Gives a list of MC estimates for a given list of samples from an RL environment.
    The MC estimator is used for this computation.

    :param rewards: The rewards that were obtained while exploring the RL environment.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The MC estimates.
    """

    mcs = np.zeros(shape=(len(rewards),))
    discount_factor = dict_with_default(hyper_ps, 'discount_factor', .9)

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
    Gives a list of TD estimates for a given list of samples from an RL environment.
    The TD(λ) estimator is used for this computation.

    :param replay_buffers: The replay buffers filled by exploring the RL environment.
    Includes: states, rewards, "final state?"s.
    :param state_values: The currently estimated state values.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The TD estimates.
    """

    states, rewards, dones = replay_buffers
    sample_count = len(states)
    tds = np.zeros(shape=(sample_count,))

    discount_factor = dict_with_default(hyper_ps, 'discount_factor', .9)
    alpha = dict_with_default(hyper_ps, 'alpha', .95)
    lam = dict_with_default(hyper_ps, 'lambda', .95)

    val = 0.
    for i in range(sample_count - 1, -1, -1):
        state_value = state_values[i]
        next_value = 0. if dones[i] else state_values[i + 1]

        error = rewards[i] + discount_factor * next_value - state_value
        val = alpha * error + discount_factor * lam * (1 - dones[i]) * val

        tds[i] = val + state_value

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


def obs_to_state(observation):
    """
    Converts a given observation into a state tensor.
    Necessary as a catch-all for MuJoCo environments.

    :param observation: The observation received from the environment.

    :return: The state tensor.
    """

    if type(observation) is dict:
        state = state_from_mujoco(observation)
    else:
        state = observation

    return t(state).float()


def state_from_mujoco(observation):
    """
    Converts the observation parts returned by a MuJoCo environment into a single vector of values.

    :param observation: The observation containing the relevant parts.

    :return: A single vector containing all the observation information.
    """

    ag = observation['achieved_goal']
    dg = observation['desired_goal']
    obs = observation['observation']

    return np.concatenate([ag, dg, obs])
