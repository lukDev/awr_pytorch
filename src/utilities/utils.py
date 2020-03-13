import torch

from sample import Sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def t(v, dtype=None, device=device, requires_grad=False):
    """Shortcut for tensor instantiation with device."""
    return torch.tensor(v, device=device, dtype=dtype, requires_grad=requires_grad)


def mc_trajectories(samples, hyper_ps):
    """
    Gives a list of trajectories for a given list of samples from an RL environment.
    The MC estimator is used for this computation.
    All non-final rewards are discounted according to the strategy specified in the given hyper-parameters.
    A sample consists of state, noise/action distr., action, reward, remaining time.
    A trajectory consists of state, noise/action distr., action, return, remaining time.

    :param samples: The samples to be used to compute their corresponding trajectories.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The trajectories.
    """

    trajectories = []

    discount_factor = hyper_ps['discount_factor']
    for i, sample in enumerate(samples):
        ret = 0.
        gamma = 1.

        for j in range(i + 1, len(samples)):
            gamma *= discount_factor
            ret += gamma * samples[j].reward

        new_sample = Sample.from_sample(sample)
        new_sample.reward = ret
        trajectories.append(new_sample)

    return trajectories


def td_trajectories(samples, critic, hyper_ps):
    """
    Gives a list of trajectories for a given list of samples from an RL environment.
    The TD(λ) estimator is used for this computation. λ is given in the hyper-parameters.
    All non-final rewards are discounted according to the strategy specified in the given hyper-parameters.
    A sample consists of state, noise/action distr., action, reward, remaining time.
    A trajectory consists of state, noise/action distr., action, return, remaining time.

    :param samples: The samples to be used to compute their corresponding trajectories.
    :param epoch: The epoch the agent is currently in.
    :param critic: The state-value estimator.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The trajectories.
    """

    trajectories = []
    mc_trajs = mc_trajectories(samples, hyper_ps)

    next_state_values = [critic.evaluate(critic_inputs([s])) for s in samples]

    max_t = len(samples)
    lam = 0.95 if 'lambda' not in hyper_ps else hyper_ps['lambda']
    discount_factor = hyper_ps['discount_factor']

    for t, sample in enumerate(samples):
        ret = 0.

        running_sum = 0.
        gamma = 1.
        for n in range(1, max_t - t):
            gamma *= discount_factor
            running_sum += gamma * samples[t+n].reward
            ret += lam**(n-1) * (running_sum + next_state_values[t+n])

        ret *= 1. - lam
        ret += lam**(max_t - t - 1) * mc_trajs[t].reward

        trajectories.append(Sample.from_sample(sample).with_reward_(ret))

    return trajectories


def critic_inputs(trajectories):
    """
    Extracts the relevant inputs for the V-critic from the given list of trajectories.

    :param trajectories: The trajectories from which the information should be taken.
    :return: The extracted information in the form of a batched tensor.
    """

    return torch.cat([tr.state.flatten().unsqueeze(0) for tr in trajectories]).to(device)


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
