import random
import torch

from sample import Sample
from utilities.debug import DebugType
from utilities.utils import t, device, td_trajectories, critic_inputs, mc_trajectories, nan_in_model


class AWRAgent:

    name = "awr"

    @staticmethod
    def train(models, environment, hyper_ps, debug_type, writer):
        assert len(models) == 2, "AWR needs exactly two models to function properly."
        actor, critic = models

        replay_buffer = []
        max_buffer_size = hyper_ps['replay_size']
        max_epoch_count = hyper_ps['max_epochs']
        beta = hyper_ps['beta']
        actor_steps = hyper_ps['actor_steps']
        critic_steps = hyper_ps['critic_steps']
        batch_size = hyper_ps['batch_size']
        max_advantage_weight = hyper_ps['max_advantage_weight']

        # debug helper field
        debug_full = debug_type == DebugType.FULL

        critic_threshold = hyper_ps['critic_threshold']
        critic_suffices_required = hyper_ps['critic_suffices_required']
        critic_suffices_count = 0
        critic_suffices = False

        return_ref_steps = 10
        return_norm_ref = 0.
        for _ in range(return_ref_steps):
            return_norm_ref += AWRAgent.highest_abs_return(actor, environment, hyper_ps, debug_type)
        return_norm_ref /= return_ref_steps
        print(f"return reference point: {return_norm_ref}")

        epoch = 0
        pre_training_epochs = 0
        max_pre_epochs = 150
        validation_epoch_mod = 20

        while epoch < max_epoch_count + pre_training_epochs:
            print(f"epoch: {epoch}")

            # sampling from env
            samples = AWRAgent.sample_from_env(
                actor,
                environment,
                debug_full,
                exploration=False  # len(replay_buffer) < 0.9 * max_buffer_size
            )
            if len(replay_buffer) + len(samples) > max_buffer_size:
                if len(samples) <= max_buffer_size:
                    replay_buffer = replay_buffer[len(replay_buffer) + len(
                        samples) - max_buffer_size:]  # delete just enough from the replay buffer to fit the new data in
                else:
                    replay_buffer = []
            td_trajs = td_trajectories(samples, critic, hyper_ps)
            replay_buffer.extend(td_trajs)

            if critic_suffices and epoch % validation_epoch_mod == 0:
                sample_return = td_trajs[0].reward
                for _ in range(validation_epoch_mod - 1):
                    samples = AWRAgent.sample_from_env(actor, environment, debug_full, exploration=False)
                    td_trajs = td_trajectories(samples, critic, hyper_ps)
                    sample_return += td_trajs[0].reward

                sample_return /= validation_epoch_mod
                writer.add_scalar('return', sample_return, epoch)
                print(f"return: {sample_return}")

            if len(replay_buffer) >= 0.:  # 0.9 * max_buffer_size
                # training the critic
                avg_loss = 0.
                for _ in range(critic_steps):
                    trajectories = random.choices(replay_buffer, k=batch_size)
                    ins = critic_inputs(trajectories).to(device)
                    tars = torch.cat([tr.reward_as_tensor().unsqueeze(0) for tr in trajectories]) / return_norm_ref

                    outs = critic(ins)
                    loss = critic.backward(outs, tars)
                    avg_loss += loss
                avg_loss /= critic_steps
                print(f"average critic loss: {avg_loss}")

                if nan_in_model(critic):
                    print("NaN values in critic\nstopping training")
                    break

                writer.add_scalar('critic_loss', avg_loss, epoch)

                if avg_loss <= critic_threshold:
                    critic_suffices_count += 1
                else:
                    critic_suffices_count = 0

                if critic_suffices_count >= critic_suffices_required:
                    critic_suffices = True
            else:
                continue

            if critic_suffices:
                # training the actor
                avg_loss = 0.
                for _ in range(actor_steps):
                    trajectories = random.choices(replay_buffer, k=batch_size)
                    rets = t([tr.reward / return_norm_ref for tr in trajectories], device=device)

                    ins = critic_inputs(trajectories)
                    state_values = critic.evaluate(ins).clone().detach().to(device).squeeze(1)

                    advantage_weights = torch.exp((1. / beta) * (rets - state_values)).requires_grad_(False)
                    advantage_weights = torch.clamp(advantage_weights, max=max_advantage_weight)

                    actor_ins = torch.cat([tr.state.unsqueeze(0) for tr in trajectories]).to(device)
                    normal, _ = actor(actor_ins)
                    actions = torch.cat([tr.action.unsqueeze(0) for tr in trajectories]).to(device)
                    log_pis = normal.log_prob(actions)
                    log_pis = torch.sum(log_pis, dim=1)

                    losses = -log_pis * advantage_weights
                    losses = losses / batch_size  # normalise wrt the batch size
                    actor.backward(losses)

                    mean_loss = torch.sum(losses)
                    avg_loss += mean_loss
                avg_loss /= len(replay_buffer)
                print(f"average actor loss: {avg_loss}")

                if nan_in_model(actor):
                    print("NaN values in actor\nstopping training")
                    break

                writer.add_scalar('actor_loss', avg_loss, epoch)
            else:
                pre_training_epochs += 1
                if pre_training_epochs > max_pre_epochs:
                    print("critic couldn't be trained in appropriate time\nstopping training")
                    break

            epoch += 1

        return actor, critic

    @staticmethod
    def sample_from_env(actor_model, env, debug, exploration):
        samples = []
        state = env.reset()
        done = False

        if debug:
            env.render()

        while not done:
            state = t(state.copy(), requires_grad=True, device=device).float()

            if exploration:
                action = t(env.action_space.sample())
            else:
                _, action = actor_model.evaluate(state)
            res = env.step(action.cpu().numpy())

            reward = res[1]
            old_state = state.clone().detach()
            state = res[0]
            samples.append(Sample(
                state=old_state,
                action=action,
                reward=reward,
                next_state=state,
            ))
            done = res[2]

            if debug:
                env.render()

        return samples

    @staticmethod
    def highest_abs_return(actor_model, environment, hyper_ps, debug_type):
        samples = AWRAgent.sample_from_env(actor_model, environment, debug_type == DebugType.FULL, exploration=True)
        trajectories = mc_trajectories(samples, hyper_ps)
        return max([abs(tr.reward) for tr in trajectories])

    @staticmethod
    def test(models, environment, debug_type):
        actor, _ = models
        samples = AWRAgent.sample_from_env(actor, environment, debug_type, exploration=False)
        return samples[-1].reward_as_tensor()
