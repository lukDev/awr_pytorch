import random
import torch

from sample import Sample
from utilities.debug import DebugType
from utilities.utils import t, device, td_trajectories, critic_inputs, mc_trajectories, nan_in_model, dict_with_default


class AWRAgent:

    name = "awr"

    @staticmethod
    def train(models, environment, hyper_ps, debug_type, writer):
        assert len(models) == 2, "AWR needs exactly two models to function properly."
        actor, critic = models

        # replay buffer
        replay_buffer = []
        max_buffer_size = hyper_ps['replay_size']
        replay_fill_threshold = dict_with_default(hyper_ps, 'replay_fill_threshold', 0.)
        random_exploration = dict_with_default(hyper_ps, 'random_exploration', False)

        # learning time setup
        max_epoch_count = hyper_ps['max_epochs']
        epoch = 0
        pre_training_epochs = 0
        max_pre_epochs = 150

        # algorithm specifics
        beta = hyper_ps['beta']
        critic_steps_start = hyper_ps['critic_steps_start']
        critic_steps_end = hyper_ps['critic_steps_end']
        actor_steps_start = hyper_ps['actor_steps_start']
        actor_steps_end = hyper_ps['actor_steps_end']
        batch_size = hyper_ps['batch_size']
        max_advantage_weight = hyper_ps['max_advantage_weight']
        min_log_pi = hyper_ps['min_log_pi']

        # debug helper field
        debug_full = debug_type == DebugType.FULL

        # critic pre-training
        critic_threshold = hyper_ps['critic_threshold']
        critic_suffices_required = hyper_ps['critic_suffices_required']
        critic_suffices_count = 0
        critic_suffices = False

        # return normalization
        return_norm = dict_with_default(hyper_ps, 'return_norm', 1.)

        # evaluation
        validation_epoch_mod = dict_with_default(hyper_ps, 'validation_epoch_mod', 30)
        test_iterations = hyper_ps['test_iterations']

        AWRAgent.compute_validation_return(
            actor,
            environment,
            hyper_ps,
            debug_type,
            test_iterations,
            epoch,
            writer
        )

        while epoch < max_epoch_count + pre_training_epochs:
            print(f"\nEpoch: {epoch}")

            # set actor and critic update steps
            epoch_percentage = ((epoch - pre_training_epochs) / max_epoch_count)
            critic_steps = critic_steps_start + int((critic_steps_end - critic_steps_start) * epoch_percentage)
            actor_steps = actor_steps_start + int((actor_steps_end - actor_steps_start) * epoch_percentage)

            # sampling from env
            samples = AWRAgent.sample_from_env(
                actor,
                environment,
                debug_full,
                exploration=random_exploration and len(replay_buffer) < replay_fill_threshold * max_buffer_size
            )
            if len(replay_buffer) + len(samples) > max_buffer_size:
                if len(samples) <= max_buffer_size:
                    replay_buffer = replay_buffer[len(replay_buffer) + len(
                        samples) - max_buffer_size:]  # delete just enough from the replay buffer to fit the new data in
                else:
                    replay_buffer = []
            replay_buffer.extend(samples)

            if len(replay_buffer) >= replay_fill_threshold * max_buffer_size:
                # training the critic
                avg_loss = 0.
                for _ in range(critic_steps):
                    samples = random.choices(replay_buffer, k=batch_size)
                    trajectories = td_trajectories(samples, critic, hyper_ps)
                    ins = critic_inputs(trajectories).to(device)
                    tars = torch.cat([tr.reward_as_tensor().unsqueeze(0) for tr in trajectories]) / return_norm

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
                for temp in range(actor_steps):
                    samples = random.choices(replay_buffer, k=batch_size)
                    trajectories = td_trajectories(samples, critic, hyper_ps)

                    rets = t([tr.reward / return_norm for tr in trajectories], device=device)

                    ins = critic_inputs(trajectories)
                    state_values = critic.evaluate(ins).clone().detach().to(device).squeeze(1)

                    advantage_weights = torch.exp((1. / beta) * (rets - state_values)).requires_grad_(False)
                    # positive_advs = [ad > 40. for ad in advantage_weights.tolist()]
                    # if any(positive_advs):
                    #     print("success")
                    advantage_weights = torch.clamp(advantage_weights, max=max_advantage_weight)

                    actor_ins = torch.cat([tr.state.unsqueeze(0) for tr in trajectories]).to(device)
                    normal, _ = actor(actor_ins)
                    actions = torch.cat([tr.action.unsqueeze(0) for tr in trajectories]).to(device)
                    log_pis = normal.log_prob(actions)
                    log_pis = torch.sum(log_pis, dim=1)
                    log_pis = torch.clamp(log_pis, min=min_log_pi)

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

            if critic_suffices and epoch % validation_epoch_mod == 0:
                AWRAgent.compute_validation_return(
                    actor,
                    environment,
                    hyper_ps,
                    debug_type,
                    test_iterations,
                    epoch,
                    writer
                )

        return actor, critic

    @staticmethod
    def compute_validation_return(actor, env, hyper_ps, debug_type, iterations, epoch, writer):
        print("computing average return")
        sample_return = AWRAgent.validation_return(actor, env, hyper_ps, debug_type, iterations)
        writer.add_scalar('return', sample_return, epoch)
        print(f"return: {sample_return}")

    @staticmethod
    def validation_return(actor, env, hyper_ps, debug_type, iterations):
        sample_return = 0.
        for _ in range(iterations):
            samples = AWRAgent.sample_from_env(actor, env, debug_type != DebugType.NONE, exploration=False)
            mc_trajs = mc_trajectories(samples, hyper_ps)
            sample_return += torch.mean(t([tr.reward for tr in mc_trajs]))

        sample_return /= iterations
        return sample_return

    @staticmethod
    def sample_from_env(actor_model, env, debug, exploration):
        samples = []
        state = t(env.reset()).float()
        done = False

        if debug:
            env.render()

        while not done:
            if exploration:
                action = t(env.action_space.sample())
            else:
                _, action = actor_model.evaluate(state)
            res = env.step(action.cpu().numpy())

            reward = res[1]
            old_state = state.clone().detach()
            state = t(res[0]).float()
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
    def test(models, environment, hyper_ps, debug_type):
        actor, _ = models
        return AWRAgent.validation_return(actor, environment, hyper_ps, debug_type is not DebugType.NONE, 1)
