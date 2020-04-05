import torch
import json
import datetime
import os

from tensorboardX import SummaryWriter

from utilities.debug import DebugType
from utilities.utils import device, t, nan_in_model


class Training:

    @staticmethod
    def train(models, agent, environment, hyper_ps, save=True, debug_type=DebugType.NONE):
        """
        Executes a full training and testing cycle and stores the results.

        :param models: The models to train.
        :param agent: The agent to train the models.
        :param environment: The environment to be trained in.
        :param hyper_ps: All hyper-parameters required for the given models and the agent.
        :param save: Triggers saving of parameters, models and results.
        :param debug_type: Specifies the amount of debug information to be printed during training and testing.
        """

        # setting the random seed
        if 'seed' in hyper_ps:
            torch.manual_seed(hyper_ps['seed'])

        # creating the directory for this test
        # this assumes that your working directory is awr/src/
        dir_path = "../trained_models/"
        dir_path += f"[{environment.unwrapped.spec.id}]_[{agent.name}]_["
        for m in models:
            dir_path += f"({type(m).name})_"
        dir_path += "]_"
        now = datetime.datetime.now()
        dir_path += now.strftime("%d.%m.%Y,%H:%M:%S.%f")
        dir_path += "/"

        os.makedirs(dir_path)

        # creating the tensorboardX writer
        writer = SummaryWriter(dir_path + 'tensorboard/')

        if 'test_iterations' not in hyper_ps:
            hyper_ps['test_iterations'] = 100
        test_iterations = hyper_ps['test_iterations']

        # extend the hyper-parameters to include environment information
        hyper_ps['state_dim'] = environment.observation_space.shape[0]
        hyper_ps['action_dim'] = environment.action_space.shape[0]

        for m in models:
            m.set_params(hyper_ps)

        # converting the models to the current device
        models = [m.to(device) for m in models]

        # training and testing
        print("training")
        models = agent.train(models, environment, hyper_ps, debug_type, writer)

        nans = [nan_in_model(m) for m in models]
        if any(nans):
            print("NaN values in some models\nskipping testing")
            return

        print("testing")
        rewards = t([])
        for _ in range(test_iterations):  # expected reward estimated with average
            rews = agent.test(models, environment, debug_type)
            rews = rews.unsqueeze(0)
            rewards = torch.cat((rewards, rews))
        rewards = rewards.mean(0)

        if not save:
            print(f"Average reward: {rewards}")
        else:
            # saving the hyper-parameters
            parameter_text = json.dumps(hyper_ps)
            parameter_file = open(dir_path + "hyper-parameters.json", "w")
            parameter_file.write(parameter_text)
            parameter_file.close()

            # saving the models
            for model in models:
                params = model.state_dict()
                file_path = type(model).name + ".model"

                torch.save(params, dir_path + file_path)

            # saving the results
            result_text = "Results\n===\n"
            result_text += f"Testing was conducted {test_iterations} times to obtain an estimate of the expected return.\n\n\n"
            result_text += f"\nAverage return\n---\n{rewards}"

            results_file = open(dir_path + "results.md", "w")
            results_file.write(result_text)
            results_file.close()
