class Sample:
    def __init__(self, state=None, action=None, reward=None, next_state=None, log_pi=None, remaining_time=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.log_pi = log_pi
        self.remaining_time = remaining_time

    def with_reward_(self, reward):
        self.reward = reward
        return self

    def rt_as_tensor(self):
        from utilities.utils import t, device
        return t([self.remaining_time], device=device)

    def reward_as_tensor(self):
        from utilities.utils import t, device
        return t([self.reward], device=device)

    @staticmethod
    def from_sample(sample):
        return Sample(
            state=sample.state,
            action=sample.action,
            reward=sample.reward,
            next_state=sample.next_state,
            log_pi=sample.log_pi,
            remaining_time=sample.remaining_time
        )
