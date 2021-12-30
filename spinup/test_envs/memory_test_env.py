import gym
import numpy as np
import gym.spaces

from gym.envs.registration import register

register(
    id='MemoryTestContinuousEnv-v0',
    entry_point='spinup.test_envs.memory_test_env:MemoryTestContinuousEnv'
)

class MemoryTestContinuousEnv(gym.Env):
    """ A simple toy game designed to evaluate an RL agent's ability to memorize information over time.
    The game works as follows: The agent observes a sequence of n-1 states that are all neutral (white frame) except for one colored state somewhere inbetween.
    In the final n-th state (a black frame), the agents needs to play the action that matches the colored state that it has observed before. 
    If it does, it obtains a reward and the episode ends, otherwise the episode ends without a reward.
    Also, the env allows you to use min_distance_to_reward to specified the minimum number of neutral states between the colored states and the final black state 
    - which can, e.g., be used to make sure that the temporal relationship is never observed directly during training. 
    """
    def __init__(self, sequence_length=10, min_distance_to_reward=10):
        super(MemoryTestContinuousEnv, self).__init__()
        self.sequence_length = sequence_length
        self.min_distance_to_reward = min_distance_to_reward
        self.obs_shape = (3,)
        self.nbr_actions = 3
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({"state":gym.spaces.Box(low=-1.0, high=1.0, shape=self.obs_shape)})
        #self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.obs_shape)
        
        self.white_state = np.ones(self.obs_shape, dtype=np.float32)
        self.black_state = -np.ones(self.obs_shape, dtype=np.float32)
        
        self.red_state = np.array([1.0, -1.0, -1.0], dtype=np.float32)   
        self.green_state = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
        self.blue_state = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        
        self.colors = [np.array([1.0, -1.0, -1.0], dtype=np.float32), np.array([-1.0, 1.0, -1.0], dtype=np.float32), np.array([-1.0, -1.0, 1.0], dtype=np.float32)]
        self.max_dist = 2.0 * 3.0
        self.colored_states = [self.red_state, self.green_state, self.blue_state]

        self.current_state = 0
        self.current_sequence, self.desired_final_action = self.generate_sequence()

    def generate_sequence(self):
        if self.sequence_length > 1:
            seq = [self.white_state for _ in range(self.sequence_length-1)] + [self.black_state] 
        else:
            seq = [self.black_state]
        rnd_idx = np.random.randint(0, self.sequence_length-1-self.min_distance_to_reward) if self.sequence_length >= 2 + self.min_distance_to_reward else 0
        rnd_color_idx = np.random.randint(0, len(self.colored_states))
        seq[rnd_idx] = self.colored_states[rnd_color_idx]
        return seq, self.colors[rnd_color_idx]

    def step(self, action_vec):

        thresh = 0.0
        action = np.array([-1.0 if a <= -thresh else 1.0 if a >= thresh else 0.0 for a in action_vec]) # TODO keep?

        if self.current_state == self.sequence_length-1:
            #rew = np.linalg.norm(action_vec - self.desired_final_action) / self.max_dist
            # I think the optimal uninformed agent plays [-1,-1,-1] and gets on average 2/3 of the positions right.
            #rew = np.sum(np.abs(action_vec - self.desired_final_action)) / self.max_dist

            #if all(action == self.desired_final_action):
            #    print(f"predicted {self.desired_final_action}")
            rew = 1.0 if all(action == self.desired_final_action) else 0

            #rew = (rew > 0.8) * 10 # TODO remove
            return {"state":self.current_sequence[self.current_state]}, rew, True, {}
            #return self.current_sequence[self.current_state], rew, True, {}
        else:
            self.current_state += 1
            return {"state":self.current_sequence[self.current_state]}, 0, False, {}
            #return self.current_sequence[self.current_state], 0, False, {}

    def reset(self):
        self.current_state = 0
        self.current_sequence, self.desired_final_action = self.generate_sequence()
        return {"state":self.current_sequence[self.current_state]}
        #return self.current_sequence[self.current_state]

    def render(self, mode="rgb_array"):
        return self.current_sequence[self.current_state]
        
    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length
