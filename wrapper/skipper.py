import gym
import numpy as np
from gym.spaces import Box

  
def make_env(env_name, int_reward, max_repeat, vanilla):
    """
    Make an environment, potentially wrapping it in MeasureWrapper.
    
    Args:
        env_name: Environment name
        int_reward: The intrinsic reward given for not measuring to use for the wrapper
        max_repeat: The maximum number of time that the agent can skip action selection cycle
        vanilla: If True, uses the original environment without a wrapper. Ignores int_reward

    Returns:
        A gym environment.
    """
    env = gym.make(env_name)
    if vanilla:
        return VanillaWrapper(env)
    else:
        env = SkipperWrapper(env, int_reward=int_reward, max_repeat=max_repeat)
        return env

class VanillaWrapper(gym.Wrapper):
    def is_measure_action(self, _action):
        return False
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)

        info['int_reward'] = 0.0
        info['ext_reward'] = reward

        return state, reward, done, info

class SkipperWrapper(gym.Wrapper):
    """Augments environments to take a behaviour action and a repeat action.
    Stores the original reward in the info['ext_reward'] attribute, and the intrinsic reward for not measuring in info['int_reward'] attribute.

    "action_pair: (a,k)" the first element is the bahaviour action and the second 
    element is an integer between [1, max_repeat] specifying the number of times
    to repeat the action. The action, a, is applied k times. measurements are 
    skipped for the first k-1 time steps. The total intrinsic reward is k-1 x int_reward
    and the total extrinsic reward is sum of the rewards produced by the behaviour
    policy. 
    
    "a" is any legitiment behaviour actions

    "k" is an integer in the range [1, max_repeat]

    "total_reward" the total reward is the sum of the int_rewards and ext_rewards
    """
    def __init__(self, env, int_reward, max_repeat):
        super().__init__(env)
        self.int_reward = int_reward
        self.max_repeat = max_repeat
        self.continuous_action_space = True if env.action_space.shape else False
        
        #Action space becomes a tuple where the first element is the number of time to repeat and the second action is the behaviour action
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.max_repeat), env.action_space))

        self.observation_space = env.observation_space

    def step(self, action_pair):        
        skipper_num = action_pair[0]+1
        action = action_pair[1]
        int_reward = -self.int_reward
        ext_reward = 0
        for _ in range(skipper_num):
            print("step")
            state, reward, done, info = self.env.step(action)
            int_reward += self.int_reward
            ext_reward += reward
            if done:
                break
        info['int_reward'] = int_reward
        info['ext_reward'] = ext_reward
        return state, int_reward+ext_reward, done, info
    
    def render(self):
        pass
    
    def reset(self):
        state = self.env.reset()
        return state