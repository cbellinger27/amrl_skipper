import gym
from wrapper.skipper import make_env

env = make_env('CartPole-v0', 1, 3, False)
env.reset()

for _ in range(5):
    done = False
    while not done:
        ap = env.action_space.sample()
        print(ap)
        s, r, done, i = env.step(ap)
    env.reset()
