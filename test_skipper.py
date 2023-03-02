import gym
from wrapper.skipper import make_env

env = make_env('CartPole-v0', 1, 3, False)
env.reset()

for _ in range(5):
    done = False
    trunc = False
    while not done and not trunc:
        ap = env.action_space.sample()
        print(ap)
        s, r, done, trunc, i = env.step(ap)
        print(i)
        print(r)
    env.reset()
