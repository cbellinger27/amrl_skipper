#%%

#!/usr/bin/env python3
import gym
from wrapper.skipper import make_env
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation_behave', 'observation_skipper', 'action'])


def iterate_batches(env, net_bahave, net_skipper, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        
        act_probs_v_bahave = sm(net_behave(obs_v))
        act_probs_bahave = act_probs_v_bahave.data.numpy()[0]
        action_bahave = np.random.choice(len(act_probs_bahave), p=act_probs_bahave)

        obs_v_skipper = torch.FloatTensor([obs,action_bahave])
        act_probs_v_skipper = sm(net_skipper(obs_v_skipper))
        act_probs_skipper = act_probs_v_skipper.data.numpy()[0]
        action_skipper = np.random.choice(len(act_probs_skipper), p=act_probs_skipper)
        
        next_obs, reward, is_done, _, _ = env.step((action_skipper, action_bahave))
        episode_reward += reward
        step = EpisodeStep(observation_behave=obs, observation_skipper=[obs,action_skipper],action=(action_skipper, action_bahave))
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs_behave = []
    train_act_skipper = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs_behave.extend(map(lambda step: step.observation_behave, steps))
        train_obs_skipper.extend(map(lambda step: step.observation_skipper, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v_behave = torch.FloatTensor(train_obs_behave)
    train_obs_v_skipper = torch.FloatTensor(train_obs_skipper)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v_behave, train_obs_v_skipper, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = make_env('CartPole-v1', 1, 3, False)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions_behave = env.action_space.spaces[1].n
    n_actions_skipper = env.action_space.spaces[0].n

    net_behave = Net(obs_size, HIDDEN_SIZE, n_actions_behave)
    net_skipper = Net(obs_size+1, HIDDEN_SIZE, n_actions_skipper)
    
    objective_behave = nn.CrossEntropyLoss()
    optimizer_behave = optim.Adam(params=net_behave.parameters(), lr=0.01)

    objective_skipper = nn.CrossEntropyLoss()
    optimizer_skipper = optim.Adam(params=net_skipper.parameters(), lr=0.01)

    writer = SummaryWriter(comment="-cartpole_skipper")

    for iter_no, batch in enumerate(iterate_batches(
            env, net_behave, net_skipper, BATCH_SIZE)):
        obs_v, acts_v_behave, acts_v_skipper, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE)
        optimizer_behave.zero_grad()
        action_scores_v_behave = net_behave(obs_v)
        loss_v_behave = objective_behave(action_scores_v_behave, acts_v_behave)
        loss_v_behave.backward()
        optimizer_behave.step()

        optimizer_skipper.zero_grad()
        action_scores_v_skipper = net_skipper(obs_v)
        loss_v_skipper = objective_behave(action_scores_v_skipper, acts_v_skipper)
        loss_v_skipper.backward()
        optimizer_skipper.step()

        print("%d: loss_behave=%.3f, loss_skipper=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v_behave.item(), loss_v_skipper.item(),reward_m, reward_b))
        writer.add_scalar("loss", loss_v_behave.item(), loss_v_skipper.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
# %%
