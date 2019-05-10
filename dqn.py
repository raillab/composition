import numpy as np
import random
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        indices = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(indices)


class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        # for grey scale
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        # for single frame colour
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        # for 2 stacked frames colour
        # self.conv1 = nn.Conv2d(6, 32, kernel_size=8, stride=4)
        # for 3 stacked frames colour
        # self.conv1 = nn.Conv2d(9, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(3136, 512)
        # self.linear1 = nn.Linear(18496, 512)
        self.head = nn.Linear(512, self.n_action)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.linear1(x.view(x.size(0), -1)))
        x = self.head(x)
        return x


class ComposedDQN(nn.Module):
    def __init__(self, dqns, weights=None, or_compose=True):
        super().__init__()
        self.dqns = dqns
        if weights is None:
            self.weights = [1] * len(dqns)
        else:
            self.weights = weights
        self.or_compose = or_compose

    def forward(self, x):
        qs = [self.dqns[i](x) * self.weights[i] for i in range(len(self.weights))]
        q = torch.stack(tuple(qs), 2)
        if self.or_compose:
            return q.max(2)[0]
        return 0.5 * q.sum(2)


def get_value(dqn, obs):
    return dqn(Variable(obs, volatile=True)).data.max(1)[0].item()

def get_action(dqn, obs):
    return dqn(Variable(obs, volatile=True)).data.max(1)[1].item()

class Agent(object):
    def __init__(self,
                 env,
                 max_timesteps=2000000,
                 learning_starts=10000,
                 train_freq=4,
                 target_update_freq=1000,
                 learning_rate=1e-4,
                 batch_size=32,
                 replay_buffer_size=1000000,
                 gamma=0.99,
                 eps_initial=1.0,
                 eps_final=0.01,
                 eps_timesteps=500000,
                 print_freq=10):
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        self.env = env
        self.max_timesteps = max_timesteps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.print_freq = print_freq

        self.eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)

        self.q_func = DQN(self.env.action_space.n)
        self.target_q_func = DQN(self.env.action_space.n)
        self.target_q_func.load_state_dict(self.q_func.state_dict())

        if use_cuda:
            self.q_func.cuda()
            self.target_q_func.cuda()

        self.optimizer = optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.steps = 0

    def select_action(self, obs):
        sample = random.random()
        eps_threshold = self.eps_schedule(self.steps)
        if sample > eps_threshold:
            obs = np.array(obs)
            obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return self.q_func(Variable(obs, volatile=True)).data.max(1)[1].view(1, 1)
        else:
            sample_action = self.env.action_space.sample()
            return torch.IntTensor([[sample_action]])

    def train(self):
        obs = self.env.reset()
        episode_rewards = [0.0]

        for t in range(self.max_timesteps):
            action = self.select_action(obs)
            new_obs, reward, done, info = self.env.step(action[0][0])
            self.replay_buffer.add(obs, action, reward, new_obs, done)
            obs = new_obs

            episode_rewards[-1] += reward
            if done:
                obs = self.env.reset()
                episode_rewards.append(0.0)

            if t > self.learning_starts and t % self.train_freq == 0:
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
                obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor))
                act_batch = Variable(torch.from_numpy(act_batch).type(LongTensor))
                rew_batch = Variable(torch.from_numpy(rew_batch).type(FloatTensor))
                next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(FloatTensor))
                not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(FloatTensor)

                if use_cuda:
                    act_batch = act_batch.cuda()
                    rew_batch = rew_batch.cuda()

                current_q_values = self.q_func(obs_batch).gather(1, act_batch.squeeze(2)).squeeze()
                next_max_q = self.target_q_func(next_obs_batch).detach().max(1)[0]
                next_q_values = not_done_mask * next_max_q
                target_q_values = rew_batch + (self.gamma * next_q_values)

                loss = F.smooth_l1_loss(current_q_values, target_q_values)

                self.optimizer.zero_grad()
                loss.backward()
                for params in self.q_func.parameters():
                    params.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            # Periodically update the target network by Q network to target Q network
            if t > self.learning_starts and t % self.target_update_freq == 0:
                self.target_q_func.load_state_dict(self.q_func.state_dict())

            self.steps += 1

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and self.print_freq is not None and len(episode_rewards) % self.print_freq == 0:
                print("--------------------------------------------------------")
                print("steps {}".format(t))
                print("episodes {}".format(num_episodes))
                print("mean 100 episode reward {}".format(mean_100ep_reward))
                print("% time spent exploring {}".format(int(100 * self.eps_schedule(t))))
                print("--------------------------------------------------------")
