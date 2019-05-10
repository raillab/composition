"""
Purple circle vs beige square as a function of weights
"""
import gym
import torch
import json
from gym.wrappers.monitor import Monitor

from dqn import ComposedDQN, FloatTensor, get_action
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame
import numpy as np


class MaxLength(gym.Wrapper):
    def __init__(self, env, length):
        gym.Wrapper.__init__(self, env)
        self.max_length = length
        self.steps = 0

    def reset(self):
        self.steps = 0
        return self.env.reset()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.steps += 1
        if self.steps == self.max_length:
            done = True
        return ob, reward, done, info


if __name__ == '__main__':

    max_iterations = 80
    max_episodes = 100
    max_trajectory = 50

    task = MaxLength(WarpFrame(CollectEnv(goal_condition=lambda x: (x.colour == 'beige' and x.shape == 'square')
                                                                   or (x.colour == 'purple' and x.shape == 'circle'))),
                     max_trajectory)
    env = Monitor(task, './experiment_weighted_or/', video_callable=False, force=True)

    dqn_purple_circle = load('./models/purple_circle/model.dqn', task)  # entropy regularised functions
    dqn_beige_crate = load('./models/beige_crate/model.dqn', task)  # entropy regularised functions
    weights = np.arange(1/3, 3.01, 0.05)

    tally = {i: [] for i in range(len(weights))}

    for iter in range(max_iterations):
        for i, weight in enumerate(weights):
            collected_count = [0, 0]
            weight = 1
            dqn_composed = ComposedDQN([dqn_beige_crate, dqn_purple_circle], [weight, 1])
            for episode in range(max_episodes):
                if episode % 1000 == 0:
                    print(episode)
                obs = env.reset()

                for _ in range(max_trajectory):
                    obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
                    # action = dqn_composed(Variable(obs, volatile=True)).data.max(1)[1].view(1, 1)[0][0]
                    action = get_action(dqn_composed, obs)
                    obs, reward, done, info = env.step(action)
                    if done:
                        collected = info['collected']
                        if len([c for c in collected if c.colour == 'beige' and c.shape == 'square']) > 0:
                            collected_count[0] += 1
                        elif len([c for c in collected if c.colour == 'purple' and c.shape == 'circle']) > 0:
                            collected_count[1] += 1
                        else:
                            print("Missed")
                        break
            tally[i].append(collected_count)
            #print('Weight = {}'.format(weight))
            print(tally[i])

        print(tally)


    with open('./experiment_weighted_or_more/tally.json', 'w') as fp:
        json.dump(tally, fp)