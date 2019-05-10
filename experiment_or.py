"""
Experiment for task collect blue OR purple objects
"""
import torch
from gym.wrappers import Monitor

from dqn import ComposedDQN, FloatTensor, get_action
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame, MaxLength

if __name__ == '__main__':

    max_episodes = 50000
    max_trajectory = 50

    task = MaxLength(WarpFrame(CollectEnv(goal_condition=lambda x: x.colour == 'blue' or x.colour == 'purple')),
                     max_trajectory)

    dqn_blue = load('./models/blue/model.dqn', task)
    dqn_purple = load('./models/purple/model.dqn', task)
    dqn_composed = ComposedDQN([dqn_blue, dqn_purple], [1, 1])

    for dqn, name in [(dqn_blue, 'blue'), (dqn_purple, 'purple'), (dqn_composed, 'composed')]:
        env = Monitor(task, './experiment_or/' + name + '/', video_callable=False, force=True)
        for episode in range(max_episodes):
            if episode % 1000 == 0:
                print(episode)
            obs = env.reset()
            for _ in range(max_trajectory):
                obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
                action = get_action(dqn, obs)
                obs, reward, done, _ = env.step(action)
                env.render()
                if done:
                    break
