from torch.autograd import Variable

from dqn import ComposedDQN, FloatTensor, get_action
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame
import numpy as np
import torch

if __name__ == '__main__':
    start_positions = {'player': (5, 5),
                       'crate_purple': (6, 3),
                       'circle_purple': (7, 7),
                       'circle_beige': (1, 7),
                       'crate_beige': (2, 2),
                       'crate_blue': (8, 1),
                       'circle_blue': (2, 8)}
    env = WarpFrame(CollectEnv(start_positions=start_positions,
                               goal_condition=lambda x: x.shape == 'square' and x.colour == 'blue'))

    dqn1 = load('../../models/crate/model.dqn', env)
    dqn2 = load('../../models/blue/model.dqn', env)
    dqn = ComposedDQN([dqn1, dqn2], [1, 1], or_compose=False)
    obs = env.reset()
    positions = list()
    positions.append(env.env.player.position)
    env.render()

    for _ in range(100):
        obs = np.array(obs)
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)

        action = get_action(dqn, obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        positions.append(env.env.player.position)
        if done:
            obs = env.reset()
            env.render()
            break

    np.savetxt('trajectories/2.txt', positions, fmt='%d')