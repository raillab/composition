"""
Generate the value function plots for weighted combinations
"""
import copy

import numpy as np
import torch

from dqn import ComposedDQN, FloatTensor, get_value
from gym_repoman.envs import CollectEnv
from trainer import load
from wrappers import WarpFrame


def remove(positions, pos):
    found = None
    for key, val in positions.items():
        if key != 'player' and pos == val:
            found = key
            break

    if found is not None:
        positions[found] = None
    return positions


if __name__ == '__main__':

    start_positions = {'player': (3, 4),
                       'crate_purple': (6, 3),
                       'circle_purple': (7, 7),
                       'circle_beige': (1, 7),
                       'crate_beige': (2, 2),
                       'crate_blue': (8, 1),
                       'circle_blue': (2, 8)}

    env = WarpFrame(CollectEnv(start_positions=start_positions,
                               goal_condition=lambda x: (x.colour == 'beige' and x.shape == 'square')
                                                        or (x.colour == 'purple' and x.shape == 'circle')))

    dqn_purple_circle = load('../../models/purple_circle/model.dqn', env)
    dqn_beige_crate = load('../../models/beige_crate/model.dqn', env)
    dqn = ComposedDQN([dqn_purple_circle, dqn_beige_crate], [3, 2])  # TODO put weights here!

    values = np.zeros_like(env.env.board, dtype=float)
    for pos in env.env.free_spaces:
        positions = copy.deepcopy(start_positions)

        positions = remove(positions, pos)

        positions['player'] = pos
        env = WarpFrame(CollectEnv(start_positions=positions,
                                   goal_condition=lambda x: (x.colour == 'beige' and x.shape == 'square')
                                                            or (x.colour == 'purple' and x.shape == 'circle')))

        obs = env.reset()
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
        v = get_value(dqn, obs)
        values[pos] = v

    values[start_positions['circle_purple']] = 3
    values[start_positions['crate_beige']] = 2

    update = list(map(tuple, np.argwhere(values == 0)))

    for _ in range(1000):
        for i in range(len(values)):
            for j in range(len(values[i])):
                if (i, j) in update:

                    s = values[i, j]
                    count = 1
                    for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        m = i + offset[0]
                        n = j + offset[1]
                        if m < 0 or m >= len(values) or n < 0 or n >= len(values[i]):
                            continue
                        count += 1
                        s += values[m, n]
                    values[i, j] = s / count

    print(values)

    np.savetxt('value_data32.txt', values, fmt='%.4f')