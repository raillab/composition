import copy

import numpy as np
import torch
from torch.autograd import Variable

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

start_positions = {'player': (3, 4),
                   'crate_purple': (6, 3),
                   'circle_purple': (7, 7),
                   'circle_beige': (1, 7),
                   'crate_beige': (2, 2),
                   'crate_blue': (8, 1),
                   'circle_blue': (2, 8)}
env = WarpFrame(CollectEnv(start_positions=start_positions,
                           goal_condition=lambda x: x.colour == 'purple' or x.colour == 'blue'))

dqn_blue = load('../../models/blue/model.dqn', env)
dqn_purple = load('../../models/purple/model.dqn', env)
dqn = ComposedDQN([dqn_blue, dqn_purple], [1, 1])

values = np.zeros_like(env.env.board, dtype=float)
for pos in env.env.free_spaces:
    positions = copy.deepcopy(start_positions)

    positions = remove(positions, pos)

    positions['player'] = pos
    env = WarpFrame(CollectEnv(start_positions=positions,
                               goal_condition=lambda x: x.colour == 'purple' or x.colour == 'blue'))
    obs = env.reset()
    obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
    v= get_value(dqn, obs)
    values[pos] = v

values[start_positions['circle_purple']] = 1
values[start_positions['crate_purple']] = 1
values[start_positions['circle_blue']] = 1
values[start_positions['crate_blue']] = 1
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

np.savetxt('value_data_beige_crate.txt', values, fmt='%.4f')
#
# from matplotlib._png import read_png
# img = read_png('../../map.png')
#
#
# x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
# z = values[(x, y) / 40]
#
# # x = np.arange(0, values.shape[0])
# # y = np.arange(0, values.shape[1])
# # x, y = np.meshgrid(x, y)
#
# # # z = values.ravel()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #
# ax.plot_surface(x, y, values,facecolors=img)
#
# # levels = np.arange(np.min(z), np.max(z), 0.1)
# # plt.contour(values, 20, extent=(0.5, 10.5, 0.5, 10.5))
# # plt.show()
#
# # print(values)
# #plt.imshow(values, cmap='gray', interpolation='nearest')
#
#
# plt.show()
#
# # big_values = np.zeros((20, 20))
# # for row in range(len(values)):
# #     for col in range(len(values[row])):
# #
# #         big_values[row * 2, col * 2] = values[row, col]
# #         big_values[row * 2 + 1, col * 2] = values[row, col]
# #         big_values[row * 2, col * 2 + 1] = values[row, col]
# #         big_values[row * 2 + 1, col * 2 + 1] = values[row, col]
# #
# # plt.contour(big_values, 20)
# # plt.show()
# #
#
