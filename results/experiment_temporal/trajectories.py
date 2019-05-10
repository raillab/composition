import numpy as np
import torch

from dqn import ComposedDQN, FloatTensor, get_action
from gym_repoman.envs import MultiCollectEnv
from trainer import load
from wrappers import WarpFrame

if __name__ == '__main__':
    start_positions = {'player': (5, 5),
                       'crate_purple': (6, 3),
                       'circle_purple': (7, 7),
                       'circle_beige': (1, 7),
                       'crate_beige': (2, 2),
                       'crate_blue': (8, 1),
                       'circle_blue': (2, 8)}

    targets = {('purple', 'square'), ('blue', 'circle'), ('blue', 'square'), ('beige', 'square')}

    env = WarpFrame(
        MultiCollectEnv(termination_condition=lambda collected: targets.issubset({(c.colour, c.shape) for c in collected}),
                        reward_condition=lambda x: (x.colour, x.shape) in targets, start_positions=start_positions))

    dqn1 = load('../../models/purple/model.dqn', env)
    # dqn2 = load('../../models/purple_circle/model.dqn', env)
    dqn3 = load('../../models/blue/model.dqn', env)
    dqn4 = load('../../models/beige/model.dqn', env)
    # dqn = ComposedDQN([dqn1, dqn2, dqn3, dqn4], [1,1,1,1])
    #dqn1 = load('../../models/crate/model.dqn', env)
    #dqn2 = load('../../models/blue/model.dqn', env)
    dqn = ComposedDQN([dqn1, dqn3, dqn4])

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
