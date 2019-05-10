"""
Experiment for collecting ALL objects
"""
import gym
import torch
from gym.wrappers import Monitor

from dqn import ComposedDQN, FloatTensor, get_action
from trainer import load
from gym_repoman.envs import CollectEnv, MultiCollectEnv
from wrappers import WarpFrame


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
    max_episodes = 50000
    max_trajectory = 200

    targets = {('purple', 'square'), ('purple', 'circle'), ('blue', 'square'), ('blue', 'circle'), ('beige', 'square'),
               ('beige', 'circle')}
    task = MaxLength(WarpFrame(
        MultiCollectEnv(lambda collected: targets.issubset({(c.colour, c.shape) for c in collected}),
                        lambda x: (x.colour, x.shape) in targets)), max_trajectory)

    #agent = train('./models/temporal3/results', task) # 1 million
    #save('./models/temporal3/model.dqn', agent)

    dqn = load('./models/temporal3/model.dqn', task)  # dqn trained on full task

    max_episodes = 50000
    max_trajectory = 50

    dqn1 = load('./models/purple/model.dqn', task)
    dqn2 = load('./models/blue/model.dqn', task)
    dqn3 = load('./models/beige/model.dqn', task)
    dqn_composed = ComposedDQN([dqn1, dqn2, dqn3])

    for dqn, name in [(dqn, 'full_task'),  (dqn_composed, 'composed')]:
        env = Monitor(task, './experiment_temporal/' + name + '/', video_callable=False, force=True)
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