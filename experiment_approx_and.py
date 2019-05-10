"""
Experiment for approximating blue AND square
"""

from gym.wrappers import Monitor

from dqn import ComposedDQN, get_action
from gym_repoman.envs import CollectEnv
from trainer import load
from wrappers import WarpFrame, MaxLength

if __name__ == '__main__':

    max_episodes = 50000
    max_trajectory = 50

    task = MaxLength(WarpFrame(CollectEnv(goal_condition=lambda x: x.colour == 'blue' and x.shape == 'square')),
                     max_trajectory)

    dqn_blue_crate = load('./models/blue_crate/model.dqn', task)
    dqn_blue = load('./models/blue/model.dqn', task)
    dqn_crate = load('./models/crate/model.dqn', task)

    dqn_composed_or = ComposedDQN([dqn_blue, dqn_crate])
    dqn_composed_and = ComposedDQN([dqn_blue, dqn_crate], or_compose=False)

    for dqn, name in [(dqn_blue_crate, 'blue_crate'), (dqn_composed_or, 'blue_or_crate'),
                      (dqn_composed_and, 'blue_and_crate')]:

        env = Monitor(task, './experiment_approx_and/' + name + '/', video_callable=False, force=True)
        for episode in range(max_episodes):
            if episode % 1000 == 0:
                print(episode)
            obs = env.reset()
            for _ in range(max_trajectory):
                action = get_action(dqn, obs)
                obs, reward, done, _ = env.step(action)
                if done:
                    break
