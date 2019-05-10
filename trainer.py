import numpy as np
import torch
from gym.wrappers import Monitor

from dqn import Agent, DQN, FloatTensor, ComposedDQN, get_action
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame


def video_callable(episode_id):
    return episode_id > 1 and episode_id % 500 == 0


def train(path, env):
    env = Monitor(env, path, video_callable=video_callable, force=True)
    agent = Agent(env)
    agent.train()
    return agent


def save(path, agent):
    torch.save(agent.q_func.state_dict(), path)


def load(path, env):
    dqn = DQN(env.action_space.n)
    dqn.load_state_dict(torch.load(path))
    return dqn


def enjoy(dqn, env, timesteps):
    obs = env.reset()
    env.render()
    for _ in range(timesteps):
        obs = np.array(obs)
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
        # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
        #weights = nn.Softmax()(dqn(Variable(obs, volatile=True)))
        #action = torch.multinomial(weights, 1, replacement=True).data[0][0]
        action = get_action(dqn, obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            env.render()


def compose(dqns, weights):
    return ComposedDQN(dqns, weights)


def learn(colour, shape, condition):
    name = colour + shape
    base_path = './models/{}/'.format(name)
    env = WarpFrame(CollectEnv(goal_condition=condition))
    agent = train(base_path + 'results', env)
    save(base_path + 'model.dqn', agent)

if __name__ == '__main__':

    learn('purple', 'circle', lambda x: x.colour == 'purple' and x.shape == 'circle')
