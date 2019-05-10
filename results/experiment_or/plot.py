import json

import matplotlib.pyplot as plt
import numpy as np

data_blue = json.load(open('./blue/openaigym.episode_batch.0.6412.stats.json'))
data_blue_or_purple = json.load(open('./composed/openaigym.episode_batch.2.6412.stats.json'))
data_purple = json.load(open('./purple/openaigym.episode_batch.1.6412.stats.json'))

rewards_blue = data_blue['episode_rewards']
rewards_blue_or_purple = data_blue_or_purple['episode_rewards']
rewards_purple = data_purple['episode_rewards']

print(np.mean(rewards_blue_or_purple)) # 0.77

plt.boxplot([rewards_blue, rewards_blue_or_purple, rewards_purple], 0, '')
plt.show()

np.savetxt('rewards_blue.txt', rewards_blue, fmt='%.4f')
np.savetxt('rewards_blue_or_purple.txt', rewards_blue_or_purple, fmt='%.4f')
np.savetxt('rewards_purple.txt', rewards_purple, fmt='%.4f')
