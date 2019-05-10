import json

import matplotlib.pyplot as plt
import numpy as np

data_and = json.load(open('./blue_and_crate/openaigym.episode_batch.2.2723.stats.json'))
data_blue_crate = json.load(open('./blue_crate/openaigym.episode_batch.0.2723.stats.json'))
data_or = json.load(open('./blue_or_crate/openaigym.episode_batch.1.2723.stats.json'))
rewards_and = data_and['episode_rewards']
rewards_blue_crate = data_blue_crate['episode_rewards']
rewards_or = data_or['episode_rewards']

plt.boxplot([rewards_blue_crate, rewards_and, rewards_or], 0, '')
plt.show()

np.savetxt('rewards_and.txt', rewards_and, fmt='%.4f')
np.savetxt('rewards_blue_crate.txt', rewards_blue_crate, fmt='%.4f')
np.savetxt('rewards_or.txt', rewards_or, fmt='%.4f')
