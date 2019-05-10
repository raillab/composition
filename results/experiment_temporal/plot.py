import json

import matplotlib.pyplot as plt
import numpy as np

data_composed = json.load(open('./composed/openaigym.episode_batch.1.17246.stats.json'))
data_full = json.load(open('./full_task/openaigym.episode_batch.0.17246.stats.json'))

rewards_composed = data_composed['episode_rewards']
rewards_full = data_full['episode_rewards']

plt.boxplot([rewards_composed, rewards_full], 0, '')
plt.show()

np.savetxt('rewards_composed.txt', rewards_composed, fmt='%.4f')
np.savetxt('rewards_full.txt', rewards_full, fmt='%.4f')
