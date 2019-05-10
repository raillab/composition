import json

import matplotlib.pyplot as plt
import numpy as np

data_all_and = json.load(open('./full_task/openaigym.episode_batch.0.5738.stats.json'))

rewards_all_and = data_all_and['episode_rewards']


plt.boxplot([rewards_all_and], 0, '')
plt.show()

np.savetxt('rewards_all_and.txt', rewards_all_and, fmt='%.4f')
