import json

import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt

data = json.load(open('./tally.json'))



weights = list(np.arange(1/3, 3.01, 0.05))
xs = list()
ys = list()
std = list()
out = list()

for i in range(len(weights)):
    weight = weights[i]
    nums = data[str(i)]
    ratios = [n[0] / n[1] for n in nums]

    out.append((weight, np.median(ratios), np.percentile(ratios, 25), np.percentile(ratios, 75)))

    xs.append(weight)
    ys.append(np.median(ratios))

    std.append(stats.sem(ratios))

# print(std)
plt.errorbar(np.log(xs), np.log(ys), yerr=std)
plt.xlabel('w1/w2')
plt.ylabel('# beige crate/purple circle')
plt.show()

np.savetxt('weighted_or.txt', out, fmt='%.4f')
