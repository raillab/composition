import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

"""
Plot the number of object types collected as a function of the weight ratios
"""
if __name__ == '__main':
    data = json.load(open('./results/experiment_weighted_or/tally.json'))

    weights = list(np.arange(1/3, 3.01, 0.05))
    xs = list()
    ys = list()
    std = list()

    for i in range(len(weights)):
        weight = weights[i]
        nums = data[str(i)]
        ratios = np.log([n[0] / n[1] for n in nums])
        xs.append(weight)
        ys.append(np.mean(ratios))
        std.append(stats.sem(ratios))

    print(std)
    plt.errorbar(np.log(xs), ys, yerr=std)
    plt.xlabel('w1/w2')
    plt.ylabel('# beige crate/purple circle')
    plt.show()



#blue = 0.683138; 0.4341905951952436; 211
#purple = 0.6508320000000001; 0.6001137456982634; 478
#composed = 0.7702000000000001; 0.4765169042122219; 297


# 0.48108199999999995
