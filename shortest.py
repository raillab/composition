import numpy as np
import itertools


"""
Use Floyd-Warshall to compute shortest distances between all states to compute optimal reward in expectation
"""
if __name__ == '__main__':
    board = ['##########',
             '#        #',
             '#        #',
             '#    #   #',
             '#   ##   #',
             '#  ##    #',
             '#   #    #',
             '#        #',
             '#        #',
             '##########']

    arr = np.array([list(row) for row in board])
    free_spaces = list(map(tuple, np.argwhere(arr != '#')))

    dist = {(x, y) : np.inf for x in free_spaces for y in free_spaces}

    for (u, v) in dist.keys():
        d = abs(u[0] - v[0]) + abs(u[1] - v[1])
        if d == 0:
            dist[(u, v)] = 0
        elif d == 1:
            dist[(u, v)] = 1

    for k in free_spaces:
        for i in free_spaces:
            for j in free_spaces:
                if dist[(i, j)] > dist[(i, k)] + dist[(k, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]


    reward = []
    count = 0
    N = 2
    for points in itertools.combinations(free_spaces, N):
        distances = [dist[(points[0], points[i])] for i in range(1, N)]
        d = np.min(distances)
        if d > 0:
            reward.append(1 + -0.1 * (d-1))

    print(reward)
    print(np.mean(reward))
    print(np.std(reward))

# for i in range(1):
#
#     env = CollectEnv(goal_condition=lambda x: x.colour == 'purple' and x.shape == 'square')
#     env.reset()
#     env.render()
#     for _ in range(10000):
#         obs, reward, done, _ = env.step(env.action_space.sample())
#         env.render()
#         if done:
#             env.reset()




# 1 let dist be a |V| × |V| array of minimum distances initialized to ∞ (infinity)
# 2 for each edge (u,v)
# 3    dist[u][v] ← w(u,v)  // the weight of the edge (u,v)
# 4 for each vertex v
# 5    dist[v][v] ← 0
# 6 for k from 1 to |V|
# 7    for i from 1 to |V|
# 8       for j from 1 to |V|
# 9          if dist[i][j] > dist[i][k] + dist[k][j]
# 10             dist[i][j] ← dist[i][k] + dist[k][j]
# 11         end if
