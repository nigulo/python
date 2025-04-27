import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../")
sys.path.append("../../..")

from td import Method
from td import TD
import utils.plot as plot

NX = 12
NY = 4

CLIFF = [(x, 0) for x in range(1, NX -1)]
X_START, Y_START = 0, 0
X_GOAL, Y_GOAL = NX - 1, 0


def actions(s):
    x, y = s
    actions = {(-1, 0), (0, -1), (0, 1), (1, 0)}
    if x == 0:
        actions = actions.difference({(-1, 0)})
    if y == 0:
        actions = actions.difference({(0, -1)})
    if x == NX - 1:
        actions = actions.difference({(1, 0)})
    if y == NY - 1:
        actions = actions.difference({(0, 1)})
    return actions                

def transitions(s, a):
    if s == (X_GOAL, Y_GOAL):
        return []
    x, y = s
    dx, dy = a
    x += dx
    y += dy
    if (x, y) in CLIFF:
        return [((X_START, Y_START), -100, 1)]
    return [((x, y), -1, 1)]


def total_reward(pi):
    reward = 0
    s = X_START, Y_START
    for _ in range(NX*NY):
        if s == (X_GOAL, Y_GOAL):
            break
        if s not in pi:
            return -NX*NY*100
        [(s, r, _)] = transitions(s, pi[s])
        reward += r
    return reward

if __name__ == '__main__':                        
    n_episodes=100

    alphas = np.arange(0.1, 1.1, step=0.1)
    sarsa_rewards = np.empty_like(alphas)
    q_learning_rewards = np.empty_like(alphas)
    expected_sarsa_rewards = np.empty_like(alphas)
    t = tqdm(total=len(alphas)*3*n_episodes//100, desc="Calculating stats")
    for i, alpha in enumerate(alphas):
        td = TD(actions, transitions, (lambda _: (X_START, Y_START)))
        for _ in range(n_episodes//100):
            q, pi = td.train(n_episodes=100, method=Method.SARSA)
            t.update()
        sarsa_rewards[i] = total_reward(pi)
        
        td = TD(actions, transitions, (lambda _: (X_START, Y_START)))
        for _ in range(n_episodes//100):
            q, pi = td.train(n_episodes=100, method=Method.Q_LEARNING)
            t.update()
        q_learning_rewards[i] = total_reward(pi)
    
        td = TD(actions, transitions, (lambda _: (X_START, Y_START)))
        for _ in range(n_episodes//100):
            q, pi = td.train(n_episodes=100, method=Method.EXPECTED_SARSA)
            t.update()
        expected_sarsa_rewards[i] = total_reward(pi)

    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(NX*25, NY*25))
    
    plt.plot(alphas, sarsa_rewards, "blue")    
    plt.plot(alphas, q_learning_rewards, "red")    
    plt.plot(alphas, expected_sarsa_rewards, "green")    
    plt.save("td_stats.png")
    plt.close()    
