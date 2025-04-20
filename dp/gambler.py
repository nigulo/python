import sys
sys.path.append("..")
import numpy as np

from dp import DP
import utils.plot as plot

p = 0.25

if __name__ == '__main__':

    capital = np.arange(0, 101)
    states = set(capital)
    actions = {}
    probs = {}
    for s in states:
        if s == 0 or s == 100:
            continue
        actions[s] = np.arange(1, min(s, 100 - s) + 1)
        for a in actions[s]:
            probs[(s, a)] = [(s + a, int(s + a == 100), p), (s - a, 0, 1 - p)]
                        
    print(probs)                
    dp = DP(states, actions, probs)
    v, q, pi = dp.solve(gamma=1, policy_or_value=False, eps=1e-3)

    plt = plot.plot(nrows=2, ncols=1)
    print(pi)
    capital = capital[1:-1]
    v_array = [v[s] for s in capital]
    pi_array = [pi[s] for s in capital]
    plt.plot(capital, v_array, ax_index=[0])
    plt.plot(capital, pi_array, ax_index=[1])
    
    plt.set_axis_limits([0], limits=[[1, 99], None])
    plt.set_axis_limits([1], limits=[[1, 99], None])
    plt.set_axis_title("Value", [0])
    plt.set_axis_title("Policy", [1])
    plt.save("gambler.png")
    plt.close()    

