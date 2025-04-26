import sys
sys.path.append("../")
sys.path.append("../../..")
import numpy as np
from scipy.stats import poisson
import pickle
import os

from dp import DP
import utils.plot as plot

eps = 1e-5

MAX_CARS = 20
MIN_CARS = 0
MAX_CARS_MOVE = 5

RENT_COST = 10
MOVE_COST = 2

DAY = 0
NIGHT = 1

LOCATION_1 = 0
LOCATION_2 = 1

def load(name="data"):
    file = f"{name}.pkl"
    if os.path.isfile(file):
        return pickle.load(open(file, "rb"))
    else:
        return None

def save(data, name="data"):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)

class States:
    def __init__(self):
        self.n1 = MIN_CARS
        self.n2 = MIN_CARS
        self.time = DAY

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        n1 = self.n1
        n2 = self.n2
        time = self.time
        if self.time == DAY:
            self.time = NIGHT
        else:
            self.time = DAY
            self.n2 += 1
            if self.n2 > MAX_CARS:
                self.n2 = MIN_CARS
                self.n1 += 1
        if self.n1 > MAX_CARS and self.time == NIGHT:
            self.__init__()
            raise StopIteration
        return n1, n2, time

def p_rented(location, n):
    return poisson.pmf(n, 3) if location == LOCATION_1 else poisson.pmf(n, 4)

def p_returned(location, n):
    return poisson.pmf(n, 3) if location == LOCATION_1 else poisson.pmf(n, 2)

def avg_rented(location):
    return 3 if location == LOCATION_1 else 4

def avg_renturend(location):
    return 3 if location == LOCATION_1 else 2

def get_actions(s):
    _, _, time = s
    if time == DAY:
        return {None}
    return set(np.arange(-MAX_CARS_MOVE, MAX_CARS_MOVE+1))

def get_rewards(loc, n):
    rewards = {}
    n_rets = np.arange(MAX_CARS - n + 1)
    p_rets = [p_returned(loc, n_ret) for n_ret in n_rets]
    p_rets[-1] +=  1 - np.sum(p_rets)
    for i, n_ret in enumerate(n_rets):
        p_ret = p_rets[i]
        n_rens = np.arange(n + n_ret - MIN_CARS + 1)
        p_rens = [p_rented(loc, n_ren) for n_ren in n_rens]
        p_rens[-1] +=  1 - np.sum(p_rens)
        for j, n_ren in enumerate(n_rens):
            p_ren = p_rens[j]
            n_diff = n_ret - n_ren
            r, p = rewards.get(n_diff, (0, 0))
            p_ret_ren = p_ret*p_ren
            rewards[n_diff] = (r + RENT_COST*n_ren*p_ret_ren, p + p_ret_ren)
    return rewards

def get_transitions(s, a):
    n1, n2, time = s
    if time == DAY:
        rewards1 = get_rewards(LOCATION_1, n1)
        rewards2 = get_rewards(LOCATION_2, n2)
        ret_val = []
        for n1_diff in rewards1.keys():
            r1, p1 = rewards1[n1_diff]
            r1 /= p1
            for n2_diff in rewards2.keys():
                r2, p2 = rewards2[n2_diff]
                r2 /= p2
                if p1*p2 > eps:
                    ret_val.append(((n1 + n1_diff, n2 + n2_diff, NIGHT), r1+r2, p1*p2))
        return ret_val
    n1_prime = n1 + a
    n2_prime = n2 - a
    if n1_prime >= MIN_CARS and n1_prime <= MAX_CARS and n2_prime >= MIN_CARS and n2_prime <= MAX_CARS:
        return [((n1_prime, n2_prime, DAY), -MOVE_COST*abs(a), 1)]
    return [((n1, n2, DAY), -MOVE_COST*MAX_CARS_MOVE, 1)]
    
if __name__ == '__main__':
    transitions = load()
    if transitions is None:
        print("Initializing...")
        transitions = {}
        for s in States():
            for a in get_actions(s):
                transitions[(s, a)] = get_transitions(s, a)
        save(transitions)

    result = load("result")
    if result is None:
        print("Optimizing...")
        dp = DP(States(), get_actions, transitions)
        result = dp.solve()
        save(result, "result")
    v, q, pi = result
    
    policy = np.empty((MAX_CARS, MAX_CARS))
    value = np.empty((MAX_CARS, MAX_CARS))
    for n1 in range(policy.shape[0]):
        for n2 in range(policy.shape[1]):
            policy[n1, n2] = pi.get((n1, n2, NIGHT), 0)
            value[n1, n2] = v.get((n1, n2, NIGHT), 0)
    
    plt = plot.plot(nrows=1, ncols=2, size=plot.default_size(policy.shape[0]*50, policy.shape[1]*50))
    plt.colormap(policy.T[::-1,::], [0], show_colorbar=True, cmap_name="bwr", vmin=-MAX_CARS_MOVE, vmax=MAX_CARS_MOVE)
    plt.colormap(value.T[::-1,::], [1])
    
    plt.set_axis_title("Policy", [0])
    plt.set_axis_title("Value", [1])
    plt.save("car_rental.png")
    plt.close()

    print("Done.")
    
