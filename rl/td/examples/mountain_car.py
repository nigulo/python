import sys
sys.path.append("../")
sys.path.append("../../..")
import numpy as np
import numpy.random as random
import pickle
import os

from td_func import TD
import utils.plot as plot

eps = 1e-5


x_min, x_max = -1.2, 0.5
v_min, v_max = -0.07, 0.07

x_range = x_max - x_min
v_range = v_max - v_min

n = 8
n2 = 8
nn2 = n*n2
d = n2*n*n*3

deltas = np.array([[0, 0], [1, 3], [-3, 1], [-2, 6], [6, -2], [3, -9], [9, -3], [-12, 4]])*[x_range/nn2, v_range/nn2]

def load(name="data"):
    file = f"{name}.pkl"
    if os.path.isfile(file):
        return pickle.load(open(file, "rb"))
    else:
        return None

def save(data, name="data"):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)

def actions(s):
    return {-1, 0, 1}

def transitions(s, a):
    x, v = s
    if x >= x_max:
        return []
    if x < x_min:
        return [((x_min, 0), -1, 1)]
    
    x1 = max(x_min, min(x_max, x + v))
    v1 = max(v_min, min(v_max, v + 0.001*a - 0.0025*np.cos(3*x)))
        
    return [((x1, v1), -1, 1)]

def s0(_):
    return (0.2*random.random() - 0.6, 0)

def get_features(x, v, a):
    features = np.zeros((n2, n, n, 3))
    a_ind = a + 1

    for i, (x_delta, v_delta) in enumerate(deltas):
        x_ind = int(round((n-1)*(x - x_min + x_delta) / x_range))
        v_ind = int(round((n-1)*(v - v_min + v_delta) / v_range))
        
        if 0 <= x_ind < n and 0 <= v_ind < n:
            features[i, x_ind, v_ind, a_ind] = 1
    
    return features.flatten()

def q(s, a, w):
    x, v = s
    features = get_features(x, v, a)
    return np.sum(features*w)

def q_grad(s, a, w):
    x, v = s
    features = get_features(x, v, a)
    return features
    
if __name__ == '__main__':
    td = TD(actions, transitions, d, q, q_grad)
    w = td.train(s0, n_episodes=100)
    
    q_max = np.ones((nn2, nn2))
    for i in range(nn2):
        x = x_min + x_range*i/nn2
        for j in range(nn2):
            v = v_min + v_range*j/nn2
            q_s_a = [q((x, v), a, w) for a in actions((x, v))]
            q_max[i, j] = q_s_a[np.argmax(q_s_a)]
    q_max[q_max == 1] = np.min(q_max)
    
    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(n*50, n*50))
    plt.colormap(q_max.T[::-1,::], 0, show_colorbar=True)
    

    x, v = s0(0)
    for _ in range(nn2*nn2):
        as_ = list(actions((x, v)))
        q_s_a = [q((x, v), a, w) for a in as_]
        a = as_[np.argmax(q_s_a)]
        t = transitions((x, v), a)
        if not t:
            break
        [((x_prime, v_prime), _, _)] = t
        if x_min < x < x_max and x_min < x_prime < x_max:
            plt.line((x-x_min)/x_range*nn2+0.5, (v-v_min)/v_range*nn2+0.5, (x_prime-x_min)/x_range*nn2+0.5, (v_prime-v_min)/v_range*nn2+0.5, ax_index=0, color='r', linestyle='-', linewidth=1.5, alpha=0.5)
        x = x_prime
        v = v_prime

    plt.set_axis_title("Q", 0)
    ticks = np.arange(0, nn2, nn2//5)
    n_ticks = len(ticks)
    plt.set_axis_ticks(0, [ticks, ticks])
    plt.set_axis_tick_labels(0, "xy", np.round([np.arange(x_min, x_max+1e-10, x_range/(n_ticks-1)), np.arange(v_min, v_max+1e-10, v_range/(n_ticks-1))], 2))
    plt.set_axis_labels(0, ["x", "v"])
    plt.save("mountain_car.png")
    plt.close()
    
