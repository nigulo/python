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

n = 8
d = 2*n + 3

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
    if x <= x_min:
        return [((x, 0), -1, 1)]
    
    x1 = max(x_min, min(x_max, x + v))
    v1 = max(v_min, min(v_max, v + 0.001*a - 0.0025*np.cos(3*x)))
        
    return [((x1, v1), -1, 1)]

def s0(_):
    return (0.2*random.random() - 0.6, 0)

def get_features(x, v, a):
    x_ind = int(round((n-1)*(x - x_min) / (x_max - x_min)))
    v_ind = int(round((n-1)*(v - v_min) / (v_max - v_min)))
    a_ind = a + 1
    f_x = np.zeros(n)
    f_v = np.zeros(n)
    f_a = np.zeros(3)
    f_x[x_ind] = 1
    f_v[v_ind] = 1
    f_a[a_ind] = 1
    
    return np.concatenate((f_x, f_v, f_a)) 

def q(s, a, w):
    x, v = s
    features = get_features(x, v, a)
    return np.sum(features*w)

def q_grad(s, a, w):
    x, v = s
    features = get_features(x, v, a)
    return features
    
if __name__ == '__main__':
    print("Training...")
    td = TD(actions, transitions, d, q, q_grad)
    w, pi = td.train(s0, n_episodes=10)
    
    q_max = np.empty((n, n))
    for i in range(n):
        x = x_min + (x_max - x_min)*i/n
        for j in range(n):
            v = v_min + (v_max - v_min)*j/n
            q_s_a = [q((x, v), a, w) for a in actions((x, v))]
            q_max[i, j] = q_s_a[np.argmax(q_s_a)]
    
    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(n*50, n*50))
    plt.colormap(q_max.T[::-1,::], 0, show_colorbar=True)
    
    plt.set_axis_title("Q", 0)
    plt.save("mountain_car.png")
    plt.close()
    
