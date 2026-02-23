import numpy as np
import numpy.random as random
import torch
import time

from tqdm import tqdm
import argparse
import os
import pathlib
import pickle
import gymnasium as gym
import ale_py
import sys
import cv2

from model import Model
from ring_buf import RingBuf

# Render a gameplay once per given number of iterations
RENDER_INTERVAL = 10

# Save checkpoint and memory after given number of iterations
SAVE_INTERVAL = 100

# Maximum number of games to train on in one go (application will exit after that)
MAX_GAMES = 4000

# Defines the number of iterations during which exploration/exploitation
# ratio is linearly reduced from 1 to 0.1
NUM_EXPLORE = 1_000_000

# Number of consecutive frames to use in the model to capture kinematics
N_FRAMES = 4

# The original frame is downsampled to this resolution (nonconfigurable)
NUM_X = 52
NUM_Y = 40

# Adjust these properties based on available RAM.
# The higher the MEMORY_SIZE*NUM_MEMORIES the less model forgets
# already seen states
MEMORY_SIZE = 1_000_000 # Size of memory kept in RAM at once
NUM_MEMORIES = 1 # Total number of memories kept on disc

# Number of states fed into the model during single optimization step.
# More precisely this is the number of randomly selected past states + 1 (current state)
BATCH_SIZE = 32

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def to_float(byte_array):
    return np.asarray(byte_array, dtype=np.float32) / 255


def downsample1(img):
    return img[::2, ::2]


def downsample2(img):
    return np.max((
            img[:-1:2, 1::2], 
            img[1::2, 1::2], 
            img[:-1:2, :-1:2], 
            img[1::2, :-1:2]), axis=0)


def preprocess(img):
    return downsample2(downsample1(to_grayscale(img)))


def load_memory(active_memory):
    mem_file = 'memory' + str(active_memory) + '.pkl'
    if os.path.isfile(mem_file):
        return pickle.load(open(mem_file, 'rb'))
    else:
        return None


def save_memory(memory, active_memory):
    with open('memory' + str(active_memory) + '.pkl', 'wb') as f:
        pickle.dump(memory, f)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Breakout:

    def __init__(self, checkpoint=None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        pathlib.Path('trained/').mkdir(parents=True, exist_ok=True)
        
        current_time = time.strftime("%Y-%m-%d-%H-%M")
        self.out_name = f'trained/{current_time}'

        self.env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        self.active_memory = 0
        self.iteration = 0
        self.game_index = 0

        self.loss = []
        self.epsilon = 1.0

        self.model = Model(n_frames=N_FRAMES, n_actions=self.env.action_space.n, nx=NUM_X, ny=NUM_Y)
        self.model.to(self.device)

        if checkpoint is not None:
            self.checkpoint = f'{checkpoint}.pth'
            print(f"Loading checkpoint '{self.checkpoint}'")
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
            self.iteration = checkpoint['iteration']
            self.game_index = checkpoint['game_index']
            if 'active_memory' in checkpoint:
                self.active_memory =  checkpoint['active_memory']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print("Checkpoint loaded")
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000)


    def update_epsilon(self):
        if self.iteration < NUM_EXPLORE:
            self.epsilon = 1.0 - 0.9 * self.iteration / NUM_EXPLORE
        else:
            self.epsilon = 0.1
        return self.epsilon


    def init_state(self):
        self.update_epsilon()
        frame1, info = self.env.reset()
        state = [preprocess(frame1)]
        for _ in range(N_FRAMES - 1):
            action = self.env.action_space.sample()
            frame, reward, _, _, info = self.env.step(action)
            state.append(preprocess(frame))
        return state


    def init_memory(self):
        self.memory = RingBuf(MEMORY_SIZE)
        state = self.init_state()
        for _ in tqdm(range(MEMORY_SIZE//10), desc=f"Initializing memory {self.active_memory}"):
            action = self.env.action_space.sample()
            state, _, _ = self.step(state, action)


    def step(self, state, action):
        action_mask = np.zeros(self.env.action_space.n)
        action_mask[action] = 1

        new_frame, reward, is_done, _, info = self.env.step(action)
        new_state = preprocess(new_frame)
        mem_entry = [state, action_mask, np.sign(reward), new_state, is_done]
        self.memory.add(mem_entry)
        if is_done:
            return self.init_state(), mem_entry, is_done
        return state[1:] + [new_state], mem_entry, is_done


    def render(self, wait = 33):
        frame = self.env.render()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (400, 500), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Breakout", frame)
            cv2.waitKey(wait)


    def choose_best_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(np.asarray(state)).float().to(self.device).unsqueeze(0)
            next_q_values = self.model(state)
            next_q_values = next_q_values.detach().cpu().numpy()
            if np.all(next_q_values == 0):
                return random.choice(self.env.action_space.n)
            return np.argmax(next_q_values)


    def test(self):
        self.model.eval()
        is_done = False
        state = self.init_state()

        while not is_done:
            if random.random() < 0.1:
                action = self.env.action_space.sample()
            else:
                action = self.choose_best_action(to_float(state))
            frame, reward, is_done, _, info = self.env.step(action)
            
            state = state[1:] + [preprocess(frame)]
            self.render()


    def train(self):
        new_memory = load_memory(self.active_memory)
        if new_memory is not None:
            self.memory = new_memory
            self.memory.reset()
        else:
            for i in range(NUM_MEMORIES):
                self.init_memory()
                save_memory(self.memory, self.active_memory)
                self.active_memory += 1
            self.active_memory = 0
            new_memory = load_memory(self.active_memory)
            if new_memory is not None:
                self.memory = new_memory
                self.memory.reset()

        self.loss = []
        print(f"Active memory {self.active_memory}")
        print(f"Iteration {self.iteration}")
        train_log = open(f'{self.out_name}.loss.csv', 'w')

        for game_index in range(self.game_index, self.game_index + MAX_GAMES):
            t_start = time.perf_counter()
            self.train_one_game(game_index)
            t_end = time.perf_counter()
            print(f"Game {game_index} took {t_end - t_start:0.4f} seconds. Epsilon: {self.epsilon}")

            #self.scheduler.step()

            train_log.write(f'{game_index},{self.loss[-1]}\n')
            train_log.flush()

            if (game_index + 1) % SAVE_INTERVAL == 0:
                print(f"Iteration {self.iteration}")
                print(f"Average loss: {np.mean(self.loss[-100:])}")
                save_checkpoint({
                    'iteration': self.iteration,
                    'game_index': self.game_index,
                    'active_memory': self.active_memory,
                    'state_dict': self.model.state_dict(),
                    'epsilon': self.epsilon,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }, filename=f'{self.out_name}.pth')

                save_memory(self.memory, self.active_memory)
                if self.memory.get_num_sampled() >= MEMORY_SIZE:
                    self.active_memory += 1
                    self.active_memory %= NUM_MEMORIES
                    new_memory = load_memory(self.active_memory)
                    if new_memory is not None:
                        self.memory = new_memory
                        self.memory.reset()
                    else:
                        self.init_memory()
            if (game_index + 1) % MAX_GAMES == 0:
                sys.exit()

        train_log.close()


    def train_one_game(self, game_index):
        self.model.train()

        loss_avg = 0.0
        n = 1

        is_done = False
        state = self.init_state()

        while not is_done:
            self.optimizer.zero_grad()
            loss, is_done, state = self.q_iteration(state)

            loss_avg += (loss.item() - loss_avg) / n
            n += 1

            loss.backward()
            self.optimizer.step()
            
            self.iteration += 1

            if (game_index + 1) % RENDER_INTERVAL == 0:
                self.render(wait=1)
            else:
                cv2.destroyAllWindows()

        self.loss.append(loss_avg)


    def q_iteration(self, state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            self.model.eval()
            action = self.choose_best_action(to_float(state))
            self.model.train()
        state, mem_entry, is_done = self.step(state, action)

        mem_batch = self.memory.sample(BATCH_SIZE - 1)
        mem_batch = [mem_entry] + mem_batch
        states = np.empty((BATCH_SIZE, N_FRAMES, NUM_X, NUM_Y), dtype=np.float32)
        action_masks = [None] * BATCH_SIZE
        rewards = [None] * BATCH_SIZE
        next_states = np.empty((BATCH_SIZE, N_FRAMES, NUM_X, NUM_Y), dtype=np.float32)

        for i in range(BATCH_SIZE):
            state_i = mem_batch[i][0]
            states[i] = to_float(state_i)
            action_masks[i] = mem_batch[i][1]
            rewards[i] = mem_batch[i][2]
            next_states[i] = to_float(state_i[1:] + [mem_batch[i][3]])

        loss = self.calc_loss(
                  states, 
                  np.asarray(action_masks), 
                  np.asarray(rewards), 
                  next_states)
        return loss, is_done, state


    def calc_loss(self, start_states, actions, rewards, next_states):
        actions = torch.from_numpy(actions).float().to(self.device)
        start_states = torch.from_numpy(start_states).float().to(self.device)
        pred_q_values = self.model(start_states)
        pred_q_values = torch.mul(pred_q_values, actions)

        self.model.eval()
        with torch.no_grad():
            next_q_values = self.model(torch.from_numpy(next_states).float().to(self.device))
        self.model.train()

        gamma = 0.99

        q_values = torch.max(next_q_values, dim=1).values*gamma + torch.from_numpy(rewards).float().to(self.device)

        loss = torch.sum((actions * q_values[:, None] - pred_q_values)**2)
    
        return loss


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    breakout = Breakout(checkpoint=args.checkpoint)
    if args.test:
        breakout.test()
    else:
        breakout.train()

