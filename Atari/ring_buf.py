import random

class RingBuf:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        self.num_sampled = 0
        
    def add(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def expand(self, count):
        self.end = len(self.data) - 1
        self.data += [None] * (count)
        self.start = 0
    
    def sample(self, count, rnd=True):
        self.num_sampled += count
        if rnd:
            return random.choices(self.data[:self.__len__()], k=count)
        else:
            return self.data[:self.__len__()][self.num_sampled - count:self.num_sampled] + self.data[:max(0, self.num_sampled - self.__len__())]
    
    def get_num_sampled(self):
        return self.num_sampled

    def reset(self):
        self.num_sampled = 0
        
    def get_last(self):
        return self.data[(self.end - 1) % len(self.data)]
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
