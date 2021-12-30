import numpy as np

class DeepMimicStyleBuffer:
    """
    This buffer is more or less a copy of DeepMimic's ReplayBufferRandStorage and is meant for internal use by the
    AMP discriminator in an AMPReplayBuffer.
    """
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.curr_size = 0
        self.total_count = 0
        self.buffer = None

        self.clear()
        return

    def sample(self, n):
        curr_size = self.get_current_size()
        idx = np.random.randint(0, curr_size, size=n)
        return idx

    def get(self, idx):
        return self.buffer[idx]

    def store(self, data):
        n = len(data)
        if n > 0:
            if self.buffer is None:
                self.init_buffer(data)

            idx = self.request_idx(n)
            self.buffer[idx] = data
                
            self.curr_size = min(self.curr_size + n, self.buffer_size)
            self.total_count += n
        return

    def is_full(self):
        return self.curr_size >= self.buffer_size

    def clear(self):
        self.curr_size = 0
        self.total_count = 0
        return
    
    def get_buffer_size(self):
        return self.buffer_size
    
    def get_current_size(self):
        return self.curr_size

    def init_buffer(self, data):
        dtype = data[0].dtype
        shape = [self.buffer_size] + list(data[0].shape)
        self.buffer = np.zeros(shape, dtype=dtype)
        return

    def request_idx(self, n):
        curr_size = self.get_current_size()

        idx = []
        if not self.is_full():
            start_idx = curr_size
            end_idx = min(self.buffer_size, start_idx + n)
            idx = list(range(start_idx, end_idx))

        remainder = n - len(idx)
        if remainder > 0:
            rand_idx = list(np.random.choice(curr_size, remainder, replace=False))
            idx += rand_idx

        return idx