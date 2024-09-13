import torch
import numpy as np

'''
Vanilla implementation of Experience Replay Memory
'''

class ReplayBuffer:
    def __init__(self, 
            device, 
            framestack=4,
            batch_size=100,
            num_envs=25,
            capacity=50000):
        
        assert batch_size % num_envs == 0, \
            "Batch size should be divisible by number of environments"
        
        self.device = device
        self.capacity = capacity
        self.framestack = framestack
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.pointer = 0
        
        # Define circular buffers for each variable
        self.states = np.zeros((capacity, framestack, 84, 84), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.int32)
        self.next_states = np.zeros((capacity, framestack, 84, 84), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def add(self, step_data):
        # Optimize storage
        state, action, reward, next_state, done = step_data
        i = self.pointer % self.capacity
        j = (self.pointer + self.num_envs) % (self.capacity + 1)
        
        self.states[i:j] = state
        self.actions[i:j] = action
        self.rewards[i:j] = reward
        self.next_states[i:j] = next_state
        self.dones[i:j] = done

        if j == self.capacity:
            self.pointer = 0
        else:
            self.pointer = j

    def sample(self):
        # Uniform random sampling
        idx = np.random.choice(
            len(self.states), self.batch_size, replace=False)
        
        states = torch.from_numpy(self.states[idx]).to(self.device)
        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx]).to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.states)
'''
DQN implementation from: https://www.nature.com/articles/nature14236
'''

class DQN(torch.nn.Module):
    def __init__(
            self, 
            in_channels, 
            num_actions
        ):
        super(DQN, self).__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 512)
        self.fc2 = torch.nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x