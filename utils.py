'''
Observation wrapper for preprocessing:

Input obs: (num_envs, 210, 160, 3)
Preprocess: Luminance(RGB) -> Resize(84, 84) -> Normalize(0, 1) -> FrameStack(4)
Luminance(R, G, B) = 0.299*R + 0.587*G + 0.114*B
Output observation: (num_envs, 84, 84, 4)
'''
import random
import pdb
from collections import deque
import matplotlib.pyplot as plt

import cv2
import torch
import numpy as np
import gymnasium as gym
from einops import rearrange

from wrappers import *

def preprocess(obs):
    # Use the luminance formula to convert RGB to grayscale
    # Convert to float32 and normalize to [0, 1]
    # Resize to 84x84
    luminance = np.array([0.299, 0.587, 0.114])
    obs = np.dot(obs, luminance)
    obs = rearrange(obs, 'n h w -> h w n').astype(np.float32) / 255.0
    # print("before : ", obs.shape)
    obs = cv2.resize(obs, (84, 110))
    # print("after : ", obs.shape)
    obs = obs[18:102,:]
    # print("after new shape: ", obs.shape)
    return obs

def render_(game, device, dqn, name, episode, framestack=4, steps=3000):
    
    # Define environment
    env = gym.make('ALE/'+game, render_mode="rgb_array")
    env = FireResetEnv(env)

    # Reset and fire
    frames = deque([], maxlen=framestack)
    env.reset()
    obs, reward, done, _, _ = env.step(1)

    size = (obs.shape[1], obs.shape[0])
    name = name + ".mp4"
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
    out.write(obs)

    obs = np.expand_dims(obs, axis=0)
    obs = preprocess(obs)

    for i in range(framestack):
        frames.append(obs)
    
    obs = rearrange(np.array(frames), 'k h w -> 1 k h w')
    all_actions = []
    for i in range(steps):
        # print(f"Step: {i} Game Lives {env.unwrapped.ale.lives()}, What env thinks : {env.lives}")
        # print(f"is really done: {env.was_real_done}")
        with torch.no_grad():
            state = torch.from_numpy(obs).float().to(device)
            q_values = dqn(state)
            actions_ = torch.argmax(q_values, 1).cpu().numpy()
        
        all_actions.append(actions_[0])
        obs, _, done, _, _ = env.step(actions_[0])
        out.write(obs)

        obs = np.expand_dims(obs, axis=0)
        obs = preprocess(obs)
        frames.append(obs)
        obs = rearrange(np.array(frames), 'k h w -> 1 k h w')
    
    out.release()
    print("Video saved ... All Actions: ", all_actions[-10:])


class VectorObservationWrapper():
    def __init__(self, 
            env,
            framestack=4,
            max_pixel=255
        ):
        # gym.Wrapper.__init__(self, env)
        self.env = env
        self.framestack = framestack
        self.num_envs = env.num_envs
        self.action_space = env.action_space
        self.frames = deque([], maxlen=framestack)
    
    def reset(self):
        obs, _ = self.env.reset()
        obs = preprocess(obs) 

        # Stack k frames: historic data for first frame is repeat(k)
        for _ in range(self.framestack):
            self.frames.append(obs)

        obs = rearrange(np.array(self.frames), 'k h w b -> b k h w')
        return obs

    def step(self, action):
        # Obs gets preprocessed and stacked
        # TODO: Check flickering issue in Atari and fix
        # Rewards go to {-1, 0, 1}
        # print(action)
        obs, reward, done, _, _ = self.env.step(action)
        # print(reward)
        obs = preprocess(obs)
        self.frames.append(obs)
        obs = rearrange(np.array(self.frames), 'k h w b -> b k h w')
        # reward = np.sign(reward)

        return obs, reward, done

def eps_greedy_policy(i, obs_, dqn, device, envs, experience):
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1e3
    random_policy = lambda: envs.action_space.sample()
    
    s = random.random()
    eps_threshold = 0.15
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     np.exp(-1. * i / EPS_DECAY)

    # if i % 100 == 0:
    #     print("### Epsilon : ", eps_threshold)

    # Follow epsilon greedy policy
    if s > eps_threshold:
        with torch.no_grad():
            state = torch.from_numpy(obs_).float().to(device)
            q_values = dqn(state)
            actions_ = torch.argmax(q_values, 1).cpu().numpy()
    else:
        actions_ = random_policy()
    
    # Store the step in the buffer
    next_obs_, rewards_, dones_ = envs.step(actions_)
    experience.add((obs_, actions_, rewards_, next_obs_, dones_))
    obs_ = next_obs_

    return rewards_, obs_

def plot_rewards(rewards, name, n=100):

    # Cumulative rewards
    n = 1000
    rewards = np.cumsum(rewards)
    rewards[n:] = (rewards[n:] - rewards[:-n])
    rewards = rewards[n - 1:]

    steps = np.arange(len(rewards))*100

    plt.clf()
    plt.plot(steps, rewards)
    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.savefig(name+"_rewards.png")