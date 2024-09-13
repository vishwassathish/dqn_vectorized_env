'''
Vishwas Sathish
https://vishwassathish.github.io/

Simple DQN implementation for Atari's discrete action space
Algorithm from: https://www.nature.com/articles/nature14236

Code references:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://github.com/KaleabTessera/DQN-Atari/tree/master
https://github.com/diegoalejogm/deep-q-learning/tree/master
https://github.com/jzhanson/breakout-demo/tree/master
'''

import sys
import time
import torch
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from dqn import ReplayBuffer, DQN
from wrappers import *
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def env_fn(game):
    env = gym.make('ALE/'+game, render_mode="rgb_array")
    env = FireResetEnv(EpisodicLifeEnv(NoopResetEnv(env)))
    return env

if __name__ == "__main__":
    # Hyperparameters
    file = './files/'
    game = 'Pong-v5'
    name = file + game
    num_envs = 25
    num_actions = 6 # Number of actions in Breakout
    framestack = 4 # Number of frames to stack
    capacity = 100_000 # DQN paper has 1M buffer size
    batch_size = 100
    episodes = 1000
    steps_per_episode = int(1e4)
    DISCOUNT = 0.99

    # Define environments
    # env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
    # env = FireResetEnv(EpisodicLifeEnv(NoopResetEnv(env)))
    envs = [lambda: env_fn(game) for _ in range(num_envs)]
    envs = gym.vector.AsyncVectorEnv(envs)
    envs = VectorObservationWrapper(envs, framestack=framestack)

    # 1. Define empty buffer and model
    experience = ReplayBuffer(device, framestack, batch_size, 
                              num_envs, capacity)
    random_policy = lambda: envs.action_space.sample()
    dqn = DQN(framestack, num_actions).to(device)
    try:
        ckpt = torch.load(name+"_model.pth")
    except:
        print("Checkpoint does not exist ... ")
        ckpt = None

    if ckpt is not None:
        dqn.load_state_dict(ckpt["model"])
        print("Model loaded ... ")
    else:
        print("Model not found ... Starting from scratch")
    
    optimizer = torch.optim.AdamW(dqn.parameters(), lr=1e-4, amsgrad=True)
    if ckpt is not None and "optimizer" in ckpt.keys():
        optimizer.load_state_dict(ckpt["optimizer"])

    target_dqn = DQN(framestack, num_actions).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    
    loss_fn = torch.nn.SmoothL1Loss()

    if ckpt is not None and "episode" in ckpt.keys():
        step_rewards = pickle.load(open(name+"_rewards.pkl", "rb"))
        start = ckpt["episode"] + 1
        print("Rewards loaded ... ")
    else:
        step_rewards = []
        start = 0

    # 2. Fill the empty buffer to have at least one batch
    obs = envs.reset()
    print("Filling buffer ... ")
    counter = 0
    for i in range(batch_size*2):
        action = random_policy()
        next_obs, reward, done = envs.step(action)
        experience.add((obs, action, reward, next_obs, done))
        obs = next_obs

    # 3. Init Model params and run the DQN algorithm

    # Training loop
    total_time = time.time()
    for ep in range(start, episodes):
        epoch_start_time = time.time()
        render_(game, device, dqn, name, ep)
        temp_obs = envs.reset()
        
        for i in range(steps_per_episode):
            step_time = time.time()

            # Sample an action from policy and store in buffer
            temp_rewards, temp_obs = eps_greedy_policy(i, temp_obs, dqn, 
                                    device, envs, experience)
            
            # Plot rewards periodically
            if i % 100 == 0:
                step_rewards.append(np.mean(temp_rewards))
                print(
                        "Episode : ", ep, \
                        " | Step : ", i, \
                        " | Total steps : ", (ep * steps_per_episode) + i, \
                        " | Rewards : ", step_rewards[-3:-1]
                    )
                plot_rewards(step_rewards, name)

            # Sample batch from experience
            states, actions, rewards, next_states, dones = experience.sample()

            # Compute Q-values Q(s_t, a_t) and target-values V(s_{t+1})
            q_values = dqn(states)
            q_values = q_values.gather(1, actions.unsqueeze(1))

            # The Q-values for the next state are masked out for games with 
            # multiple lives. This is to prevent the agent from learning high
            # variance Q-value estimates.

            with torch.no_grad():
                target_q_values = target_dqn(next_states)
                target_values = torch.max(target_q_values, dim=1)[0] * (1 - dones.float())
                target_values = rewards + DISCOUNT * target_values 
                target_values = target_values.unsqueeze(1)

            # Compute loss
            loss = loss_fn(q_values, target_values)
            loss = torch.clamp(loss, -1, 1)
            
            
            # Optimize
            dqn.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(dqn.parameters(), 100)
            optimizer.step()

            # Update target network
            if i % 100 == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            if i % 100 == 0:
                print("Loss : ", loss.item())
                print("Step time : ", time.time() - step_time)
        
        torch.save(
            {
                "model": dqn.state_dict(), 
                "optimizer": optimizer.state_dict(),
                "episode": ep
            },  
            name+"_model.pth"
        )
        pickle.dump(step_rewards, open(name+"_rewards.pkl", "wb"))
        print("Model saved ... ")
        
        print("Epoch time : ", time.time() - epoch_start_time)
    
    print("Total time : ", time.time() - total_time)
        