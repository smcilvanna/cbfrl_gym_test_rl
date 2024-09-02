import gymnasium as gym
import register_env  # need this so module registers the custom environment!
from stable_baselines3 import TD3
from itertools import product
import time
import numpy as np
import os


if __name__ == "__main__": 
    
    model = TD3.load("./td3_best.zip")  # Load the trained model

    # Validate the trained model
    env = gym.make('cbf-value-env-v1')  # Create the environment to test the model

    # Evaluate the trained model
    obs, info = env.reset()
    sum_reward = 0
    min_reward = 999
    val_steps = 100

    for episode in range(val_steps):  # Run for N episodes
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            env.a = action
            env.r = reward
            env.render()  # Render the environment (to console)
            done = terminated or truncated  # Check if the episode is finished
            sum_reward += reward
            min_reward = min(min_reward, reward)
        obs, info = env.reset()
    ave_reward = sum_reward / val_steps
    print(f"Average reward: {ave_reward}, Minimum reward: {min_reward}")