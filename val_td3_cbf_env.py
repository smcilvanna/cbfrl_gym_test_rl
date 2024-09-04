import gymnasium as gym
import register_env  # need this so module registers the custom environment!
from stable_baselines3 import TD3
from itertools import product
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def select_file():  # Use to select file for processing
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])  # Open file dialog and get the file path
    return file_path


if __name__ == "__main__": 
    
    filepath = select_file()               # let user select file, return full path
    model = TD3.load(filepath)  # Load the trained model

    # Validate the trained model
    # env = gym.make('cbf-value-env-v2')  # Create the environment to test the model

    obslist = np.linspace(0.1, 3.0, 30)     # Define the range of obstacle radii to test
    rl_act_list = np.zeros_like(obslist)    # Initialize the list to store the predicted RL actions
    # rl_rew_list = np.zeros_like(obslist)    # Initialize the list to store the RL rewards with predicted actions        
    opt_act_list = np.zeros_like(obslist)   # Initialize the list to store the optimal actions
    
    for i, _ in enumerate(obslist):
        obs_test = np.array([obslist[i]])  # Define the first obstacle radius and action
        rl_act_list[i] = model.predict(obs_test, deterministic=True)[0]  # Extract the predicted action from the sequence
        opt_act_list[i] = 0.05*(obs_test-5)**2

plt.figure()
plt.plot(obslist, rl_act_list, label='RL Action')
plt.plot(obslist, opt_act_list, label='Optimal Action')
plt.legend()
plt.xlabel('Obstacle Size')
plt.ylabel('Action')
plt.title('RL-Predicted vs Optimal Action')
plt.show()
    
  