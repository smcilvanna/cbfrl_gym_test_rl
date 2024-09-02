import gymnasium as gym
import register_env  # need this so module registers the custom environment!
from stable_baselines3 import TD3
from itertools import product
import time
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__": 
    
    model = TD3.load("./td3_best.zip")  # Load the trained model

    # Validate the trained model
    env = gym.make('cbf-value-env-v2')  # Create the environment to test the model

    obslist = np.linspace(0.1, 3.0, 30)     # Define the range of obstacle radii to test
    rl_act_list = np.zeros_like(obslist)    # Initialize the list to store the predicted RL actions
    rl_rew_list = np.zeros_like(obslist)    # Initialize the list to store the RL rewards with predicted actions        

    for i,o in enumerate(obslist):
        env.reset(orad=o)                                       # Reset the environment
        a = model.predict(env.observation, deterministic=True)  # Predict the action
        X = env.step(action=a[0])                               # Take the action and get the reward
        rl_act_list[i] = a[0]                                   # Store the predicted action                       
        rl_rew_list[i] = X[1]                                   # Store the reward with the predicted action

    for i,_ in enumerate(obslist):
        print(f"Obstacle Radius: {round(obslist[i],3)}, RL Action: {rl_act_list[i]}, RL Reward: {rl_rew_list[i]}")

    plt.figure()

    # Plot RL Action vs Obstacle Radius
    plt.subplot(2, 1, 1)
    plt.plot(obslist, rl_act_list, label='RL Action')
    plt.title('RL Action vs Obstacle Radius')

    # Plot RL Reward vs Obstacle Radius
    plt.subplot(2, 1, 2)
    plt.plot(obslist, rl_rew_list, label='RL Reward')
    plt.title('RL Action Reward vs Obstacle Radius')

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.suptitle("Validation Of Trained Model", fontweight='bold', fontsize=14)  # Add a title to the figure with bold and larger font
    plt.show()

    
    # for orad in obslist:
    #     print(orad)
    #     # Evaluate the trained model
    #     env.reset(oradius=orad)  # Reset the environment
    #     N = 20
    #     for episode in range(N):  # Run for N episodes
    #         done = False
    #         while not done:
    #             action, _states = model.predict(orad, deterministic=False)
    #             print(action)
    #             thisobs = np.array([orad, action[0]])
    #             cbfbest.append(thisobs)
    
    # np.savetxt('predicted_cbf.csv', cbfbest, delimiter=",")
