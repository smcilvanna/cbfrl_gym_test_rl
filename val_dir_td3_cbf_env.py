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

def select_folder():  # Use to select folder for processing
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(initialdir='./outputs/td3/optimised_hps_a/')  # Open folder dialog and set the initial directory to the 'outputs' folder
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.zip')]
    info = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    return folder_path, file_list, info


if __name__ == "__main__": 

    tofile = True  # Set to True to save the plot to file, false to dispaly to screen
    
    pth, files, info = select_folder()              # let user select file, return full path
    print(info)
    if len(info) > 0:
        infopath = os.path.join(pth, info[0])           # Get the info file path
        info = np.genfromtxt(infopath, delimiter=',')   # Read the info file to np array

    for file in files:
        fullpath = os.path.join(pth, file)
        print(file)
        i = file.split('_')[0]
        i = int(i)-1
        print(i)

        if len(info) > 0:
            lrn_rate=   str(info[i][0])
            dis_ftr=    str(info[i][1])
            b_size=     str(info[i][2])
            buf_siz=    '1000'
            net_arc=    '2x64 node mlps'
            plttxt= 'LR: ' + lrn_rate + '\nDF: ' + dis_ftr + '\nBS: ' + b_size + '\nBUF: ' + buf_siz + '\nNET: ' + net_arc
        else:
            plttxt = 'No Info File'

        model = TD3.load(fullpath)  # Load the trained model
        obslist = np.linspace(0.1, 3.0, 30)     # Define the range of obstacle radii to test
        rl_act_list = np.zeros_like(obslist)    # Initialize the list to store the predicted RL actions    
        opt_act_list = np.zeros_like(obslist)   # Initialize the list to store the optimal actions
    
        for i, _ in enumerate(obslist):
            obs_test = np.array([obslist[i]])  # Define the first obstacle radius and action
            rl_act_list[i] = model.predict(obs_test, deterministic=True)[0]  # Extract the predicted action from the sequence
            opt_act_list[i] = 0.05*(obs_test-5)**2

        plt.figure(figsize=(16, 9))
        plt.plot(obslist, rl_act_list, label='RL Action')
        plt.plot(obslist, opt_act_list, label='Optimal Action')
        plt.text(0.1, 0.05, plttxt, fontsize=10, ha='left', va='bottom', wrap=True)
        plt.legend()
        plt.xlabel('Obstacle Size')
        plt.ylabel('Action')
        plt.suptitle('RL-Predicted vs Optimal Action')
        plt.title(file)
        plt.ylim(0, 1.2)
        plt.xlim(0, 3)
        plt.axis('scaled')

        if tofile:
            plt.savefig('./outputs/tmp/' + file + ".png", bbox_inches='tight', dpi=300)
            print('Saved: ' + file)
        else:
            plt.show()
    
  