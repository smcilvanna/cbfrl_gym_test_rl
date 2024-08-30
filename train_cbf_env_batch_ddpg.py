import gymnasium as gym
import register_env  # need this so module registers the custom environment!
from stable_baselines3 import DDPG
from itertools import product
import time
import numpy as np
import os

def train_and_evaluate_ddpg(learning_rate=1e-3, gamma=0.99, batch_size=100, train_steps=20000, val_steps=100):
    # Create the environment
    env = gym.make('cbf-value-env-v1')

    # Define the DDPG model
    model = DDPG(
        "MlpPolicy",  # Policy type
        env,          # Your custom environment
        verbose=1,    # Verbosity mode
        learning_rate=learning_rate,   # Learning rate passed as an argument
        gamma=gamma,                   # Discount factor passed as an argument
        batch_size=batch_size,                # Batch size for training
        buffer_size=1000000,           # Replay buffer size
        learning_starts=10,            # Number of steps before training starts
        #tensorboard_log="./ddpg_tensorboard/"  # Path to the directory where TensorBoard logs will be saved, uncomment for logging
        tau=0.005                     # Target network update coefficient
    )

    # Train the model
    #model.learn(total_timesteps=train_steps, log_interval=10)  # with tensorboard logging
    model.learn(total_timesteps=train_steps)                    # without tensorboard logging

    # Evaluate the trained model
    obs, info = env.reset()
    sum_reward = 0
    min_reward = 999

    for episode in range(val_steps):  # Run for N episodes
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.a = action
            env.r = reward
            env.render()  # Optionally render the environment
            # Check if the episode is finished
            done = terminated or truncated
            sum_reward += reward
            min_reward = min(min_reward, reward)
        obs, info = env.reset()
    ave_reward = sum_reward / val_steps    
    return ave_reward, min_reward

if __name__ == "__main__": # batch testing of hyperparameters

    outfile = 'v2_results40k.csv'     # name of output file for results
    if os.path.exists(outfile): # do a check if the results file already exists
        raise FileExistsError(f"The file '{outfile}' already exists. Choose a different filename or delete the existing file.")

    learnrate_set = [1e-3 , 1e-4, 1e-5]     # define parameters to test
    gamma_set     = [0.99 , 0.5 , 0.1 ]
    batch_set     = [1 , 100 ]
    test_set      = list(product(learnrate_set, gamma_set, batch_set))  # create test schedule with all cominations
    n_tests       = len(test_set)
    results       = np.zeros((n_tests,5))   # empty array for results
    # results [columns] = 0.learnrate 1.gamma 2.batch_size 3.ave reward 4.min reward

    for i in range(n_tests):    # loop through test schedule and run each test set and record results
        print("Starting test with RL params", test_set[i])

        steps = 40000
        ar, mr = train_and_evaluate_ddpg(test_set[i][0], test_set[i][1], test_set[i][2],train_steps=steps,val_steps=100)
        results[i,0] = test_set[i][0]
        results[i,1] = test_set[i][1]
        results[i,2] = test_set[i][2]
        results[i,3] = ar
        results[i,4] = mr
        print("Test ", i, " / ",n_tests," completed.")
        np.savetxt(outfile, results, delimiter=",")