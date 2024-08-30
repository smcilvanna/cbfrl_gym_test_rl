import gymnasium as gym
import register_env  # Ensure this module registers your custom environment
from stable_baselines3 import DDPG
import time

# Create the environment
env = gym.make('cbf-value-env-v1')

# Define the DDPG model
model = DDPG(
    "MlpPolicy",  # Policy type
    env,          # Your custom environment
    verbose=1,    # Verbosity mode
    
    learning_rate=  1e-3,   # Learning rate
    gamma=          0.10,   # Discount factor
    batch_size=     100,      # Batch size for training
    
    buffer_size=1000000,  # Replay buffer size
    learning_starts=10,  # Number of steps before training starts
    
    
    tau=0.005,  # Target network update coefficient
    tensorboard_log="./ddpg_tensorboard/"  # Path to the directory where TensorBoard logs will be saved

)

# Train the model
train_steps = 40000
model.learn(total_timesteps=train_steps, log_interval=10)

# Evaluate the trained model
obs, info = env.reset()

sum_reward = 0
min_reward = 999
N = 100
for episode in range(N):  # Run for N episodes
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
        min_reward = min(min_reward,reward)

    obs, info = env.reset()

ave_reward = sum_reward / N
print("\n\nAverage Reward : ", ave_reward)
print("Min Reward : ", min_reward)

sima = time.strftime("%H:%M:%S", time.gmtime(train_steps * 5))
simb = time.strftime("%H:%M:%S", time.gmtime(train_steps * 8))

print("At 5s/sim, this would take ", sima )
print("At 8s/sim, this would take ", simb )