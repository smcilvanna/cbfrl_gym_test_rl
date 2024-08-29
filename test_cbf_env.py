import register_env
import gymnasium as gym
from time import sleep


env = gym.make('cbf-value-env-v1')
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Example action selection (random)
    next_state, reward, terminated, truncated, info = env.step(action)
    env.a = action
    env.r = reward
    env.render()  # Optionally render the environment
    sleep(1)