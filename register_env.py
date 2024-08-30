from gymnasium.envs.registration import register

# Import your custom environment class
from envs.cbf_env import CustomEnv  # Adjust the import path accordingly

# Register the custom environment with Gymnasium
register(
    id='cbf-value-env-v1',  # Unique identifier for the environment
    entry_point='envs.cbf_env:CustomEnv',  # Corrected path
    max_episode_steps=1,  # Maximum number of steps per episode
)