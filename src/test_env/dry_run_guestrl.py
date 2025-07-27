# Dry-run script for GuestRL environment
# Save this as, e.g., `dry_run_guestrl.py` and run with `python dry_run_guestrl.py`
import sys
import os
project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from env.env_gym import GuestEnv
import numpy as np

def main():
    # Instantiate the environment
    env = GuestEnv()
    
    # Reset to get initial observation and info
    obs, info = env.reset()
    print("=== After reset ===")
    print("Initial Observation:")
    print(obs)
    print("Initial Info:")
    print(info)
    print()
    
    # Take a few random steps to see how the environment evolves
    num_steps = 10
    for step in range(num_steps):
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Apply the action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Print step details
        print(f"--- Step {step+1} ---")
        print("Action:", action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        print()
        
        if done:
            print("Episode finished after step", step+1)
            break

if __name__ == "__main__":
    main()
