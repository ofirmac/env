import sys
import os
project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from guest_env.env_gym import GuestEnv
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
    
    # Since action_space.sample() may not be implemented, pick actions manually
    # ACTIONS = {0: 'wait', 1: 'stop', 2: 'stare_at(0)', 3: 'stare_at(1)', 4: 'stare_at(2)',
    #            5: 'encourage(0)', 6: 'encourage(1)', 7: 'encourage(2)'}
    num_steps = 10
    for step in range(num_steps):
        # Rotate through each action for demonstration
        action = step % env.action_space.n
        
        # Apply the action
        result = env.step(action)
        # Gymnasium v0.27+ returns (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        # Print step details
        print(f"--- Step {step+1} ---")
        print("Action index:", action)
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
