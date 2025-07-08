import gymnasium as gym
import numpy as np

cliff_env = gym.make("CliffWalking-v1", render_mode="human")

done = False
state, _ = cliff_env.reset()

while not done:
    action = np.random.randint(low=0, high=4)
    print(f"State: {state} --> Action: {['up','right','down','left'][action]}")
    state, reward, terminated, truncated, _ = cliff_env.step(action)
    done = terminated or truncated

cliff_env.close()