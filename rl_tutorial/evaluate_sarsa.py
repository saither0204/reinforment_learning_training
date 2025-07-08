import gymnasium as gym
import numpy as np
import pickle
import glob

q_table_files = sorted(glob.glob("sarsa_q_table_alpha_*.pkl"))
max_steps = 100  # Set a reasonable max step limit

for q_file in q_table_files:
    with open(q_file, "rb") as f:
        Q = pickle.load(f)
    env = gym.make("CliffWalking-v1", render_mode="human")
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done and steps < max_steps:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        print(f"{q_file} | Step: {steps} | State: {state} | Reward: {reward}")
    env.close()
    if steps >= max_steps:
        print(f"Skipped {q_file}: agent got stuck (>{max_steps} steps)")
        continue
    print(f"Evaluation for {q_file}: Total reward = {total_reward}")