import gymnasium as gym
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

alphas = [0.05, 0.1, 0.2, 0.5]
num_episodes = 500
results = {}

for ALPHA in alphas:
    q_table = np.zeros((48, 4))
    EPSILON = 1.0
    MIN_EPSILON = 0.05
    EPSILON_DECAY = 0.995
    GAMMA = 0.9
    rewards = []
    env = gym.make("CliffWalking-v1")
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        action = np.random.randint(0, 4)
        total_reward = 0
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = np.random.randint(0, 4) if np.random.rand() < EPSILON else int(np.argmax(q_table[next_state]))
            q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])
            state, action = next_state, next_action
            total_reward += reward
            done = terminated or truncated
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        rewards.append(total_reward)
        print(f"ALPHA={ALPHA} | Episode {episode+1}/{num_episodes} | Total Reward: {total_reward} | EPSILON: {EPSILON:.3f}")
    env.close()
    # Save Q-table for this ALPHA
    with open(f"sarsa_q_table_alpha_{ALPHA}.pkl", "wb") as f:
        pkl.dump(q_table, f)
    results[ALPHA] = np.convolve(rewards, np.ones(20)/20, mode='valid')  # Moving average

for ALPHA, avg_rewards in results.items():
    plt.plot(avg_rewards, label=f'ALPHA={ALPHA}')
plt.xlabel('Episode')
plt.ylabel('Average Reward (20-episode MA)')
plt.title('SARSA: Effect of ALPHA on Reward')
plt.legend()
plt.show()