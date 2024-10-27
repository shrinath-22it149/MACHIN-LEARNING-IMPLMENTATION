import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 5
num_episodes = 500
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

# Action space: 0 - Up, 1 - Down, 2 - Left, 3 - Right
actions = [0, 1, 2, 3]
q_table = np.zeros((grid_size, grid_size, len(actions)))

# Rewards setup
goal = (4, 4)
obstacles = [(1, 1), (2, 1), (3, 3)]
rewards = np.zeros((grid_size, grid_size))
rewards[goal] = 1  # Reward for reaching the goal
for obs in obstacles:
    rewards[obs] = -1  # Penalty for hitting obstacles

def get_state(position):
    return position[0], position[1]

# Training the agent
for episode in range(num_episodes):
    position = (0, 0)  # Start position
    total_reward = 0

    while position != goal:
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)  # Explore
        else:
            action = np.argmax(q_table[position[0], position[1]])  # Exploit

        # Move based on the action
        if action == 0 and position[0] > 0:  # Up
            new_position = (position[0] - 1, position[1])
        elif action == 1 and position[0] < grid_size - 1:  # Down
            new_position = (position[0] + 1, position[1])
        elif action == 2 and position[1] > 0:  # Left
            new_position = (position[0], position[1] - 1)
        elif action == 3 and position[1] < grid_size - 1:  # Right
            new_position = (position[0], position[1] + 1)
        else:
            new_position = position  # Stay in place if out of bounds

        # Update reward and Q-value
        reward = rewards[new_position]
        total_reward += reward
        q_table[position[0], position[1], action] += learning_rate * (
            reward + discount_factor * np.max(q_table[new_position[0], new_position[1]]) - 
            q_table[position[0], position[1], action]
        )
        position = new_position  # Move to new position

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training completed.")

# Visualization of learned Q-values
fig, axs = plt.subplots(1, len(actions), figsize=(15, 5))

for i in range(len(actions)):
    axs[i].imshow(q_table[:, :, i], cmap='hot', interpolation='nearest')
    axs[i].set_title(f'Q-values for Action {i}')
    axs[i].set_xticks(np.arange(grid_size))
    axs[i].set_yticks(np.arange(grid_size))
    axs[i].set_xticklabels(np.arange(grid_size))
    axs[i].set_yticklabels(np.arange(grid_size))
    axs[i].grid(False)

plt.suptitle('Q-values for Each Action in the Grid Environment')
plt.tight_layout()
plt.colorbar(axs[0].imshow(q_table[:, :, 0], cmap='hot', interpolation='nearest'), ax=axs, orientation='vertical', label='Q-value')
plt.show()
