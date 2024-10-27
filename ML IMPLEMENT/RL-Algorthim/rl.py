import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_episodes = 20000  # Increased number of episodes
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.9995  # Slower decay to encourage exploration
epsilon_min = 0.01

# Action space: 0-8 (positions on the board)
q_table = np.random.rand(3**9, 9) * 0.01  # Small random values to start

def state_to_index(state):
    """Convert board state to a unique index."""
    index = 0
    for i in range(9):
        index += (3**i) * state[i]
    return index

def check_winner(state):
    """Check if there is a winner."""
    winning_positions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                         (0, 3, 6), (1, 4, 7), (2, 5, 8),
                         (0, 4, 8), (2, 4, 6)]
    for pos in winning_positions:
        if state[pos[0]] == state[pos[1]] == state[pos[2]] != 0:
            return state[pos[0]]
    return 0 if 0 in state else -1  # Return 0 for ongoing, -1 for draw

# Performance tracking
win_counts = np.zeros(num_episodes)
draw_counts = np.zeros(num_episodes)
loss_counts = np.zeros(num_episodes)

def smart_opponent(state):
    """Return the opponent's action (blocking strategy)."""
    available_actions = np.where(state == 0)[0]
    for action in available_actions:
        state[action] = 2
        if check_winner(state) == 2:  # If the opponent can win
            state[action] = 0  # Reset
            return action
        state[action] = 0  # Reset
    
    # If no blocking move, return random
    return np.random.choice(available_actions)

# Training the agent
for episode in range(num_episodes):
    state = np.zeros(9, dtype=int)  # Empty board

    while True:
        state_index = state_to_index(state)

        if np.random.rand() < epsilon:
            available_actions = np.where(state == 0)[0]  # Available moves
            if available_actions.size == 0:  # No available actions
                draw_counts[episode] += 1
                break  # End the game if board is full
            action = np.random.choice(available_actions)  # Explore
        else:
            action = np.argmax(q_table[state_index])  # Exploit

        # Make the move
        state[action] = 1  # Agent's move

        # Check for winner
        winner = check_winner(state)
        if winner == 1:  # Agent wins
            win_counts[episode] += 1
            break
        elif winner == -1:  # Draw
            draw_counts[episode] += 1
            break

        # Opponent's smart move
        opponent_action = smart_opponent(state)
        state[opponent_action] = 2  # Opponent's move

        # Check for opponent's win
        winner = check_winner(state)
        if winner == 2:  # Opponent wins
            loss_counts[episode] += 1
            break

        # Update Q-value
        next_state_index = state_to_index(state)
        reward = 0  # Default reward
        if winner == 1:
            reward = 1  # Win
        elif winner == -1:
            reward = 0  # Draw
        elif winner == 2:
            reward = -1  # Loss
        
        q_table[state_index, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state_index]) - 
            q_table[state_index, action]
        )

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Plotting the results
plt.figure(figsize=(12, 6))
episodes = np.arange(num_episodes)

plt.plot(episodes, np.cumsum(win_counts), label='Wins', color='green')
plt.plot(episodes, np.cumsum(loss_counts), label='Losses', color='red')
plt.plot(episodes, np.cumsum(draw_counts), label='Draws', color='blue')

plt.xlabel('Episodes')
plt.ylabel('Total Outcomes')
plt.title('Tic-Tac-Toe Agent Performance Over Episodes')
plt.legend()
plt.grid()
plt.show()