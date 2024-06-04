import numpy as np
import tensorflow as tf
from tensorflow import Dense, LSTM, Flatten
from tensorflow import Adam
import gym

# Define RL environment
class ImageToTextEnv(gym.Env):
    def __init__(self, api_key):
        super(ImageToTextEnv, self).__init__()
        self.api_key = api_key
        self.action_space = gym.spaces.Discrete(10)  
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)  
        # Other initialization, such as API setup

    def reset(self):
        # Reset environment
        return np.zeros((224, 224, 3), dtype=np.uint8)  

    def step(self, action):
        reward = 1 if action == 0 else -1
        next_state = np.zeros((224, 224, 3), dtype=np.uint8)  
        done = False  
        info = {}  
        return next_state, reward, done, info

    def render(self):
        # Optional: visualize environment state
        pass

# Define RL model
class RLModel(tf.keras.Model):
    def __init__(self, action_space):
        super(RLModel, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.lstm = LSTM(128)
        self.dense2 = Dense(action_space, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.lstm(x[:, tf.newaxis, :])  # Reshape to (batch_size, timesteps, input_dim)
        return self.dense2(x)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
num_episodes = 1000
epsilon = 0.1  # Epsilon for ε-greedy exploration

# Create environment
api_key = "K88153881288957"
env = ImageToTextEnv(api_key)

# Create RL model
model = RLModel(env.action_space.n)  # Here we directly pass the number of actions
optimizer = Adam(learning_rate)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for timestep in range(1000): 
        with tf.GradientTape() as tape:
            # Convert state to suitable format 
            state = np.array(state)[np.newaxis, :] 

            # Get action probabilities from model
            action_probs = model(state)
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n) 
            else:
                action = np.argmax(action_probs.numpy()[0])

            # Take action, get next state and reward from environment
            next_state, reward, done, _ = env.step(action)

            # Compute discounted reward
            episode_reward += reward
            discounted_reward = reward * (gamma ** timestep)

            # Apply REINFORCE loss
            loss = -tf.math.log(action_probs[0, action]) * discounted_reward

        # Backpropagation
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if done:
            break
        state = next_state

    # Print episode info
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# Save trained model
model.save("image_to_text_rl_model.h5")
