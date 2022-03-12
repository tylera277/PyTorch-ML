
import numpy as np
import matplotlib.pyplot as plt

import random
import gym
import time

import torch
from torch import optim, nn
from collections import deque

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

print(env.observation_space.shape[0])
print(env.action_space.n)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.out = nn.Linear(64, 4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        action_prob = self.out(out)
        return action_prob


# Hyper parameters
LR = 0.001
GAMMA = 0.99
EPSILON = 0.9
# MEMORY_CAPACITY = 500
BATCH_SIZE = 64
Q_NETWORK_ITERATION = 1000


class DQN():

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.prediction_net = NeuralNetwork(), NeuralNetwork()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = deque(maxlen=1000000)

        self.optimizer = torch.optim.Adam(self.prediction_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()

    def choose_action(self, state, episode, epsilon_decrements):

        try:
            EPSILON = epsilon_decrements[episode]
        except IndexError:
            EPSILON = 0.01

        action_value = self.prediction_net(state)

        random_number = random.random()

        if random_number < EPSILON:
            # Random action
            index = random.randint(0, 3)
        else:
            # Action which has the maximum q value
            index = np.argmax(action_value.detach().numpy())


        return index

    def store_transition(self, state, action, reward, next_state, terminal):
        state = state.detach().numpy()
        state = state.flatten()
        next_state = next_state.flatten()

        self.memory.append((state, action, reward, next_state, terminal))

    def learn(self):
        # Every C iterations, copies over parameters from prediction to target network
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.prediction_net.state_dict())

        self.learn_step_counter += 1

        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        batch_state = np.array([i[0] for i in minibatch])
        batch_action = np.array([i[1] for i in minibatch])
        batch_reward = np.array([i[2] for i in minibatch])
        batch_next_state = np.array([i[3] for i in minibatch])
        batch_terminal = np.array([i[4] for i in minibatch])

        batch_state = torch.from_numpy(batch_state).float()
        batch_next_state = torch.from_numpy(batch_next_state).float()

        batch_action = torch.from_numpy(batch_action).float()
        batch_action = batch_action.unsqueeze(-1)
        batch_action = batch_action.type(torch.int64)

        batch_terminal = torch.from_numpy(batch_terminal).float()
        batch_reward = torch.from_numpy(batch_reward).float()

        q_eval = self.prediction_net(batch_state).gather(1, batch_action)

        q_next = torch.max(self.target_net(batch_next_state), dim=1)[0]

        q_target = batch_reward + (GAMMA*(q_next*(1-batch_terminal)))
        q_target = q_target.unsqueeze(-1)

        q_target.detach()
        loss = self.loss_func(q_target, q_eval)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss




###############

dqn = DQN()

# Storage elements
score_storage = []
counter = 0
score = -200
epsilon_decrements = np.linspace(1, 0.01, 200)

for epoch in range(400):

    state = env.reset()
    score_storage.append(score)
    print(epoch)
    print(score)
    score = 0
    counter = 0
    while counter < 1000:

        state = np.reshape(state, (1, 8))
        state = torch.from_numpy(state).float()

        action = dqn.choose_action(state, epoch, epsilon_decrements)
        env.render()
        next_state, reward, done, _ = env.step(action)
        score += reward

        next_state = np.reshape(next_state, (1, 8))

        dqn.store_transition(state, action, reward, next_state, done)
        if counter % 4 == 0:
            loss = dqn.learn()

        state = next_state
        # print(len(reward_storage))
        counter += 1

        if done:
            break

    # Average score of last 100 episode
    is_solved = np.mean(score_storage[-100:])
    if is_solved > 200:
        print('\n Task Completed! \n')
        break
    print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        #time.sleep(1)

fig1, ax1 = plt.subplots()
ax1.plot(score_storage)
ax1.set_title('score')

fig1.savefig('./score3.png')

plt.show()




