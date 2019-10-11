# -*- coding: utf-8 -*-
import random
import copy
import game as cf
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import huber_loss
from keras import backend as K
import matplotlib.pyplot
rewardsOne = []
rewardsTwo = []
EPISODES = 200000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.1 # discount rate
        self.gamma_max = 1.5
        self.gamma_rise = 0.002
        self.epsilon = 35.0  # exploration rate
        self.epsilon_min = 0.20
        self.epsilon_decay = 0.99998
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """
    Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(84, input_dim=self.state_size, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(168, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
        if self.gamma < self.gamma_max:
            self.gamma += self.gamma_rise
            if self.gamma > self.gamma_max:
                self.gamma = self.gamma_max
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)

if __name__ == "__main__":
    state_size = cf.stateSize()
    print(state_size)
    action_size = cf.actionSize()
    print(action_size)
    playerOne = DQNAgent(state_size, action_size)
    playerTwo = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn-EP20.h5")
    done = 0
    batch_size = 42

    for e in range(EPISODES):
        cf.reset()
        state = cf.board
        state = np.reshape(state, [1, state_size])
        time = 0  # time is used just to count frames as a measurement of how long the ai lasted
        prevRewardOne = 0
        prevRewardTwo = 0
        while True:
            time += 1
            actionOne = playerOne.act(state)
            moveOne = cf.dropTile(actionOne, 1)
            nextStateOne = copy.deepcopy(cf.board)
            nextStateOne = np.reshape(nextStateOne, [1, state_size])
            actionTwo = playerTwo.act(nextStateOne)
            moveTwo = cf.dropTile(actionTwo, -1)
            nextStateTwo = copy.deepcopy(cf.board)
            nextStateTwo = np.reshape(nextStateTwo, [1, state_size])
            #cf.render()
            #print("/////////////////")
            rewardOne = cf.reward(1) - (cf.reward(-1)+1)
            rewardTwo = cf.reward(-1) - (cf.reward(1)+1)
            done = cf.hasWon(1)
            if (done == 0):
                done = cf.hasWon(-1)
            # for this problem we want the pole to balance forever
            # so giving a negative reward if its finished should train the000000000000 ai to play forever
                        
            if (moveOne != -1 and moveTwo != -1):
                state = nextStateOne;
            if (done == -1):
                rewardOne += -200
                rewardTwo = -1 * cf.moveCount() + 200 + rewardOne
            elif (done == 1):
                rewardOne = -1 * cf.moveCount() + 200 + rewardTwo
                rewardTwo += -200
            if (moveOne == -1):
                rewardOne += -400
            if (moveTwo == -1):
                rewardTwo += -400
            playerOne.remember(state, actionOne, rewardOne - prevRewardOne, nextStateOne, done)
            playerTwo.remember(nextStateOne, actionTwo, rewardTwo - prevRewardTwo, nextStateTwo, done)
            prevRewardOne = rewardOne
            prevRewardTwo = rewardTwo
            if (done != 0):
                playerOne.update_target_model()
                playerTwo.update_target_model()
                print("Player One: episode: {}/{}, score: {}, e: {:.2}, g: {:.2}"
                      .format(e, EPISODES, rewardOne, playerOne.epsilon, playerOne.gamma))
                print("Player Two: episode: {}/{}, score: {}, e: {:.2}, g: {:.2}"
                      .format(e, EPISODES, rewardTwo, playerTwo.epsilon, playerTwo.gamma))
                rewardsOne.append(rewardOne)
                rewardsTwo.append(rewardTwo)
                break
            if len(playerOne.memory) > batch_size:
                playerOne.replay(batch_size)
            if len(playerTwo.memory) > batch_size:
                playerTwo.replay(batch_size)
        #this saves every 10 episodes
        if e % 10 == 0:
            playerOne.save("./save/cfOne"+str(e)+".h5")
            playerTwo.save("./save/cfTwo"+str(e)+".h5")
        #if (e % 200 == 0 and e != 0):
            #matplotlib.pyplot.plot(rewardsOne)
            #matplotlib.pyplot.plot(rewardsTwo, color="olive")
            #matplotlib.pyplot.show()
