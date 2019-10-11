# -*- coding: utf-8 -*-
from time import *
import random
import game as cf
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import huber_loss
from keras import backend as K
EPISODES = 2001

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.995 # discount rate
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
        model.add(Dense(42, input_dim=self.state_size, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(84, activation='relu'))
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
    playerOne.load("./save/cfOne2000.h5")
    playerTwo.load("./save/cfTwo2000.h5")
    done = 0
    batch_size = 42

    for e in range(EPISODES):
        cf.reset()
        state = cf.board
        state = np.reshape(state, [1, state_size])
        time = 0  # time is used just to count frames as a measurement of how long the ai lasted
        while True:
            sleep(5)
            time += 1
            actionOne = playerOne.act(state)
            moveOne = cf.dropTile(actionOne, 1)
            cf.render()
            print("/////////////////")
            currentBoard = np.reshape(cf.board, [1, state_size])
            actionTwo = playerTwo.act(currentBoard)
            moveTwo = cf.dropTile(actionTwo, -1)
            cf.render()
            print("/////////////////")
            nextStateOne = cf.board
            backupBoard = cf.board
            currentBoard = np.reshape(cf.board, [1, state_size])
            cf.dropTile(playerOne.act(currentBoard), 1)
            nextStateTwo = cf.board
            cf.board = backupBoard
            
            rewardOne = time
            rewardTwo = time
            done = cf.hasWon(1)
            if (done == 0):
                done = cf.hasWon(-1)
            # for this problem we want the pole to balance forever
            # so giving a negative reward if its finished should train the000000000000 ai to play forever
            if (done == -1):
                rewardOne += 22
                rewardTwo = -1 * rewardTwo + 22
            elif (done == 1):
                rewardOne = -1 * rewardOne + 22
                rewardTwo += 22
            if (moveOne == -1):
                rewardOne = -1000
            if (moveTwo == -1):
                rewardTwo = -1000
            nextStateOne = np.reshape(nextStateOne, [1, state_size])
            nextStateTwo = np.reshape(nextStateTwo, [1, state_size])
            state = nextStateOne
            if (done != 0):
                print("Player One: episode: {}/{}, score: {}, e: {:.2}, g: {:.2}"
                      .format(e, EPISODES, rewardOne, playerOne.epsilon, playerOne.gamma))
                print("Player Two: episode: {}/{}, score: {}, e: {:.2}, g: {:.2}"
                      .format(e, EPISODES, rewardTwo, playerTwo.epsilon, playerTwo.gamma))
                break