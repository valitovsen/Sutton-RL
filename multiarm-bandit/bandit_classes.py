import random
import numpy as np


class Setting:
    def __init__(self, k, epsilon=0.1):
        self.arms = [Arm() for i in range(k)]
        self.agent = Agent(epsilon)

    def seedActionValues(self):
        for arm in self.arms:
            arm.value = random.gauss(0, 1)

    def seedEstimates(self, q=0):
        self.agent.estimates = [q for i in range(len(self.arms))]
        self.agent.actions = [0 for i in range(len(self.arms))]

    def reset(self):
        self.seedActionValues()
        self.seedEstimates()

    def start(self, steps):
        self.reset()
        log = {'rewards': [], 'errors': [], 'opts': []}
        values = [arm.value for arm in self.arms]
        opt = [i for i,x in enumerate(values) if x == max(values)]
        while steps:
            action = self.agent.chooseAction()
            reward = self.arms[action].reward()
            error = sum([(x-y)**2/len(self.agent.estimates) for x,y in
                            zip(self.agent.estimates, values)])
            log['rewards'].append(reward)
            log['errors'].append(error)
            log['opts'].append(action in opt)
            self.agent.updateEstimates(action, reward)
            steps -= 1
        return log


class Arm:
    def __init__(self):
        self.value = None
    def reward(self):
        return random.gauss(self.value, 1)


class Agent:
    def __init__(self, epsilon):
        self.estimates = None
        self.actions = None
        self.epsilon = epsilon

    def chooseAction(self):
        if random.uniform(0,1) < self.epsilon:
        #Choose random action
            action = random.randint(0, len(self.estimates)-1)
        else:
        #Choose optimal action, break ties randomly
            action = random.choice([i for i,x in enumerate(self.estimates)
                                    if x == max(self.estimates)])
        self.actions[action] += 1
        return action

    def updateEstimates(self, action, reward):
        #Incremental update of estimates
        self.estimates[action] = (self.estimates[action] +
                                        (reward - self.estimates[action]) /
                                            self.actions[action])
