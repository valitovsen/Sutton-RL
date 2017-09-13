import random
import numpy as np
import matplotlib.pyplot as plt


class Setting:
    def __init__(self, k=10, initial=0, **agentKwargs):
        self.arms = [Arm() for i in range(k)]
        self.agent = Agent(**agentKwargs)
        self.initial = initial
        for i in range(len(self.arms)):
            self.agent.estimates.append(self.initial)
            self.agent.history.append(0)


    def reset(self):
        for i in range(len(self.arms)):
            self.agent.estimates[i] = self.initial
            self.agent.history[i] = 0
            self.arms[i].value = random.gauss(0,1)

    def start(self, steps):
        self.reset()
        log = np.zeros(shape=(2,steps)) #first row is reward, second is if opt
        values = [arm.value for arm in self.arms]
        opt = [i for i,x in enumerate(values) if x == max(values)]
        step = 0
        while step < steps:
            action = self.agent.chooseAction()
            reward = self.arms[action].reward()
            self.agent.updateEstimates(action, reward)
            log[0][step] = reward
            log[1][step] = action in opt
            step += 1
        return log


class Arm:
    def __init__(self):
        self.value = None
    def reward(self):
        return random.gauss(self.value, 1)


class Agent:
    def __init__(self, policy='egreedy', epsilon=0.1, degree=2, step=0.1):
        self.estimates = []
        self.history = []
        if policy == 'egreedy':
            self.polargs = [self.estimates, epsilon]
            self.pol = epsilonGreedySelection
            self.updargs = [self.estimates, self.history]
            self.upd = incrementAverageUpdate
        elif policy == 'ucb':
            self.polargs = [self.estimates, self.history, degree]
            self.pol = UCBSelection
        elif policy == 'grad':
            self.averReward = 0
            self.polargs = [self.estimates]
            self.pol = gradSelection
            self.updargs = [self.estimates, self.averReward, step, self.history]
            self.upd = gradUpdate
        else: raise ValueError('Invalid policy value')

    def chooseAction(self):
        action = self.pol(*self.polargs)
        self.history[action] += 1
        return action

    def updateEstimates(self, action, reward):
        self.upd(action, reward, *self.updargs)

'''
AGENT POLICY FUNCTIONS:
'''
def epsilonGreedySelection(estimates, epsilon):
    '''
    Implementation of epsilon-greedy policy.
    estimates: list of action value estimates
    epsilon: epsilon in [0,1]
    Returns index of action to take.
    '''
    if epsilon > 1 or epsilon < 0: raise ValueError('Invalid epsilon value')
    if random.uniform(0,1) < epsilon:
        action = random.randint(0, len(estimates)-1)
    else:
        action = random.choice([i for i,x in enumerate(estimates) #breaks ties randomly
                                    if x == max(estimates)])
    return action

def UCBSelection(estimates, history, degree=2):
    '''
    Implementation of upper-confidence-bound action selection policy.
    estimates: list of action value estimates
    history: list of number of times taken per action
    degree: controls the degree of exploration
    '''
    lnt = np.log(sum(history)+1)
    ucb = np.add([degree*np.sqrt(lnt/n) if n!=0 else np.inf for n in history], #if statement to avoid ZeroDivisionError
                                                                    estimates)
    action = np.random.choice([i for i,x in enumerate(ucb) if x == max(ucb)]) #breaks ties randomly
    return action

def gradSelection(preferences):
    '''
    Implementation of softmax action selection policy.
    preferences: list of action preferences
    '''
    probs = np.exp(preferences)/ np.sum(np.exp(preferences), axis=0) #apply softmax
    action = np.random.choice(len(preferences), p=probs) #choose random according to probs
    return action

'''
AGENT ESTIMATES UPDATE FUNCTIONS:
'''
def incrementAverageUpdate(action, reward, estimates, history):
    estimates[action] = (estimates[action] +
                (reward - estimates[action]) / history[action])

def constantStepUpdate(action, reward, estimates, step):
    if step<=0 or step>1: raise ValueError('Invalid step value')
    estimates[action] = (estimates[action] + step*(reward - estimates[action]))

def gradUpdate(action, reward, preferences, averageReward, step, history):
    probs = np.exp(preferences)/ np.sum(np.exp(preferences), axis=0)
    averageReward = averageReward + (reward - averageReward)/sum(history)
    for i in range(len(preferences)):
        if i == action:
            preferences[i] = (preferences[i] +
                    step*(reward - averageReward)*(1-probs[i]))
        else:
            preferences[i] = (preferences[i] - step*(reward -
                    averageReward)*probs[i])


if __name__ == '__main__':
    k = 10
    initial = 0
    steps = 1000
    runs = 2000
    agentKwargs = {'policy':'grad', 'step':0.1}
    env = Setting(k, initial, **agentKwargs)
    allRewards = np.zeros(steps)
    allOpts = np.zeros(steps)
    for run in range(runs):
        env.reset()
        print('\r\u03b5 = %s: %s/%s' %(0.1,run+1,runs), end='', sep='', flush=True)
        res = env.start(steps)
        allRewards = np.add(allRewards, res[0])
        allOpts = np.add(allOpts, res[1])
    allRewards = allRewards/runs
    allOpts = allOpts/runs
    plt.figure(1)
    t = np.arange(0,steps,1)
    plt.plot(t, allRewards,linewidth=0.3)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.figure(2)
    t = np.arange(0,steps,1)
    plt.plot(t, allOpts,linewidth=0.3)
    plt.xlabel('steps')
    plt.ylabel('% opts')
    plt.show()
