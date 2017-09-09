from bandit_classes import Setting
import matplotlib.pyplot as plt
import numpy as np


runs = 2000
steps = 1000
k = 10
epsilon = [0.1, 0.01, 0]

def runTest(runs, eps):
    setting = Setting(k, eps)
    log = []
    for i in range(runs):
            print('\r\u03b5 = %s: %s/%s' %(eps,i,runs), end='', sep='', flush=True)
            setting.reset()
            result = setting.start(steps)
            log.append(result)
    averRewards = getAverage(log, 'rewards')
    averOpts = getAverage(log,'opts')
    print('')
    return averRewards, averOpts


def getAverage(log, key):
    return [sum(timestep)/len(timestep) for
                            timestep in zip(*[dict[key] for dict in log])]

def plotData(data, index, label):
    plt.figure(index+1)
    t = np.arange(0,steps,1)
    for key in results:
        plt.plot(t, results[key][index], label='\u03b5 = %s' %key, linewidth=0.3)
    plt.xlabel('Steps')
    plt.ylabel(label)
    plt.title(label)
    plt.legend()

if __name__ == '__main__':
    print('Starting Multiarm Bandit...')
    print('K = %s, %s runs, %s steps each' %(k,runs,steps))
    results = {}
    for eps in epsilon:
        results[eps] = runTest(runs, eps)
    plotData(results, 0, 'Average reward')
    plotData(results, 1, '% Optimal action')
    plt.show()
