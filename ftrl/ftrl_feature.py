from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import hashlib
import time

train = './data/train'               # path to training file

# B, model
alpha = .05  # learning rate
beta = 1.    # smoothing parameter for adaptive learning rate
L1 = 0.5     # L1 regularization, larger value means more regularized
L2 = 0.3     # L2 regularization, larger value means more regularized

D = 2000*10000             # number of weights to use
do_interactions = False  # whether to enable poly2 feature interactions

epoch = 1      # learn training data for N passes
holdout = 100  # use every N training instance for holdout validation

class ftrl_proximal(object):
    def __init__(self, alpha, beta, L1, L2, D, interaction=False):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = [0.] * D  # use this for execution speed up
        # self.w = {}  # use this for memory usage reduction

    def _indices(self, x):
        for i in x:
            yield i

        if self.interaction:
            D = self.D
            L = len(x)
            for i in xrange(1, L):  # skip bias term, so we start at 1
                for j in xrange(i+1, L):
                    yield (i * j) % D

    def predict(self, x):
        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        n = self.n
        z = self.z
        w = self.w  # use this for execution speed up

        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            if sign * z[i] <= L1:
                w[i] = 0.
            else:
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        self.w = w
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        alpha = self.alpha

        n = self.n
        z = self.z
        w = self.w  # no need to change this, it won't gain anything

        g = p - y
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def data(path, D):
    for t, row in enumerate(DictReader(open(path))):
        ID = row['id']
        del row['id']
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        row['hour'] = row['hour'][6:]
        row["C15C16"] = row['C15'] + row['C16'] 
        del row['C15']
        del row['C16']

        x = [0]
        for key in sorted(row):  # sort is for preserving feature ordering
            value = row[key]
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield t, ID, x, y


start = datetime.now()

learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction=do_interactions)
for e in xrange(epoch):
    loss = 0.
    count = 0

    for t, ID, x, y in data(train, D):  # data is a generator
        p = learner.predict(x)

        if t % holdout == 0:
            loss += logloss(p, y)
            count += 1
        else:
            learner.update(x, p, y)

        if t % (holdout*10) == 0 and t > 1:
            print(' %s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), t, loss/count))

    print('Epoch %d finished, holdout logloss: %f, elapsed time: %s' % (e, loss/count, str(datetime.now() - start)))
