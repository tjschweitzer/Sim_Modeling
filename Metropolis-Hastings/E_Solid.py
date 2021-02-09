import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def exchange(W):
    taker = np.random.randint(W.size)
    giver = np.random.randint(W.size)
    while W[giver] ==0 or taker == giver:
        giver = np.random.randint(W.size)
    W[giver] -=1
    W[taker] +=1

n = 1024
q= 1
W = np.ones(n)
for i in range(2**18):
    exchange(W)

v = np.histogram(W,density=True,bins=12)
#Per caveman grog
X = v[1][:-1] + v[1][1:]/2
Y = v[0]

Y_t = Y[0] *np.exp((-X*np.log(2)))

plt.plot(X,Y,'ko',X,Y_t,'g-')
plt.show()