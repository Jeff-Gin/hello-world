# hello-world
print('hello world')
import numpy as np
import random
import matplotlib.pyplot as plt


def Greedy(Phi, x_pin, error):
    y = Phi * x_pin
    x = np.mat(np.zeros(x_pin.shape))
    r = y - Phi * x
    S = []
    while np.linalg.norm(r) >= error:
        L = [np.linalg.norm(r) ** 2 - ((Phi[:, i].T * r).item(0, 0))/
             np.linalg.norm(Phi[:, i]) ** 2 for i in range(x_pin.shape[0])]
        S.append(L.index(min(L)))
        S = sorted(S) 
        x[S, :] = np.linalg.pinv(Phi[:, S]) * y
        r = y - Phi[:, S] * x[S, 0]
        
    return x


X = range(1, 11)
M, N = 30, 50
Phi0 = np.random.normal(0, 10, (M, N))
error0 = 0.001

x0 = np.mat(np.zeros((N, 1)))
a0 = sorted(random.sample(range(N), 8))
x0[a0, :] = np.mat(np.random.uniform(1, 2, (len(a0), 1)))
x_hat = Greedy(Phi0, x0, error0)
tes=x_hat-x0
print(tes)
