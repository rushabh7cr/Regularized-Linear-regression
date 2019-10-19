import numpy as np
from numpy import genfromtxt
import sys
import matplotlib.pyplot as plt
import time
start_time = time.time()

def train(file,x,t,x1,t1):
    x_transpose = np.transpose(x)

    MSE = []
    MSE_test = []
    for j in range(151):
        t_hat = []
        # w = (λI + ΦTΦ)^−1 * ΦTt
        a = np.linalg.inv(x_transpose.dot(x) + (j * np.identity(len(x_transpose))))
        w = a.dot(x_transpose.dot(t))

        MSE_test.append(test(w,x1,t1,j)) # for each w corresponding to λ find MSE for test set

        # ti = WTφ(xi)
        for i in range(len(x)):
            ti = 0
            for k in range(len(w)):
                ti += w[k]*x[i][k]
            t_hat.append(ti)

        # MSE for train set
        e=0
        for i in range(len(t)):
            e += (t[i] - t_hat[i]) * (t[i] - t_hat[i])

        mse = (e/len(x))
        MSE.append(mse)
    #print(MSE)
    #print(MSE_test)
    lamdaa = []
    for i in range(151):
        lamdaa.append(i)

    y = MSE
    plt.ylabel("MSE")
    plt.xlabel("λ")
    plt.title(file)
    plt.plot(lamdaa,y,label='Train')

    y1 = MSE_test

    plt.plot(lamdaa, y1,label='Test')
    plt.legend()
    #plt.show()
    plt.savefig('part1-'+file+'.png')

def test(w,x,t,j):
    t_hat = []
    for i in range(len(x)):
        th = 0
        for k in range(len(w)):
            th += w[k] * x[i][k]
        t_hat.append(th)

    e = 0
    for i in range(len(t)):
        e += (t[i] - t_hat[i]) * (t[i] - t_hat[i])

    mse = (e / len(x))
    return mse

if __name__ == "__main__":
    try:
        file = sys.argv[1]
        filename_train = 'train-' + file + '.csv'
        filename_trainR = 'trainR-' + file + '.csv'
        filename_test = 'test-' + file + '.csv'
        filename_testR = 'testR-' + file + '.csv'

        x = genfromtxt(filename_train,
                   delimiter=',',dtype=None)

        t = genfromtxt(filename_trainR,
                           delimiter=',',dtype=None)
        x1 = genfromtxt(filename_test,
                           delimiter=',',dtype=None)
        t1 = genfromtxt(filename_testR,
                           delimiter=',',dtype=None)
        #print(len(t))
        train(file,x,t,x1,t1)
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        print("File name incorrect. Please run again.")