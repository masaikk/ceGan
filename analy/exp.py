import matplotlib.pyplot as plt
import numpy as np
temper=100
N=3000

def get_exp_list(temper :int)->list:
    return [temper ** (i / 3000) for i in range(3000)]
if __name__ == '__main__':
    # x=[i for i in range(3000)]
    exp=[100 ** (i / 3000) for i in range(3000)]
    sigmoid=[(temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1 for i in range(3000)]
    plt.figure(figsize=(25.6, 19.2))
    # datas=[]
    # datas.append(exp)
    # datas.append(sigmoid)
    # plt.plot([i for i in range(N)],datas)
    plt.plot(sigmoid)
    plt.plot(exp)
    plt.show()