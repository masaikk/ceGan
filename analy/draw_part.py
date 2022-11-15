from analy.exp import get_exp_list
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

begin_index = 1000
end_index = 1050

base_url = '/home/b8313/coding/py/gan-torch-text/'
if __name__ == '__main__':
    df = pd.read_csv(base_url + 'load_pic/1103000829/1103_0008_29.csv')
    plt.figure(figsize=(8, 6))
    plt.plot(range(begin_index, end_index), df['selected'][begin_index:end_index], label='evolution')
    plt.plot(range(begin_index, end_index), get_exp_list(25)[begin_index:end_index], label='pure')
    plt.legend()
    plt.show()
