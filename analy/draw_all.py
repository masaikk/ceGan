import re
from re import Pattern
from typing import AnyStr

import matplotlib.pyplot as plt

base_url = '/home/b8313/coding/py/gan-torch-text/log/'

bleu2mode = re.compile('BLEU-\[2, 3, 4, 5\] = \[(.*?),')
nll_gen_mode = re.compile('NLL_gen = (.*?),')
nll_div_mode = re.compile('NLL_div = (.*?),')

filename = 'log_1102_1543_10_exp+log.txt'


def get_nll(mode) -> list:
    with open(base_url + filename, "r") as f:
        content = f.read()

        # match = re.findall(nll_gen_mode, content)
        match = re.findall(mode, content)

        for count in range(0, 4):
            del match[0]

        datas = []
        for data in match:
            data = float(data)
            datas.append(data)
        # print(len(datas))
        print(filename + str(datas))

        x = []
        for count in range(0, len(datas)):
            x.append(count * 50)
        # print(len(x))
        x[len(datas) - 1] -= 1
        print(x)

        # plt.title('BLEU-2')
        # plt.xlabel("epoch")
        # # plt.ylabel("BLEU2")
        #
        # # plt.plot(x, datas, label=filename, color=cmap(color))
        # plt.plot(x, datas, label=filename)
        #
        # plt.grid()
        # plt.legend()
    return datas


def main():
    data_ora = get_nll(nll_div_mode)
    data_gen= get_nll(nll_gen_mode)
    plt.figure(figsize=(8, 6))
    # plt.plot([50 * i for i in range(len(data_ora))], data_ora)

    plt.plot([50 * i for i in range(len(data_ora))], data_ora)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title('NLL-oracle',fontsize=25)
    plt.show()


if __name__ == '__main__':
    main()
