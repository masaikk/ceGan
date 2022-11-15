import matplotlib.pyplot as plt
import re

rull = re.compile('BLEU-\[2, 3, 4, 5\] = \[(.*?),')
nll_gen_mode=re.compile('NLL_gen = (.*?),')
base_url = '/home/b8313/coding/py/gan-torch-text/log/'

if __name__ == '__main__':
    files_list = ['log_1025_1037_01_ce_vec0_8.txt']
    for filename in files_list:
        with open(base_url + filename, "r") as f:
            content=f.read()

            match = re.findall(nll_gen_mode, content)
            for count in range(0, 4):
                del match[0]

            datas = []
            for data in match:
                data = float(data)
                datas.append(data)
            # print(len(datas))
            print(datas)

            x = []
            for count in range(0, len(datas)):
                x.append(count * 50)
            # print(len(x))
            x[len(datas) - 1] -= 1
            print(x)

            # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.title('BLEU-2')
            plt.xlabel("epoch")
            plt.ylabel("BLEU2")
            plt.plot(x, datas, color='b')
            plt.grid()
            plt.show()
