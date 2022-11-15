# -*- coding = utf-8 -*-
# @TIME : 2021/10/27 16:27
# @Author : Ling
# @File : test1
# @Software : PyCharm

import matplotlib.pyplot as plt
import re

bleu2mode = re.compile('BLEU-\[2, 3, 4, 5\] = \[(.*?),')
nll_gen_mode = re.compile('NLL_gen = (.*?),')
nll_div_mode = re.compile('NLL_div = (.*?),')
base_url = '/home/b8313/coding/py/gan-torch-text/log/'
# base_url = 'D:\coding\py\gan-torch-text\log\\'

# N = 160


# def get_cmap(n, name='hsv'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)


if __name__ == '__main__':
    files_list = [
        # 'log_1023_2112_01_ce_vec0_8.txt',
        # 'log_1023_2326_06_ce_vec0_9.txt',
        # 'log_1023_2334_14_ce_vec0_7.txt',
        # 'log_1024_0757_24_ce_vec0_6.txt',
        # 'log_1024_1424_26_ce_vec0_2.txt',
        # 'log_1024_1631_49_ce_vec0_1.txt',
        # 'log_1024_1633_52_ce_vec0_0.txt',
        # # 'log_1025_0904_05_ce_vec0_5.txt',
        # 'log_1024_2306_48_ce_vec0_0.txt',
        # 'log_1025_1126_03_ce_vec1_0.txt',
        # 'log_1025_1445_17_ce_vec1_0.txt',
        # 'log_1025_2112_11_ce_vec0_0.txt',
        # 'log_1025_2259_34_ce_vec0_0.txt',
        # "log_1026_0853_19_ce_vec0_0.txt",
        # 'log_1026_0859_47_ce_vec0_0.txt',
        # 'log_1026_1702_17_ce_vec0_0.txt',
        # 'log_1026_1702_17_ce_vec1_0.txt',
        # "log_1026_1919_58_ce_vec1_0.txt",
        # 'log_1026_1959_51_ce_vec1_0.txt',
        # 'log_1026_2002_09_ce_vec1_0.txt',
        # 'log_1027_1751_51_ce_vec1_0_log.txt',
        # 'log_1027_1947_04_ce_vec0_0_log.txt',
        # 'log_1027_2245_37_ce_vec0_5_log.txt'
        # 'log_1028_1143_34_ce_vec1_0_log_temp250.txt',
        # 'log_1028_2026_28_ce_vec0_5_log_temp250.txt',
        # 'log_1029_0827_34_ce_vec0_0_log_temp250.txt',
        # # 'log_1028_0826_32_ce_vec0_5_exp.txt',
        # 'log_1029_0945_17_ce_vec_0_8_log_temp250.txt',
        # 'log_1029_1825_45_ex_vec0_5.txt',
        # 'log_1030_0409_17_ce_vec0_5_log.txt',
        # 'log_1030_0858_17_ce_vec0_5_sqrt_temp50.txt',
        # 'log_1030_1318_16_ce_vec0_5_log_temp50.txt',
        # 'log_1030_1735_42_ce_vec0_5_exp_temp50.txt',
        # 'log_1030_2312_59_exp+log_temp1000.txt',
        # 'log_1031_0334_28_log_temp1000.txt',
        # 'log_1031_0757_39_exp_temp1000.txt'
        # 'log_1029_1825_45_ex_vec0_5.txt',
        # 'log_1030_0409_17_ce_vec0_5_log.txt',
        # 'log_1030_0858_17_ce_vec0_5_sqrt_temp50.txt',
        # 'log_1030_1318_16_ce_vec0_5_log_temp50.txt',
        # 'log_1030_1735_42_ce_vec0_5_exp_temp50.txt'
        # 'log_1102_0946_31_--mu_temp exp --fn_mu_temp "exp sigmoid quad"  temp150.txt'
        # 'log_1102_1543_10_exp+log.txt'
        #         'log_1102_2012_32_exp.txt'
        'log_1102_1543_10_exp+log.txt',
        'log_1102_2031_32_exp+log.txt',
        'log_1103_0052_47_exp+log.txt',
        'log_1103_0516_11_exp+log.txt',
        # 'log_1103_1056_28_exp+log.txt',
        # 'log_1103_1100_09_exp+log.txt',
        # 'log_1103_1703_45_exp.txt',
        # 'log_1103_2123_34_exp.txt',
        'log_1104_0217_47_exp.txt',
        'log_1104_0635_14_exp.txt'

    ]
    # cmap = get_cmap(N)
    colors = ['black', 'red', 'darkorange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'saddlebrown', 'lime',
              'hotpink', 'deepskyblue', 'fuchsia', 'gold', 'olive', 'navy', 'gray']
    color = 0
    plt.figure(figsize=(12.8, 9.6))
    for filename in files_list:
        with open(base_url + filename, "r") as f:
            content = f.read()

            # match = re.findall(nll_gen_mode, content)
            match = re.findall(nll_div_mode , content)

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
            plt.xlabel("epoch")
            # plt.ylabel("BLEU2")

            # plt.plot(x, datas, label=filename, color=cmap(color))
            plt.plot(x, datas, label=filename, color=colors[color])
            color += 1
            # color += 16

            plt.grid()
            plt.legend()
            # plt.show()
    plt.show()
