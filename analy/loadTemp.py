
import re
import pandas as pd
import matplotlib.pyplot as plt
from analy.exp import get_exp_list
import numpy as np


# rull to find the evolution choice
findchoice = re.compile('chosen evolution choice as (.*?) and temp is')

# rull to find the temp and corresponding score as a string like 'temp0, score0], [temp1, score1'
findresult = re.compile('all temperature with score are \[\[(.*?)\]\]')

# rull to find the selected temp and its score and index as a tuple like (selected_temp,  selected_temp.score,
# selected_temp.child_index)
findselected = re.compile(' This epoch temperature: (.*?) with score: (.*?) with child_index as (\d)')

base_url = '/home/b8313/coding/py/gan-torch-text/temp_log/'


# get the dictionary object required to build the dataframe object
def getdic(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12):
    df = {'temp-1.ec': p1, 'temp.ec': p2, 'temp+1.ec': p3,
          'temp-1': p4, 'temp': p5, 'temp+1': p6,
          'temp-1.score': p7, 'temp.score': p8, 'temp+1.score': p9,
          'selected.ec': p10, 'selected': p11, 'selected.score': p12}

    return df




if __name__ == '__main__':
    files_list = ['temp_log_1102_1543_10_exp+log.txt']

    for filename in files_list:
        with open(base_url + filename, "r") as f:
            content = f.read()

            evolution_choices = re.findall(findchoice, content)
            # print(evolution_choices)

            results = re.findall(findresult, content)
            # print(results)

            selecteds = re.findall(findselected, content)
            # print(selecteds)

            # reformat the list, make the tuples in it to be lists and change the data types to desire ones
            selected_temps = []
            for selected in selecteds:
                selected_temp = [float(selected[0]), float(selected[1]), int(selected[2])]
                selected_temps.append(selected_temp)

            datas = []
            for result in results:
                # change the format of the data to 'temp0,score0,temp1,score1'
                result = result.replace("]", "")
                result = result.replace("[", "")
                result = result.replace(" ", "")

                # change the format of the data to [temp0,score0,temp1,score1], change the data types to float
                result = result.split(",")
                for i in range(0, len(result)):
                    result[i] = float(result[i])

                datas.append(result)

            # deal with epoch 0 and 1 separately
            df = getdic('null', evolution_choices[0], evolution_choices[1],
                        'null', datas[0][0], datas[0][2],
                        'null', datas[0][1], datas[0][3],
                        evolution_choices[selected_temps[0][2]], selected_temps[0][0], selected_temps[0][1])
            df = pd.DataFrame(df, index=[0])

            new = getdic('null', evolution_choices[2], evolution_choices[3],
                        'null', datas[1][0], datas[1][2],
                        'null', datas[1][1], datas[1][3],
                        evolution_choices[selected_temps[1][2]], selected_temps[1][0], selected_temps[1][1])
            new = pd.DataFrame(new, index=[0])
            df = df.append(new, ignore_index=True)

            # deal with the rest epoch
            for epoch in range(2, len(selected_temps)):
                new = getdic(evolution_choices[epoch * 3], evolution_choices[epoch * 3 - 2], evolution_choices[epoch * 3 - 1],
                             datas[epoch][4], datas[epoch][0], datas[epoch][2],
                             datas[epoch][5], datas[epoch][1], datas[epoch][3],
                             evolution_choices[epoch * 3 + selected_temps[epoch][2] - 2], selected_temps[epoch][0], selected_temps[epoch][1])

                new = pd.DataFrame(new, index=[0])
                df = df.append(new, ignore_index=True)

            df.columns.name = 'epoch'
            # 显示所有列
            pd.set_option('display.max_columns', None)
            # 显示所有行
            pd.set_option('display.max_rows', None)
            # # 设置value的显示长度为200，默认为50
            # pd.set_option('max_colwidth', 200)

            pd.set_option('display.width', 1000)
            print(df)

            # df.to_csv('../load_pic/data/{}.csv'.format(filename[9:21]),header=True,index=False)


            plt.figure(figsize=(8,6))
            # plt.plot(df['selected.score'])
            datas=df['selected']
            data_ava=[]
            for i in range(int(3000/50)):
                data_ava.append(np.mean(datas[i*50:(i+1)*50]))
            # plt.scatter([i *30 for i in range(len(df['selected'][::30]))],df['selected'][::30])
            # plt.scatter(range(len(df['selected'])),df['selected'])
            # plt.plot(get_exp_list(25))
            plt.plot([i* 50 for i in range(len(data_ava))],data_ava)
            plt.scatter([i* 50 for i in range(len(data_ava))], data_ava,label='Temperature')
            plt.legend()

            plt.show()