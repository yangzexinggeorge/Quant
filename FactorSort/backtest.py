import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'

#计算bench和NAV
def get_bench(price):#price为股票涨跌幅
    price = price.replace('--', np.nan)
    price_avg = price[price.notna()].mean(axis=1)#求每日股票涨跌幅均值
    return (1 + price_avg / 100).cumprod()

def get_NAV(price_chosen):#price_chosen为持仓股票的每日涨跌幅
    return pd.DataFrame((1 + price_chosen.mean(axis=1) / 100).cumprod())

#画图
def plot_bench(bench):
    plt.xticks(size=12)
    plt.plot(bench.index, bench.values, color='b',label='bench')
    plt.legend()
    return plt.show()

def plot_nav(nav_list,label_list):#nav_list为nav组成的列表，label_list为图例名组成的列表
    plt.xticks(size=12)
    color_list = ['r','g','c','m','y','b']#颜色不重复的话最多能画6个策略
    for i,nav in enumerate(nav_list):
        plt.plot(nav.index, nav.values, color=f'{color_list[i]}',label=f'{label_list[i]}')
    plt.legend()
    return plt.show()

def plot_backtest(bench,nav_list,label_list,title=''):#nav_list为nav组成的列表，label_list为图例名组成的列表
    plt.xticks(size=12)
    color_list = ['r', 'g', 'c', 'm', 'y']
    plt.plot(bench.index, bench.values, color='b', label='bench')
    for i,nav in enumerate(nav_list):
        plt.plot(nav.index, nav.values, color=f'{color_list[i]}',label=f'{label_list[i]}')
    plt.legend()
    plt.title(title)
    return plt.show()
