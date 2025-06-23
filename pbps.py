import pandas as pd
import pick_stock as ps
import backtest as bt
import performance as p
import numpy as np


hs300 = pd.read_excel(r'..\data\report_data\isin_hs300.xlsx',index_col=0)
celrl = pd.read_excel(r'../data/report_data/全a-超额利润率-整体法-考虑报告披露日期.xlsx', parse_dates=[0], index_col=0)#超额利润率-整体法
benchmark = pd.read_excel(r'..\data\trade_data\hs300.xlsx',index_col='time',parse_dates=['time'])
name = pd.read_excel(r'..\data\cf\shenwan.xlsx', usecols=['股票代码','股票简称'])
price = pd.read_excel(r'..\data\trade_data\price.xlsx', parse_dates=['time'], index_col='time')
pb_per = pd.read_excel(r'..\data\trade_data\pb_per.xlsx', parse_dates=[0], index_col=0)
ps_per = pd.read_excel(r'..\data\trade_data\ps_per.xlsx', parse_dates=[0], index_col=0)
# union = hs300.iloc[0].loc[hs300.iloc[0]==1].index
# for i in range(1,len(hs300.index)):
#     union=union.union(hs300.iloc[i].loc[hs300.iloc[i]==1].index)


change_date = pb_per.index


howmany = 100
# change_date = pd.date_range(start='2011-07-04',end='2024-02-02',freq='7d')
# index_position = change_date.get_loc(pd.Timestamp('2024-01-01'))
# change_date=change_date.to_list()
# change_date[index_position]=pd.Timestamp('2024-01-02')
# change_date = change_date+[pd.Timestamp('2024-02-05')]+[pd.Timestamp('2024-02-19')]+[pd.Timestamp('2024-02-26')]+[pd.Timestamp('2024-03-04')]+[pd.Timestamp('2024-03-11')]+[pd.Timestamp('2024-03-18')]+[pd.Timestamp('2024-03-25')]+[pd.Timestamp('2024-04-01')]+[pd.Timestamp('2024-04-08')]+[pd.Timestamp('2024-04-15')]+[pd.Timestamp('2024-04-22')]+[pd.Timestamp('2024-04-29')]+[pd.Timestamp('2024-05-06')]+[pd.Timestamp('2024-05-10')]+[pd.Timestamp('2024-05-17')]+[pd.Timestamp('2024-05-24')]+[pd.Timestamp('2024-05-31')]+[pd.Timestamp('2024-06-07')]+[pd.Timestamp('2024-06-14')]+[pd.Timestamp('2024-06-21')]+[pd.Timestamp('2024-06-28')]+[pd.Timestamp('2024-07-05')]+[pd.Timestamp('2024-07-12')]+[pd.Timestamp('2024-07-19')]+[pd.Timestamp('2024-07-26')]+[pd.Timestamp('2024-08-02')]+[pd.Timestamp('2024-08-09')]+[pd.Timestamp('2024-08-16')]+[pd.Timestamp('2024-08-23')]+[pd.Timestamp('2024-08-30')]+[pd.Timestamp('2024-09-06')]
# change_date = pd.DatetimeIndex(change_date)
weight = 1/howmany
bench = (1+benchmark/100).cumprod()
bench.to_excel(r'../NAV/bench.xlsx')
start = '2011-07-04'
price = price.loc[start:]
price = price.fillna('--')
benchmark = benchmark[start:]
ps_per = ps_per.reindex(price.index).ffill().bfill()
pb_per = pb_per.reindex(price.index).ffill().bfill()
celrl = celrl.reindex(price.index).ffill().bfill()
hs300 = hs300.reindex(hs300.index.union(price.index)).ffill()
hs300.replace({'是': 1, '否': 0}, inplace=True)

stock_avaliable = pd.DataFrame()
rank = pd.DataFrame()
weights = [1,1]  # 可以设定的权重
result = pd.DataFrame()
for date in price.index:
    index_2 = pb_per.loc[date][pb_per.loc[date] < 100].index
    index_3 = ps_per.loc[date][ps_per.loc[date] < 100].index
    index_4 = hs300.loc[date][hs300.loc[date] == 1].index
    # index = index_2.intersection(index_3.intersection(index_4))
    stock_avaliable['pb_per'] = pb_per.loc[date][index_4]
    stock_avaliable['ps_per'] = ps_per.loc[date][index_4]
    rank = stock_avaliable.rank()
    rank['Weighted Sum'] = rank.iloc[:, :].dot(weights)
    rank = rank.sort_values(by='Weighted Sum')
    result[date] = rank.index[-howmany:]
    stock_avaliable = pd.DataFrame()
    rank = pd.DataFrame()
final_nav,final_result=ps.get_result(price,change_date,result)
final_result_unstack = ps.unstack_result(final_result,weight,name,change_date,howmany)
final_result_unstack = ps.data_attach(final_result_unstack,[pb_per,ps_per],['pb所处历史百分位数','ps所处历史百分位数'])
final_result_unstack = ps.position_change_reporter(final_result_unstack,change_date[0],change_date[-1])
final_result_unstack.iloc[-howmany*50:].to_excel(r'..\result\沪深300指数增强-pbps.xlsx',index=False)
NAV=bt.get_NAV(final_nav)
# NAV.to_excel(r'../NAV/pbps.xlsx')
bt.plot_backtest(bench,[NAV],['沪深300指数增强-pbps'],'高pbps')








#p.performance_analysis(NAV,bench, "TMT", "roe、成长、高PBPS（绝对值）、研发费用.xlsx", "2023-04-03")
# print('日胜率：',p.win_rate_daily(final_nav,benchmark))
# print('周胜率',p.win_rate_changedate(final_nav,benchmark,change_date))
# ic,ir,ic_series,ic_cumsum = p.ic(final_result,price,change_date)
# maxdrawdown_dl,maxdrawdown_high_dl,maxdrawdown_low_dl,date_high_dl = p.maxdrawdown(NAV)
#
# p.performance_analysis(NAV,bench,'沪深300','指数增强-pbps','2011-07-04')


