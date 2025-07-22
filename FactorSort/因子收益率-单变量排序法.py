import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False

# celrl_pj = pd.read_excel(r'../全a-超额利润率-平均法-选股回测用.xlsx', index_col=0, parse_dates=[0])
celrl_zt = pd.read_excel(r'../全a-超额利润率-整体法-选股回测用.xlsx', index_col=0, parse_dates=[0])
# celrl_QoQ_pj = pd.read_excel(r'../全a-超额利润率环比-平均法-选股回测用.xlsx', index_col=0, parse_dates=[0])
# celrl_QoQ_zt = pd.read_excel(r'../全a-超额利润率环比-整体法-选股回测用.xlsx', index_col=0, parse_dates=[0])
# celrl_QoQ_abs_pj = pd.read_excel(r'../全a-超额利润率环比-平均法-绝对值-选股回测用.xlsx', index_col=0, parse_dates=[0])
# celrl_QoQ_abs_zt = pd.read_excel(r'../全a-超额利润率环比-整体法-绝对值-选股回测用.xlsx', index_col=0, parse_dates=[0])
price = pd.read_excel(r'月度涨跌幅_后复权（分红再投）.xlsx', index_col=0, parse_dates=[0])
size = pd.read_excel(r'市值.xlsx', index_col=0, parse_dates=[0])
beta = pd.read_excel(r'市场beta.xlsx', index_col=0, parse_dates=[0])
volatility = pd.read_excel(r'年化波动率.xlsx', index_col=0, parse_dates=[0])
roa = pd.read_excel(r'ROA.xlsx', index_col=0, parse_dates=[0])
pb = pd.read_excel(r'pb市净率.xlsx', index_col=0, parse_dates=[0])


# 统一数据日期
rptdate = ['2011-05-03','2011-09-01','2011-11-01','2012-05-02','2012-09-03','2012-11-01',
            '2013-05-02','2013-11-01','2014-05-05','2014-09-01','2014-11-03',
            '2015-05-04','2015-09-01','2015-11-02','2016-05-03','2016-09-01','2016-11-01',
            '2017-05-02','2017-09-01','2017-11-01','2018-05-02','2018-09-03','2018-11-01',
            '2019-05-06','2019-09-02','2019-11-01','2020-05-06','2020-09-01','2020-11-02',
            '2021-05-06','2021-09-01','2021-11-01','2022-05-05','2022-09-01','2022-11-01',
            '2023-05-04','2023-09-01','2023-11-01','2024-05-06']

change_date = pd.DatetimeIndex(rptdate)
roa = roa.reindex(roa.index.union(change_date)).ffill().reindex(change_date)
roa = roa.reindex(size.index.union(roa.index)).shift(-1).ffill().reindex(size.index)
del rptdate, change_date



name = pd.read_excel(r'name.xlsx')
stock_available = pd.read_excel(r'可交易的股票.xlsx', index_col=0, parse_dates=[0])
# yh = name.loc[name['所属申万一级行业'] == '银行','股票代码']
# fyjr = name.loc[name['所属申万一级行业'] == '非银金融','股票代码']
# stock_available.loc[:,yh] = False
# stock_available.loc[:,fyjr] = False

time_list = celrl_zt.index

time0 = time_list[1]
time1 = time_list[2]

celrl_zt_time0 = celrl_zt.loc[time0, stock_available.loc[time0] == True].dropna()
celrl_zt_time0_sorted = celrl_zt_time0.sort_values()
price_monthly_time1 = price.loc[time1]
bins = pd.qcut(celrl_zt_time0_sorted, q=10,
               labels=['low','2','3','4','5','6','7','8','9','high'])


labels_10 = ['low','2','3','4','5','6','7','8','9','high']
labels_5 = ['low','2','3','4','high']
labels_3 = ['low','2','high']

# def calculate_returns_and_nav(time_list, factor, labels, price_monthly, group_num = 5):
#     return_monthly = []
#     size_monthly = []
#     volatility_monthly = []
#     pb_monthly = []
#     roa_monthly = []
#
#     for i in range(1, len(time_list) - 1):
#         return_this_month = []
#         size_this_month = []
#         volatility_this_month = []
#         pb_this_month = []
#         roa_this_month = []
#
#         time0 = time_list[i]
#         time1 = time_list[i + 1]
#
#         #获取本期数据
#         factor_time0 = factor.loc[time0, stock_available.loc[time0] == True].dropna()
#         factor_time0_sorted = factor_time0.sort_values()
#         price_monthly_time1 = price_monthly.loc[time1]
#
#
#         # 计算1%和99%分位数
#         lower_threshold = factor_time0_sorted.quantile(0.01)
#         upper_threshold = factor_time0_sorted.quantile(0.99)
#
#         # 过滤掉最大和最小的1%
#         filtered_factor_time0 = factor_time0_sorted[(factor_time0_sorted > lower_threshold) &
#                                               (factor_time0_sorted < upper_threshold)]
#         #对剩余数据进行分组
#         bins = pd.qcut(filtered_factor_time0, q=group_num, labels=labels)
#
#         for group in labels:
#             stock_list = bins.loc[bins == group].index
#             return_this_month.append(price_monthly_time1.loc[stock_list].mean())
#
#         return_monthly.append(return_this_month)
#         return_this_month = []
#
#     return_monthly = np.array(return_monthly)
#     return_monthly_df = pd.DataFrame(index=time_list[2:], columns=labels, data=return_monthly)
#     nav = (1 + return_monthly_df / 100).cumprod(axis=0)
#
#     return return_monthly_df, nav


def descriptive_stat(time_list, factor, labels, size, beta, volatility, pb, roa, group_num=10):
    size_monthly = []
    beta_monthly = []
    volatility_monthly = []
    pb_monthly = []
    roa_monthly = []

    for i in range(1, len(time_list)):
        size_this_month = []
        beta_this_month = []
        volatility_this_month = []
        pb_this_month = []
        roa_this_month = []

        time0 = time_list[i]
        # 获取本期数据
        factor_time0 = factor.loc[time0, stock_available.loc[time0] == True].dropna()
        factor_time0_sorted = factor_time0.sort_values()

        # 计算1%和99%分位数
        lower_threshold = factor_time0_sorted.quantile(0.01)
        upper_threshold = factor_time0_sorted.quantile(0.99)

        # 过滤掉最大和最小的1%
        filtered_factor_time0 = factor_time0_sorted[(factor_time0_sorted > lower_threshold) &
                                                    (factor_time0_sorted < upper_threshold)]
        # 对剩余数据进行分组
        bins = pd.qcut(filtered_factor_time0, q=group_num, labels=labels)

        for group in labels:
            stock_list = bins.loc[bins == group].index
            size_this_month.append(size.loc[time0,stock_list].mean())
            beta_this_month.append(beta.loc[time0,stock_list].mean())
            volatility_this_month.append(volatility.loc[time0,stock_list].mean())
            pb_this_month.append(pb.loc[time0,stock_list].mean())
            roa_this_month.append(roa.loc[time0,stock_list].mean())

        size_monthly.append(size_this_month)
        beta_monthly.append(beta_this_month)
        volatility_monthly.append(volatility_this_month)
        pb_monthly.append(pb_this_month)
        roa_monthly.append(roa_this_month)

        size_this_month = []
        beta_this_month = []
        volatility_this_month = []
        pb_this_month = []
        roa_this_month = []

    size_monthly_df = pd.DataFrame(index=time_list[1:], columns=labels, data=np.array(size_monthly))
    beta_monthly_df = pd.DataFrame(index=time_list[1:], columns=labels, data=np.array(beta_monthly))
    volatility_monthly_df = pd.DataFrame(index=time_list[1:], columns=labels, data=np.array(volatility_monthly))
    pb_monthly_df = pd.DataFrame(index=time_list[1:], columns=labels, data=np.array(pb_monthly))
    roa_monthly_df = pd.DataFrame(index=time_list[1:], columns=labels, data=np.array(roa_monthly))

    return size_monthly_df, beta_monthly_df, volatility_monthly_df, pb_monthly_df, roa_monthly_df

size_monthly_df, beta_monthly_df, volatility_monthly_df, pb_monthly_df, roa_monthly_df = descriptive_stat(time_list, celrl_zt,
                                                                                         labels_10, size, beta, volatility, pb, roa, group_num=10)

descriptive_stat_df = pd.DataFrame(index=['总市值（亿元）','市场beta','年化波动率','PB','ROA'],
                                   columns=labels_10,
                                   data=[size_monthly_df.mean(),beta_monthly_df.mean(),volatility_monthly_df.mean(),pb_monthly_df.mean(),roa_monthly_df.mean()])


def univariate_sorting(time_list, factor, labels, price, weights, group_num=10):
    price_monthly = []
    price_monthly_weighted = []

    for i in range(1, len(time_list)-1):
        price_this_month = []
        price_this_month_weighted = []

        time0 = time_list[i]
        time1 = time_list[i+1]
        # 获取本期数据
        factor_time0 = factor.loc[time0, stock_available.loc[time0] == True].dropna()
        factor_time0_sorted = factor_time0.sort_values()

        # 计算1%和99%分位数
        lower_threshold = factor_time0_sorted.quantile(0.01)
        upper_threshold = factor_time0_sorted.quantile(0.99)

        # 过滤掉最大和最小的1%
        filtered_factor_time0 = factor_time0_sorted[(factor_time0_sorted > lower_threshold) &
                                                    (factor_time0_sorted < upper_threshold)]
        # 对剩余数据进行分组
        bins = pd.qcut(filtered_factor_time0, q=group_num, labels=labels)

        for group in labels:
            stock_list = bins.loc[bins == group].index
            price_this_month.append(price.loc[time1,stock_list].mean())
            price_this_month_weighted.append(np.average(price.loc[time1,stock_list],
                                                        weights=weights.loc[time0,stock_list]))

        price_monthly.append(price_this_month)
        price_monthly_weighted.append(price_this_month_weighted)

        price_this_month = []
        price_this_month_weighted = []

    price_monthly_df = pd.DataFrame(index=time_list[2:], columns=labels, data=np.array(price_monthly))
    price_monthly_weighted_df = pd.DataFrame(index=time_list[2:], columns=labels, data=np.array(price_monthly_weighted))

    return price_monthly_df, price_monthly_weighted_df

price_df, price_weighted_df = univariate_sorting(time_list, celrl_zt, labels_10, price, size, group_num=10)
price_df['high-low'] = price_df['high'] - price_df['low']
price_weighted_df['high-low'] = price_weighted_df['high'] - price_weighted_df['low']




# 计算每列的平均值
price_df_mean = price_df.mean(axis=0)
price_df_weighted_mean = price_weighted_df.mean(axis=0)


# 计算均值的Newey-West调整的t值
def newey_west_t_value(series, lags=None):
    series = series.dropna()  # 删除NaN值
    model = sm.OLS(series, np.ones(len(series)))
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    t_value = results.tvalues[0]
    return t_value

# 选择适当的滞后期数，这里使用自动选择的方法
# 可以根据需要调整滞后期数，例如lags=4
lags = int(4 * (len(price_df.index) / 100) ** (2 / 9))

# 计算Newey-West调整的t值
nw_t_value = pd.Series(index=price_df.columns)
for group in nw_t_value.index:
    nw_t_value.loc[group] = newey_west_t_value(price_df.loc[:,group], lags=lags)

nw_t_value_weighted = pd.Series(index=price_weighted_df.columns)
for group in nw_t_value_weighted.index:
    nw_t_value_weighted.loc[group] = newey_west_t_value(price_weighted_df.loc[:,group], lags=lags)



# 计算单变量排序的秩相关系数和p值
from scipy.stats import spearmanr
def calculate_spearman_corr_and_p_values(df):
    """
    计算每对列之间的Spearman秩相关系数和p值

    参数:
    df (pd.DataFrame): 输入的DataFrame

    返回:
    spearman_corr (pd.DataFrame): Spearman秩相关系数矩阵
    p_values (pd.DataFrame): p值矩阵
    """
    # Step 1: 计算Spearman秩相关系数矩阵
    spearman_corr = df.corr(method='spearman')
    p_values = pd.DataFrame(index=df.columns, columns=df.columns)

    # Step 2: 计算每对列之间的p值
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                corr, p_value = spearmanr(df[col1], df[col2])
                p_values.loc[col1, col2] = p_value
            else:
                p_values.loc[col1, col2] = np.nan

    return spearman_corr, p_values


df = pd.concat([pd.Series(range(1,11)),price_df_mean.iloc[:-1].rank().reset_index(drop=True)],axis=1)
df_weighted = pd.concat([pd.Series(range(1,11)),price_df_weighted_mean.iloc[:-1].rank().reset_index(drop=True)],axis=1)
spearman_corr, p_values = calculate_spearman_corr_and_p_values(df)
spearman_corr_weighted, p_values_weighted = calculate_spearman_corr_and_p_values(df_weighted)



## 计算累计收益率曲线
# 等权重
accumulative_price = (1 + price_df/100).cumprod()
new_index = '2011-04-29'
new_data = [1] * len(accumulative_price.columns)
# 创建一个新的DataFrame来表示新行
new_row = pd.DataFrame([new_data], columns=accumulative_price.columns, index=[new_index])
# 将新行插入到DataFrame的第一行位置
accumulative_price = pd.concat([new_row, accumulative_price])
accumulative_price.index = pd.to_datetime(accumulative_price.index)

#画图
# 设置图形的长宽比为3:1
plt.figure(figsize=(12, 5))
plt.plot(accumulative_price.index,accumulative_price.iloc[:,0],label='超额利润率最低组', color='black', linestyle='--')
plt.plot(accumulative_price.index,accumulative_price.iloc[:,-2],label='超额利润率最高组', color='black', linestyle='-')
plt.xlabel('日期')
plt.ylabel('累计收益率')
plt.legend()
plt.show()

# 设置图形的长宽比为3:1
plt.figure(figsize=(12, 5))
plt.plot(accumulative_price.index,accumulative_price.iloc[:,-1], color='black', linestyle='-')
plt.xlabel('日期')
plt.ylabel('累计收益率')
plt.show()


#按市值加权
accumulative_price_weighted = (1 + price_weighted_df/100).cumprod()
accumulative_price_weighted = pd.concat([new_row, accumulative_price_weighted])
accumulative_price_weighted.index = pd.to_datetime(accumulative_price_weighted.index)
plt.figure(figsize=(12, 5))
plt.plot(accumulative_price_weighted.index,accumulative_price_weighted.iloc[:,0],label='超额利润率最低组', color='black', linestyle='--')
plt.plot(accumulative_price_weighted.index,accumulative_price_weighted.iloc[:,-2],label='超额利润率最高组', color='black', linestyle='-')
plt.xlabel('日期')
plt.ylabel('累计收益率')
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(accumulative_price_weighted.index,accumulative_price_weighted.iloc[:,-1], color='black', linestyle='-')
plt.xlabel('日期')
plt.ylabel('累计收益率')
plt.show()



# return_monthly_group_pj, nav_group_pj = calculate_returns_and_nav(time_list, celrl_pj, labels_10, price_monthly, group_num=10)
# return_monthly_group_zt, nav_group_zt = calculate_returns_and_nav(time_list, celrl_zt, labels_10, price_monthly, group_num=10)
# return_monthly_group_QoQ_pj, nav_group_QoQ_pj = calculate_returns_and_nav(time_list[4:], celrl_QoQ_pj, labels_10, price_monthly, group_num=10)
# return_monthly_group_QoQ_zt, nav_group_QoQ_zt = calculate_returns_and_nav(time_list[4:], celrl_QoQ_zt, labels_10, price_monthly, group_num=10)
# return_monthly_group_QoQ_abs_pj, nav_group_QoQ_abs_pj = calculate_returns_and_nav(time_list[4:], celrl_QoQ_abs_pj, labels_5, price_monthly, group_num=5)
# return_monthly_group_QoQ_abs_zt, nav_group_QoQ_abs_zt = calculate_returns_and_nav(time_list[4:], celrl_QoQ_abs_zt, labels_3, price_monthly, group_num=3)









# def calculate_weighted_returns_and_nav(time_list, celrl_pj, labels, price_monthly, weights, group_num = 5):
#     return_monthly = []
#     for i in range(1, len(time_list) - 1):
#         return_this_month = []
#         time0 = time_list[i]
#         time1 = time_list[i + 1]
#         celrl_pj_time0 = celrl_pj.loc[time0, stock_available.loc[time0] == True].dropna()
#         celrl_pj_time0_sorted = celrl_pj_time0.sort_values()
#         price_monthly_time1 = price_monthly.loc[time1]
#         weights_time1 = weights.loc[time1]
#         bins = pd.qcut(celrl_pj_time0_sorted, q=group_num, labels=labels)
#         for group in labels:
#             stock_list = bins.loc[bins == group].index
#             price_chosen = price_monthly_time1.loc[stock_list]
#             weights_chosen = weights_time1.loc[stock_list]
#             return_this_month.append(np.average(price_chosen, weights=weights_chosen))
#         return_monthly.append(return_this_month)
#         return_this_month = []
#     return_monthly = np.array(return_monthly)
#     return_monthly_df = pd.DataFrame(index=time_list[2:], columns=labels, data=return_monthly)
#     nav = (1 + return_monthly_df / 100).cumprod(axis=0)
#     return return_monthly_df, nav
#
# return_monthly_group_QoQ_abs_pj_weighted, nav_group_QoQ_abs_pj_weighted = calculate_weighted_returns_and_nav(time_list[4:], celrl_QoQ_abs_pj, labels_5, price_monthly, size, group_num=5)
# return_monthly_group_QoQ_abs_zt_weighted, nav_group_QoQ_abs_zt_weighted = calculate_weighted_returns_and_nav(time_list[4:], celrl_QoQ_abs_zt, labels_5, price_monthly, size, group_num=5)



# def plot_nav(nav, labels, title):
#     """
#     绘制累计收益率图表。
#
#     参数:
#     nav: DataFrame
#         存储各组的净值的数据。
#     labels : list
#         分组标签的列表。
#     title : str
#         图表标题。
#     xlabel : str
#         横坐标名称。
#     ylabel : str
#         纵坐标名称。
#     """
#     for group in labels:
#         plt.plot(nav.index, nav.loc[:, group].values, label=group)
#     plt.legend()
#     plt.title(title)
#     plt.xlabel('日期')
#     plt.ylabel('累计收益率')
#     plt.show()
#
#
#
#
#
# plot_nav(nav_group_QoQ_abs_pj, labels_5, '超额利润率环比绝对值-平均法-等权')
# plot_nav(nav_group_QoQ_abs_zt, labels_3, '超额利润率环比绝对值-整体法-等权')
#
# plot_nav(nav_group_QoQ_abs_pj_weighted, labels_5, '超额利润率环比绝对值-平均法-市值加权')
# plot_nav(nav_group_QoQ_abs_zt_weighted, labels_5, '超额利润率环比绝对值-整体法-市值加权')


