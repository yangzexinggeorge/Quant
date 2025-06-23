import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def portfolio_performance(nav_series, benchmark, period=1, compact=False):
    max_drawdown_series = []
    x = pd.DataFrame(nav_series).iloc[:, 0]
    r = (x - x.shift(1)) / (x.shift(1))
    r = r.fillna(0.0)
    # 年化收益率
    annual_ratio = (x.iloc[len(x) - 1] / x.iloc[0]) ** (240. / period / len(x)) - 1
    # 年化波动率
    annual_volatility = np.sqrt(r.var() * 240 / period)
    # 夏普比率
    sharpe_ratio = (r.mean() / np.sqrt(r.var())) * np.sqrt(240 / period)

    # 最大回撤(滚动时间序列)
    for i in range(len(x)):
        i_max = x.iloc[:i+1].max()
        i_maxdrawdown = -1 * (1 - x[i] / i_max)
        max_drawdown_series.append(i_maxdrawdown)

    max_drawdown_series[0] = 0
    max_drawdown_ts = pd.DataFrame(data=max_drawdown_series, index=nav_series.index, columns=['MAX_DRAWDOWN'])
    max_drawdown = np.array(max_drawdown_series).min()

    # 计算信息比率 ： 主动收益 / 主动风险
    nav_alpha = nav_series.iloc[:,0] / benchmark.iloc[:,0]
    ret_alpha = nav_alpha / nav_alpha.shift(1) - 1#收益率
    ret_alpha = ret_alpha.dropna()
    ret_alpha_yr = ret_alpha.mean() * 240 / period
    vol_alpha_yr = ret_alpha.std() * np.sqrt(240 / period)
    ir = ret_alpha_yr / vol_alpha_yr

    calmar = annual_ratio / (-1) * max_drawdown

    # 滚动1年期持有收益
    rolling_return = []
    for i in range(len(nav_alpha) - 240):
        hpr_1y = ((nav_alpha.iloc[i + 240] / nav_alpha.iloc[i]) - 1)#[0]
        rolling_return.append(hpr_1y)

    rolling_alpha = pd.DataFrame(data=rolling_return, index=nav_series[:-240].index, columns=['ROLLING_1YR_ALPHA'])

    if compact:
        indicator = pd.DataFrame([annual_ratio, annual_volatility, ir, sharpe_ratio, max_drawdown, calmar],
                                 columns=['ls'],
                                 index=['annual_ratio', 'annual_volatility', 'information_ratio',
                                        'sharpe_ratio', 'max_drawdown', 'calmar_ratio'])
        return indicator
    else:
        return annual_ratio, annual_volatility, ir, sharpe_ratio, max_drawdown, calmar, rolling_alpha, max_drawdown_ts


# 获取分年度统计表
def get_annual_stats(year_daily_ret, bench_daily_ret, alpha_ret, return_period):
    # 收益率与波动率均年化
    ret = year_daily_ret.mean() * 240 / return_period
    vol = year_daily_ret.std() * np.sqrt(240 / return_period)
    ret_bench = bench_daily_ret.mean() * 240 / return_period
    ret_alpha = alpha_ret.mean() * 240 / return_period
    vol_alpha = alpha_ret.std() * np.sqrt(240 / return_period)
    rf = 0.025
    # 夏普比率计算
    sharpe = (ret - rf) / vol
    # 信息比率计算
    ir = ret_alpha / vol_alpha
    one_year_nav = (year_daily_ret + 1).cumprod()
    one_year_max = one_year_nav.expanding().max()
    mdd = (1 - one_year_nav / one_year_max).max()
    calmar = ret / mdd
    return [ret, vol, ir, sharpe, (-1) * mdd, calmar]


# 获取整体统计表
def get_total_stats(nav_series, nav_benchmark, return_period):
    # 收益率序列
    ret_series = nav_series.iloc[:,0] / nav_series.iloc[:,0].shift(1) - 1
    ret_series = ret_series.dropna()
    ret_bench_series = nav_benchmark.iloc[:,0] / nav_benchmark.iloc[:,0].shift(1) - 1
    ret_bench_series = ret_bench_series.dropna()
    # 相对净值序列与相对收益率序列
    nav_alpha = nav_series.iloc[:,0] / nav_benchmark.iloc[:,0]
    ret_alpha = nav_alpha / nav_alpha.shift(1) - 1
    ret_alpha = ret_alpha.dropna()
    # 分年处理，调用年化矩阵
    year_bench_ret = ret_bench_series.to_period('A-DEC')
    year_index_ret = ret_series.to_period('A-DEC')
    year_alpha_ret = ret_alpha.to_period('A-DEC')
    year = year_index_ret.index.unique()
    stats_mat = pd.DataFrame(index=year, columns=['return', 'volatility', 'info_ratio', 'sharpe_ratio', 'mdd', 'calmar_ratio'])
    for spot in year:
        if spot in year_alpha_ret.index:
            alpha_ret = year_alpha_ret.loc[spot]
            stats_mat.loc[spot] = get_annual_stats(year_index_ret.loc[spot], year_bench_ret.loc[spot],
                                               alpha_ret, return_period)
    return stats_mat


# 文档输出
def excel_writer(nav_df, relative_nav, rolling_alpha,max_drawdown_ts, result_indicator, yearly_stats, s_date,
                 e_date, stock_pool, strategy_name, weight_method):
    output_dir = 'performance/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = output_dir + '组合绩效_' + strategy_name + '_' + stock_pool + '_' + weight_method + '_' + s_date.strftime(
        "%Y-%m-%d") + '_' + e_date.strftime("%Y-%m-%d") + '.xlsx'
    with pd.ExcelWriter(path) as writer:
        # 将每个 DataFrame 写入到不同的 Sheet 中
        nav_df.to_excel(writer, sheet_name='组合净值')
        relative_nav.to_excel(writer, sheet_name='相对净值')
        rolling_alpha.to_excel(writer, sheet_name='持有一年期超额收益')
        max_drawdown_ts.to_excel(writer, sheet_name='最大回撤')
        result_indicator.to_excel(writer, sheet_name='策略绩效总统计')
        yearly_stats.to_excel(writer, sheet_name='策略绩效分年统计')


# 组合绩效分析主模块
def performance_analysis(nav_df, benchmark_series, stock_pool_name,
                         strategy_name, weight_method):
    start_date = nav_df.index[0]
    end_date = nav_df.index[-1]

    # 基准可调整
    nav_series = nav_df
    benchmark_series = benchmark_series.loc[start_date:end_date]
    benchmark_nav = benchmark_series

    nav_df = pd.DataFrame({'Portfolio': nav_series.values.flatten(), 'Benchmark': benchmark_nav.values.flatten()},index=nav_df.index)

    # 计算相关参数
    annual_ratio, annual_volatility, ir, sharpe_ratio, max_drawdown, calmar, rolling_alpha, max_drawdown_ts \
        = portfolio_performance(nav_series, benchmark_nav, 1, False)
    relative_nav = pd.DataFrame(nav_series.values.flatten() / benchmark_nav.values.flatten(), index=nav_df.index)
    relative_nav.name = 'Relative_NAV'
    result_indicator = pd.DataFrame([annual_ratio, annual_volatility, ir,
                                     sharpe_ratio, max_drawdown, calmar],
                                    columns=[start_date.strftime("%Y-%m-%d") + ' To ' + end_date.strftime("%Y-%m-%d")],
                                    index=['annual_ratio', 'annual_volatility', 'information_ratio',
                                           'sharpe_ratio', 'max_drawdown', 'calmar_ratio'])
    yearly_stats = get_total_stats(nav_series, benchmark_nav, 1)

    # 输出到excel
    excel_writer(nav_df, relative_nav, rolling_alpha, max_drawdown_ts, result_indicator, yearly_stats, start_date,
                 end_date, stock_pool_name, strategy_name, weight_method)


def win_rate_changedate(final_nav,benchmark,change_date):
    daily_nav= final_nav.mean(axis=1)
    weekly_nav = pd.DataFrame(columns=['nav_weekly'])
    weekly_benchmark = pd.DataFrame(columns=['benchmark_weekly'])
    for i in range(len(change_date) - 1):
        nav = daily_nav.loc[change_date[i]:change_date[i + 1]]
        nav = nav.iloc[:-1]
        if len(nav) != 0:
            weekly_nav.loc[change_date[i]] = (1 + nav / 100).cumprod().iloc[-1]
        bench = benchmark.loc[change_date[i]:change_date[i + 1]]
        bench = bench.iloc[:-1]
        if len(bench) != 0:
            weekly_benchmark.loc[change_date[i]] = (1 + bench / 100).cumprod().iloc[-1].values
    return (weekly_nav.values > weekly_benchmark.values).sum()/len(weekly_nav)

def win_rate_daily(final_nav,benchmark):
    return (final_nav.mean(axis=1)>benchmark.iloc[:,0]).sum()/len(final_nav)

def ic(final_result,price,change_date):
    final_result = final_result.loc[final_result.index.isin(change_date)]
    howmany = len(final_result.columns)
    rank_1 = np.arange(1,howmany+1,1)
    delta_price = pd.DataFrame(index=final_result.index[1:],columns=final_result.columns)
    for i in range(len(delta_price.index)):
        for j in range(len(final_result.columns)):
            delta_price.iloc[i,j] = (1+price.loc[final_result.index[i]:final_result.index[i+1],final_result.iloc[i,j]]/100).cumprod()[-1]
    ic_series = pd.DataFrame(index=delta_price.index,columns=['ic'])
    for date in ic_series.index:
        ic_series.loc[date,'ic']=np.corrcoef(rank_1,delta_price.loc[date].rank())[0,1]
    ic = ic_series.mean()
    ir = ic/ic_series.std()
    ic_cumsum = ic_series.cumsum()
    return ic,ir,ic_series,ic_cumsum

def maxdrawdown(nav):
    maxdrawdown = 0
    highpoint_loc = 0
    for i in range(len(nav.index)):
        nav_now = nav.iloc[i].values
        delta = (nav_now-nav.iloc[highpoint_loc].values)/nav.iloc[highpoint_loc].values
        if delta<maxdrawdown:
            maxdrawdown = delta
            maxdrawdown_high = nav.index[highpoint_loc]
            maxdrawdown_low = nav.index[i]
        if nav_now>nav.iloc[highpoint_loc].values:
            highpoint_loc = i
    print("最大回撤："+str(round(maxdrawdown[0]*(-100),2))+'%')
    return maxdrawdown,maxdrawdown_high,maxdrawdown_low,nav.index[highpoint_loc]