import pandas as pd

def target_position_weight(final_result_unstack,preclose,strategy):
    preclose_unstack = preclose.unstack().reset_index()
    preclose_unstack.columns=['股票代码','成分日期','交易价格']
    choice = pd.merge(final_result_unstack,preclose_unstack,on=['成分日期','股票代码'],how='inner',sort=False)
    choice = choice.loc[:, ['成分日期', '持仓权重', '股票代码', '交易价格']]
    choice = choice[['股票代码','持仓权重','交易价格','成分日期']]
    choice.columns = ['证券代码','持仓权重','交易价格','成分日期']
    choice.to_excel(r'..\choice\{}.xlsx'.format(strategy), index=False)


