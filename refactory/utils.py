from copy import copy

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from refactory.data_source import get_point_size, get_spread_cost, get_daily_price, get_instrument_info


def get_volatily(price, span=35, min_periods=10, vol_floor=True,
                 floor_min_quant=0.05, floor_min_periods=100, floor_days=500):
    vol = price.ewm(adjust=True, span=span, min_periods=min_periods).std()
    vol_abs_min = 0.0000000001
    vol[vol < vol_abs_min] = vol_abs_min
    # 给 vol 设最低值
    if vol_floor:
        vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(q=floor_min_quant)
        vol_min.iloc[0] = 0.0
        vol_min.ffill(inplace=True)
        vol = np.maximum(vol, vol_min)
    return vol.ffill()


def ewmac(price, Lfast, Lslow, min_periods=1):
    fast_ewm = price.ewm(span=Lfast, min_periods=min_periods).mean()
    slow_ewm = price.ewm(span=Lslow, min_periods=min_periods).mean()
    raw_ewm = fast_ewm - slow_ewm
    return raw_ewm


def calculate_mixed_volatility(daily_returns, days=35, min_periods=10, slow_vol_years=20,
                               proportion_of_slow_vol=0.3, vol_abs_min=0.0000000001,
                               vol_multiplier=1.0, backfill=False):
    # 长期和短期波动进行权重处理
    vol = daily_returns.ewm(adjust=True, span=days, min_periods=min_periods).std()
    slow_vol_days = slow_vol_years * 256
    long_vol = vol.ewm(adjust=True, span=slow_vol_days).mean()
    vol = proportion_of_slow_vol * long_vol + (1 - proportion_of_slow_vol) * vol
    vol[vol < vol_abs_min] = vol_abs_min
    if backfill:
        vol_forward_fill = vol.ffill()
        vol = vol_forward_fill.bfill()
    vol = vol * vol_multiplier
    return vol


def robust_vol_calc(daily_returns: pd.Series,
    days: int = 35,
    min_periods: int = 10,
    vol_abs_min: float = 0.0000000001,
    vol_floor: bool = True,
    floor_min_quant: float = 0.05,
    floor_min_periods: int = 100,
    floor_days: int = 500,
    backfill: bool = False,):

    vol = daily_returns.ewm(adjust=True, span=days, min_periods=min_periods).std()
    vol[vol < vol_abs_min] = vol_abs_min

    if vol_floor:
        vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(q=floor_min_quant)
        vol_min.iloc[0] = 0.0
        vol_min.ffill(inplace=True)
        vol = np.maximum(vol, vol_min)
    if backfill:
        # use the first vol in the past, sort of cheating
        vol_forward_fill =vol.ffill()
        vol = vol_forward_fill.bfill()
    return vol


def get_stdev_estimator_for_instrument_weight(data_for_analysis, fit_end, span=50000, min_periods=5):
    stdev = data_for_analysis.ewm(span=span, min_periods=min_periods).std()
    last_index = data_for_analysis.index[data_for_analysis.index < fit_end].size - 1
    stdev = stdev.iloc[last_index]
    annualised_stdev_estimate = {}
    for rule_name, std_value in stdev.items():
        annualised_stdev_estimate[rule_name] = std_value * ((365.25 / 7.0) ** 0.5)
    stdev_list = [value for value in annualised_stdev_estimate.values()]
    ave_stdev = np.nanmean(stdev_list)
    norm_stdev = [ave_stdev] * len(stdev_list)
    norm_factor = [stdev / ave_stdev for stdev in stdev_list]
    return norm_stdev, norm_factor


def get_mean_estimator(data, fit_end, span=50000, min_periods=10):
    mean = data.ewm(span=span, min_periods=min_periods).mean()  # 逻辑还是config 的4倍
    last_index = data.index[data.index < fit_end].size - 1
    mean = mean.iloc[last_index]
    annualised_mean_estimate = {}
    for rule_name, mean_value in mean.items():
        annualised_mean_estimate[rule_name] = mean_value * 365.25 / 7.0
    mean_list = [value for value in annualised_mean_estimate.values()]
    return mean_list


def get_corr_estimator_for_instrument_weight(data, fit_end, span=500000, min_periods=10):
    raw_corr = data.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(
        pairwise=True)  # span 和min_periods 都是config 里面的4倍，因为4个instruments
    columns = data.columns
    size_of_matrix = len(columns)
    corr_matrix_values = (raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(
        size_of_matrix).values)  # 截取fit_period之前的数据
    corr_matrix_values = [[max(0, item) for item in sublist] for sublist in corr_matrix_values]
    return corr_matrix_values


def optimisation(number, corr, norm_mean, norm_stdev):
    def addem(weights):
        return 1.0 - sum(weights)

    def neg_SR(weights, sigma, mus):
        estimated_returns = np.dot(weights, mus)[0]
        stdev = weights.dot(sigma).dot(weights.transpose()) ** 0.5
        sr = -estimated_returns / stdev
        return sr

    mus = np.array(norm_mean, ndmin=2).transpose()  # mus 没问题
    sigma = np.diag(norm_stdev).dot(corr).dot(np.diag(norm_stdev))
    start_weights = np.array([1 / number] * number)
    bounds = [(0.0, 1.0)] * number
    cdict = [{"type": "eq", "fun": addem}]
    ans = minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', constraints=cdict, bounds=bounds, tol=0.00001)
    weight = ans['x']
    return weight

def calculate_weighted_average_with_nans(weights, list_of_values, sum_of_weights_should_be = 1.0):
    ## easier to work in np space
    np_weights = np.array(weights)
    np_values = np.array(list_of_values)

    # get safe weights
    weights_times_values_as_np = np_weights * np_values
    empty_weights = np.isnan(weights_times_values_as_np)
    np_weights[empty_weights] = 0.0
    weights_without_nan = copy(np_weights)

    sum_of_values = np.nansum(weights_without_nan)
    renormalise_multiplier = sum_of_weights_should_be / sum_of_values
    normalised_weights = weights_without_nan * renormalise_multiplier

    weights_times_values_as_np = normalised_weights * np_values
    weighted_value = np.nansum(weights_times_values_as_np)

    return weighted_value


def get_cost_per_trade(instrument_code):
    block_price_multiplier = get_point_size(instrument_code)  # 指源代码中 get_value_of_block_price_move 返回的是point_size
    notional_blocks_traded = 1

    price_slippage = get_spread_cost(instrument_code)  # TODO: 验证spread_cost和price_slippage是不是一个事情
    slippage = abs(notional_blocks_traded) * price_slippage * block_price_multiplier

    start_date = get_daily_price(instrument_code).index[-1] - pd.DateOffset(years=1)
    # FIXME: 在这里作者使用了pd.DateOffset来进行年份计算，而在rolling window中是用365天，原因存疑
    average_price = float(get_daily_price(instrument_code)[start_date:].mean())
    price_returns = get_daily_price(instrument_code).diff()    # FIXME: 又重复get了一次价格， 虽然源代码也是这么写的
    daily_vol = calculate_mixed_volatility(price_returns, slow_vol_years=10)  # TODO: 后续看是否完全复用
    average_vol = float(daily_vol[start_date:].mean())
    ann_stdev_price_units = average_vol * 16
    value_per_block = average_price * block_price_multiplier

    per_trade_commission = get_instrument_info(instrument_code).meta_data.PerTrade
    per_block_commission = notional_blocks_traded * get_instrument_info(instrument_code).meta_data.PerBlock
    percentage_commission = (notional_blocks_traded * value_per_block
                             * get_instrument_info(instrument_code).meta_data.Percentage)
    commission = max([per_trade_commission, per_block_commission, percentage_commission])

    cost_instrument_currency = commission + slippage
    ann_stdev_instrument_currency = ann_stdev_price_units * block_price_multiplier
    cost_per_trade = cost_instrument_currency / ann_stdev_instrument_currency
    return cost_per_trade
