import numpy as np
import pandas as pd
from copy import copy

from refactory.data_source import get_daily_price, get_raw_carry_data, get_point_size, get_block_value, \
    get_roll_parameters, get_raw_cost_data, get_spread_cost, get_instrument_info
from refactory.utils import get_volatily, ewmac, calculate_mixed_volatility, get_corr_estimator_for_instrument_weight, \
    get_stdev_estimator_for_instrument_weight, get_mean_estimator, optimisation, calculate_weighted_average_with_nans
from sysdata.config.configdata import Config

my_config = Config()
my_config.instruments = ["CORN", "SOFR", "SP500_micro", 'US10']


def calculate_forecasts(price):
    raw_ewmac8 = ewmac(price, 8, 32, 1)
    ewmac8 = final_forecast('ewmac8', raw_ewmac8, price, 20)
    raw_ewmac32 = ewmac(price, 32, 128, 1)
    ewmac32 = final_forecast('ewmac32', raw_ewmac32, price, 20)
    forecast_df = pd.concat([ewmac8, ewmac32], axis=1)
    forecast_df.index = pd.to_datetime(forecast_df.index)
    return forecast_df


def final_forecast(name, raw_forecast, price, upper_cap=20):
    raw_forecast[raw_forecast == 0] = np.nan

    # TODO:为什么有的用价格波动率，有的用收益率波动率？
    vol = get_volatily(price)
    adjust_forecast = raw_forecast / vol

    scalar = get_forecast_scalar(adjust_forecast)
    scaled_forecast = scalar * adjust_forecast

    lower_cap = -upper_cap
    capped_forecast = scaled_forecast.clip(lower=lower_cap, upper=upper_cap)

    capped_forecast.rename(name, inplace=True)

    return capped_forecast


def get_forecast_scalar(raw_forecast, window=250000, min_period=500, target_abs_forecast=10, backfill=True):
    forecast = copy(raw_forecast)
    forecast = forecast.abs()
    ave_abs_value = forecast.rolling(window=window, min_periods=min_period).mean()
    scaling_factor = target_abs_forecast / ave_abs_value
    if backfill:
        scaling_factor = scaling_factor.bfill()
    return scaling_factor



# FIXME 为何做了两次risk target？
def get_position_target(price, point_size, capital=1000000, risk_target=0.16):
    ret_volatility = calculate_mixed_volatility(price.diff(), slow_vol_years=10)
    daily_risk_target = risk_target / (256 ** 0.5)
    daily_cash_vol_target = daily_risk_target * capital
    position_target = daily_cash_vol_target / (ret_volatility * point_size)
    return position_target


def calculate_pnl(positions: pd.Series, prices: pd.Series):
    pos_series = positions.groupby(positions.index).last()
    both_series = pd.concat([pos_series, prices], axis=1)
    if len(both_series.columns) == 2:
        both_series.columns = ["positions", "prices"]
    both_series = both_series.ffill()
    price_returns = both_series.prices.diff()
    # 源代码在这里计算的时候是shift(1), 可经过对比，发现Position series 事先已经经历过一次shift(1), 所以一共shift(2)
    # FIXME: 解释为什么出现了shift(2)
    adjusted_both_series = both_series.loc[:, both_series.columns != 'price'].shift(2)
    returns = adjusted_both_series.mul(price_returns, axis=0)
    returns[returns.isna()] = 0.0
    return returns


def calculate_factor_pnl(forecast, price, capital, point_size, risk_target):
    position_target = get_position_target(price, point_size, capital, risk_target)
    aligned_ave = position_target.reindex(forecast.index, method='ffill')  # average_notional_position CORN 没问题
    position = forecast.mul(aligned_ave, axis=0) / 10  # notional_position CORN 没问题
    pnl_in_points = calculate_pnl(positions=position, prices=price)
    pnl = pnl_in_points * point_size  # pnl 对CORN 经过shift(2) 后没问题
    daily_pnl = pnl.resample("B").sum()
    return daily_pnl


def get_capped_forecast(instrument_code, rule_name):
    price = get_daily_price(instrument_code)
    if rule_name == 'ewmac32':
        raw_ewmac32 = ewmac(price, 32, 128, 1)
        ewmac32 = final_forecast('ewmac32', raw_ewmac32, price, 20)
        return ewmac32
    if rule_name == 'ewmac8':
        raw_ewmac8 = ewmac(price, 8, 32, 1)
        ewmac8 = final_forecast('ewmac8', raw_ewmac8, price, 20)
        return ewmac8
    else:
        raise 'Rule not defined '

def forecast_turnover_for_individual_instrument(instrument_code, rule_name):
    forecast = get_capped_forecast(instrument_code, rule_name)

    average_forecast_for_turnover = 10.0
    y = average_forecast_for_turnover
    daily_forecast = forecast.resample("1B").last()
    daily_y = pd.Series(np.full(daily_forecast.shape[0], float(y)), daily_forecast.index)
    x_normalised_for_y = daily_forecast / daily_y.ffill()
    avg_daily = float(x_normalised_for_y.diff().abs().mean())
    annual_turnover_for_forecast = avg_daily * 256
    return annual_turnover_for_forecast


def get_SR_cost_for_instrument_forecast(instrument_code, rule_name):
    pooled_instruments = my_config.instruments

    turnovers = [forecast_turnover_for_individual_instrument(instrument_code, rule_name)
                 for instrument_code in pooled_instruments]

    forecast_lengths = [len(get_capped_forecast(instrument_code, rule_name))
                        for instrument_code in pooled_instruments]
    total_length = float(sum(forecast_lengths))
    weights = [forecast_length / total_length for forecast_length in forecast_lengths]

    avg_turnover = calculate_weighted_average_with_nans(weights, turnovers)


    # holding costs calculated elsewhere

    cost_per_trade = get_cost_per_trade(instrument_code)
    transaction_cost = cost_per_trade * avg_turnover
    # holding cost
    roll_parameters = get_roll_parameters(instrument_code)
    hold_turnovers = roll_parameters.rolls_per_year_in_hold_cycle() * 2.0
    holding_cost = hold_turnovers * cost_per_trade  # Holding cost is verified

    trading_cost = transaction_cost + holding_cost
    return trading_cost


def get_cost_per_trade(instrument_code):
    block_price_multiplier = get_point_size(instrument_code)  # 指源代码中 get_value_of_block_price_move 返回的是point_size
    notional_blocks_traded = 1
    price = get_daily_price(instrument_code)  # FIXME: 又重复get了一次价格， 虽然源代码也是这么写的
    last_date = price.index[-1]
    # FIXME: 在这里作者使用了pd.DateOffset来进行年份计算，而在rolling window中是用365天，原因存疑
    start_date = last_date - pd.DateOffset(years=1)
    average_price = float(price[start_date:].mean())
    price_returns = price.diff()
    daily_vol = calculate_mixed_volatility(price_returns, slow_vol_years=10)  # TODO: 后续看是否完全复用
    average_vol = float(daily_vol[start_date:].mean())
    ann_stdev_price_units = average_vol * 16
    value_per_block = average_price * block_price_multiplier
    price_slippage = get_spread_cost(instrument_code)  # TODO: 验证spread_cost和price_slippage是不是一个事情
    slippage = abs(notional_blocks_traded) * price_slippage * block_price_multiplier
    per_trade_commission = get_instrument_info(instrument_code).meta_data.PerTrade
    per_block_commission = notional_blocks_traded * get_instrument_info(instrument_code).meta_data.PerBlock
    percentage_commission = (notional_blocks_traded * value_per_block
                             * get_instrument_info(instrument_code).meta_data.Percentage)
    commission = max([per_trade_commission, per_block_commission, percentage_commission])
    cost_instrument_currency = commission + slippage
    ann_stdev_instrument_currency = ann_stdev_price_units * block_price_multiplier
    cost_per_trade = cost_instrument_currency / ann_stdev_instrument_currency
    return cost_per_trade


SR_COST = get_SR_cost_for_instrument_forecast('US10', 'ewmac32')
# 先算turnover
# 再算SR cost
from systems.accounts.account_costs import accountCosts
accountCosts = accountCosts()
# sr_cost = accountCosts._get_SR_transaction_cost_of_rule_for_individual_instrument("CORN", 'ewmac8')

price = get_daily_price("CORN")
forecast = calculate_forecasts(price)
capital = 1000000
point_size = get_point_size("CORN")
risk_target = 0.16
daily_pnl = calculate_factor_pnl(forecast, price, capital, point_size, risk_target)
print('end')
def generate_end_list(start_date, end_date):
    start_dates_per_period = pd.date_range(end_date, start_date, freq='-365D').to_list()
    start_dates_per_period.reverse()
    end_list = start_dates_per_period[1:-1]
    return end_list


def calculate_forecast_weights(pnl_df, fit_end):
    number = len(pnl_df.columns)
    span = len(my_config.instruments) * 50000
    min_periods_corr = len(my_config.instruments) * 10
    min_periods = len(my_config.instruments) * 5
    norm_stdev, norm_factor = get_stdev_estimator_for_instrument_weight(pnl_df, fit_end, span, min_periods)
    mean_list = get_mean_estimator(pnl_df, fit_end, span, min_periods)
    norm_mean = [a / b for a, b in zip(mean_list, norm_factor)]
    corr = get_corr_estimator_for_instrument_weight(pnl_df, fit_end, span, min_periods_corr)
    weight = optimisation(number, corr, norm_mean, norm_stdev)
    return weight


def combine_instrument_pnl_df(weekly_ret):
    from itertools import chain
    all_indices_flattened = list(chain.from_iterable(data_item.index for data_item in weekly_ret))
    common_unique_index = sorted(set(all_indices_flattened))
    data_reindexed = [data_item.reindex(common_unique_index) for data_item in weekly_ret]
    for offset_value, data_item in enumerate(data_reindexed):
        data_item.index = data_item.index + pd.Timedelta("%dus" % offset_value)
    stacked_data = pd.concat(data_reindexed, axis=0)
    stacked_data = stacked_data.sort_index()
    return stacked_data


def calculate_forecast_diversify_multiplier(forecasts, forecast_weights):
    weekly_forecast = forecasts.resample('W').last()
    fit_end_list = generate_end_list(weekly_forecast.index[0], weekly_forecast.index[-1])

    full_corr = weekly_forecast.ewm(span=250, min_periods=20, ignore_na=True).corr(pairwise=True)
    size_of_matrix = len(weekly_forecast.columns)
    dm_list = []
    for fit_end in fit_end_list:
        corr = full_corr[full_corr.index.get_level_values(0) < fit_end].tail(size_of_matrix).values
        w = forecast_weights[:fit_end].iloc[-1].values
        variance = w.dot(corr).dot(w.transpose())
        dm = np.min([1 / variance ** 0.5, 2.5])
        dm_list.append(dm)

    dm_yearly = pd.Series(dm_list, index=fit_end_list)
    dm_daily = dm_yearly.reindex(forecast_weights.index, method='ffill')
    dm_daily[dm_daily.isna()] = 1.0
    dm_smoonth = dm_daily.ewm(span=125).mean()
    return dm_smoonth

def calculate_instrument_weights(pnl_df):
    daily_pnl = pnl_df.resample("1B").sum()
    daily_pnl[daily_pnl == 0.0] = np.nan

    number = len(daily_pnl.columns)
    weekly_ret = daily_pnl.resample('W').sum()  # SP500_micro 的一些数值不对，其他的都能对的上。怀疑是不是一些nan被填充了
    fit_end = weekly_ret.index[-1]
    span = 500000
    min_periods = 10

    norm_stdev, _ = get_stdev_estimator_for_instrument_weight(weekly_ret, fit_end, span, min_periods)
    norm_mean = [0.5 * asset_stdev for asset_stdev in norm_stdev]
    corr = get_corr_estimator_for_instrument_weight(weekly_ret, fit_end, span, min_periods)

    weight = optimisation(number, corr, norm_mean, norm_stdev)
    return weight


def calculate_volatility_scalar(instrument_code, capital=1000000, annual_percentage_volatility_target=0.16):
    block_value = get_block_value(instrument_code)
    block_value.ffill(inplace=True)
    # FIXME: 取错数据了
    price = get_raw_carry_data(instrument_code)
    price.ffill(inplace=True)
    price0 = get_daily_price(instrument_code)
    diff_volatility = calculate_mixed_volatility(price0.diff(), slow_vol_years=10)
    diff_volatility.ffill(inplace=True)

    percentage_volatility = 100.0 * (diff_volatility / price.abs())
    currency_volatility = block_value * percentage_volatility
    value_volatiliity = currency_volatility * 1

    pecentage_volatility_target = annual_percentage_volatility_target / 16
    cash_volatility_target = capital * pecentage_volatility_target

    volatility_scalar = cash_volatility_target / value_volatiliity

    return volatility_scalar




def apply_buffer(position_raw, volatility_scalar, buffer_size):
    buffer = volatility_scalar * buffer_size
    top_pos = (position_raw + buffer).round()
    bottom_pos = (position_raw - buffer).round()
    position_raw = position_raw.round()

    last = position_raw.values[0]
    buffered_position_list = [last]
    for index in range(len(position_raw))[1:]:
        last = adjust_by_buffer(last, position_raw.values[index],
                                top_pos.values[index], bottom_pos.values[index])
        buffered_position_list.append(last)
    buffered_position = pd.Series(buffered_position_list, index=position_raw.index)
    # last = position_raw.shift(1).bfill()
    # df = pd.DataFrame({'last': last, 'current': position_raw, 'top': top_pos, 'bottom': bottom_pos})
    # buffered_position = df.apply(lambda x: adjust_by_buffer(x['last'], x['current'], x['top'], x['bottom']), axis=1)

    return buffered_position


def adjust_by_buffer(last, current, top, bottom, trade_to_edge=True):
    result = last
    if trade_to_edge:
        if last > top:
            result = top
        elif last < bottom:
            result = bottom
    else:
        if last > top or last < bottom:
            result = current
    return result


def calculate_instrument_pnl(instrument, position_buffered, price):
    pnl_in_points = calculate_pnl(positions=position_buffered, prices=(price))
    point_size = get_point_size(instrument)
    pnl_in_ccy = pnl_in_points * point_size
    pnl_daily = pnl_in_ccy.resample("B").sum()
    return pnl_daily


############################################################################################# 分割线

def process_forecast_pnls(instrument_code, capital=1000000, risk_target=0.16, target_abs_forecast=10):
    price = get_daily_price(instrument_code)
    forecast_df = calculate_forecasts(price)
    forecast_df = forecast_df / target_abs_forecast

    point_size = get_point_size(instrument_code)
    func_pnl = lambda forecast: calculate_factor_pnl(forecast, price, capital, point_size, risk_target)
    daily_forecast_pnls = forecast_df.apply(func_pnl, axis=0)
    daily_forecast_pnls[daily_forecast_pnls == 0.0] = np.nan
    return daily_forecast_pnls


def process_instrument_pnl(instrument):
    price = get_daily_price(instrument)
    forecast_df = calculate_forecasts(price)

    instruments = my_config.instruments
    daily_forecast_pnls = [process_forecast_pnls(it) for it in instruments]
    weekly_forecast_pnls = [p.resample('W').sum() for p in daily_forecast_pnls]
    returns = combine_instrument_pnl_df(weekly_forecast_pnls)
    start_date = returns.index[0]
    end_date = returns.index[-1]
    end_list = generate_end_list(start_date, end_date)
    weight_df = pd.DataFrame([calculate_forecast_weights(returns, end) for end in end_list], index=end_list)
    weight_df = weight_df.reindex(forecast_df.index, method='ffill')
    weight_df = weight_df.fillna(1 / len(weight_df.columns))
    forecast_weights = weight_df.ewm(span=125).mean()
    forecast_weights.columns = forecast_df.columns
    # forecast_weights.rename(columns={0: 'ewmac32', 1: 'ewmac8'}, inplace=True)
    # forecast_weights = forecast_weights[['ewmac8', 'ewmac32']]

    fdm = calculate_forecast_diversify_multiplier(forecast_df, forecast_weights)
    combined_forecast = (forecast_weights * forecast_df).sum(axis=1) * fdm
    final_forecast = combined_forecast.clip(20, -20)
    volatility_scalar = calculate_volatility_scalar(instrument)
    position_raw = volatility_scalar * final_forecast / 10.0
    position_raw.fillna(0.0, inplace=True)
    position_buffered = apply_buffer(position_raw, volatility_scalar, 0.10)
    pnl_daily = calculate_instrument_pnl(instrument, position_buffered, price)

    return pnl_daily


def main(my_config):
    instruments = my_config.instruments

    pnl_list = [process_instrument_pnl(instrument) for instrument in instruments]
    pnl_df = pd.concat(pnl_list, axis=1)
    pnl_df.columns = instruments

    weight = calculate_instrument_weights(pnl_df)
    print(weight)

    return


if __name__ == '__main__':
    main(my_config)