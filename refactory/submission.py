import numpy as np
import pandas as pd
from copy import copy
from scipy.optimize import minimize

from refactory.data_source import get_daily_price, get_raw_carry_data, get_block_move_price
from refactory.utils import get_volatily, ewmac, calculate_mixed_volatility, get_corr_estimator_for_instrument_weight, \
    get_stdev_estimator_for_instrument_weight, get_stdev_estimator, get_mean_estimator, get_corr_estimator
from sysdata.config.configdata import Config
from sysquant.fitting_dates import fitDates, listOfFittingDates

my_config = Config()
my_config.instruments = ["CORN", "SOFR", "SP500_micro", 'US10']


def get_returns_for_optimisation(instrument_code, capital=1000000, risk_target=0.16,
                                 target_abs_forecast=10):
    forecast_df = calculate_forecasts(instrument_code)
    forecast_names = forecast_df.columns

    price = get_daily_price(instrument_code)
    ret_volatility = calculate_mixed_volatility(price.diff(), slow_vol_years=10)

    data_as_list = []
    for name in forecast_names:
        forecast = forecast_df[name]

        daily_risk_target = risk_target / (256 ** 0.5)
        daily_cash_vol_target = daily_risk_target * capital
        block_move_price = get_block_move_price(instrument_code)
        ave_notional_position = daily_cash_vol_target / (ret_volatility * block_move_price)

        normalised_forecast = forecast / target_abs_forecast
        aligned_ave = ave_notional_position.reindex(normalised_forecast.index, method='ffill')
        notional_position = aligned_ave * normalised_forecast

        pandl_in_points = calculate_pandl(positions=notional_position, prices=price)
        as_pd_series = pandl_in_points * block_move_price
        as_pd_series.index = pd.to_datetime(as_pd_series.index)
        curve = as_pd_series.resample("B").sum()

        data_as_list.append(curve)

    curve_df = pd.concat(data_as_list, axis=1)
    curve_df.columns = forecast_names

    daily_curve = curve_df.resample('1B').sum()
    daily_curve[daily_curve == 0.0] = np.nan

    return daily_curve


def calculate_forecasts(instrument):

    price = get_daily_price(instrument)

    raw_ewmac8 = ewmac(price, 8, 32, 1)
    forecast8 = final_forecast(raw_ewmac8, price, 20)
    forecast8 = forecast8.rename('ewmac8')

    forecast = ewmac(price, 32, 128, 1)
    forecast32 = final_forecast(forecast, price, 20)
    forecast32 = forecast32.rename('ewmac32')
    forecast_df = pd.concat([forecast8, forecast32], axis=1)
    forecast_df.index = pd.to_datetime(forecast_df.index)
    return forecast_df


def final_forecast(raw_forecast, price, upper_cap):
    raw_forecast[raw_forecast == 0] = np.nan

    vol = get_volatily(price)
    adjust_forecast = raw_forecast / vol

    scalar = get_forecast_scalar(adjust_forecast)
    scaled_forecast = scalar * adjust_forecast

    lower_cap = -upper_cap
    capped_forecast = scaled_forecast.clip(lower=lower_cap, upper=upper_cap)

    return capped_forecast


def get_forecast_scalar(raw_forecast, window=250000, min_period=500, target_abs_forecast=10, backfill=True):
    forecast = copy(raw_forecast)
    forecast = forecast.abs()
    ave_abs_value = forecast.rolling(window=window, min_periods=min_period).mean()
    scaling_factor = target_abs_forecast / ave_abs_value
    if backfill:
        scaling_factor = scaling_factor.bfill()
    return scaling_factor


def get_div_mult(instrument_code, weights):
    forecast_df = calculate_forecasts(instrument_code)

    weekly_index = pd.date_range(start=forecast_df.index[0], end='2023-09-03', freq='W')  # 结束日期需要自行根据品种设定
    weekly_forecast = forecast_df.reindex(weekly_index, method='ffill')
    start_date = weekly_forecast.index[0]
    end_date = weekly_forecast.index[-1]

    # 根据数据起始和结束日期，生成各period的开始日期，且从后朝前看
    start_dates_per_period = list(pd.date_range(end_date, start_date, freq='-' + '365D'))
    start_dates_per_period.reverse()
    # Rolling 方法
    periods = []
    fit_end_list = []
    for periods_index in range(len(start_dates_per_period))[1:-1]:
        period_start = start_dates_per_period[periods_index]
        period_end = start_dates_per_period[periods_index + 1]
        fit_start = start_date
        end = start_dates_per_period[periods_index]  # 之前的fit end 是 period end, 所以就造成了fit 和 use 的一部分重叠
        fit_end_list.append(end)
        fit_date = fitDates(fit_start, end, period_start, period_end)
        periods.append(fit_date)

    weights.index = forecast_df.index
    weights = weights.rename(columns={0: 'ewmac32', 1: 'ewmac8'})
    corr_list = []
    for i in range(len(fit_end_list)):
        fit_end = fit_end_list[i]
        raw_corr = weekly_forecast.ewm(span=250, min_periods=20, ignore_na=True).corr(pairwise=True)
        columns = weekly_forecast.columns
        size_of_matrix = len(columns)
        corr_matrix_values = (raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(
            size_of_matrix).values)
        corr_list.append(corr_matrix_values)

    div_mult = []
    for corrmatrix, start_of_period in zip(corr_list, fit_end_list):
        weight_slice = weights[:start_of_period]
        weight_np = np.array(weight_slice.iloc[-1])
        variance = weight_np.dot(corrmatrix).dot(weight_np.transpose())
        risk = variance ** 0.5
        dm = np.min([1 / risk, 2.5])
        div_mult.append(dm)
    div_mult_df = pd.Series(div_mult, index=(fit_end_list))
    div_mult_df_daily = div_mult_df.reindex(weights.index, method='ffill')
    div_mult_df_daily[div_mult_df_daily.isna()] = 1.0
    div_mult_df_smoothed = div_mult_df_daily.ewm(span=125).mean()

    return div_mult_df_smoothed


############################################################################################# 分割线


def get_net_return():
    gross_returns_dict = {}
    for instrument1 in my_config.instruments:
        gross_returns_dict[instrument1] = get_returns_for_optimisation(instrument1)

    ret_df_list = gross_returns_dict.values()

    weekly_ret = [item.resample('W').sum() for item in ret_df_list]

    from itertools import chain
    all_indices_flattened = list(chain.from_iterable(data_item.index for data_item in weekly_ret))
    common_unique_index = sorted(set(all_indices_flattened))
    data_reindexed = [data_item.reindex(common_unique_index) for data_item in weekly_ret]

    for offset_value, data_item in enumerate(data_reindexed):
        data_item.index = data_item.index + pd.Timedelta("%dus" % offset_value)

    stacked_data = pd.concat(data_reindexed, axis=0)
    stacked_data = stacked_data.sort_index()

    return stacked_data


def process_instrument_pnl(instrument):
    forecast_df = calculate_forecasts(instrument)

    returns = get_net_return()

    start_date = returns.index[0]
    end_date = returns.index[-1]
    # 根据数据起始和结束日期，生成各period的开始日期，且从后朝前看
    start_dates_per_period = list(pd.date_range(end_date, start_date, freq='-' + '365D'))
    start_dates_per_period.reverse()
    # Rolling 方法
    periods = []
    end_list = []
    for periods_index in range(len(start_dates_per_period))[1:-1]:
        period_start = start_dates_per_period[periods_index]
        period_end = start_dates_per_period[periods_index + 1]
        fit_start = start_date
        fit_end = start_dates_per_period[periods_index]
        end_list.append(fit_end)
        fit_date = fitDates(fit_start, fit_end, period_start, period_end)
        periods.append(fit_date)
    periods = [fitDates(start_date, start_date, start_date, start_dates_per_period[1], no_data=True)] + periods
    fit_period = listOfFittingDates(periods)
    fit_end_list = end_list
    weight_list = []
    for i in range(len(fit_period) - 1):
        weight = get_weight(returns, i, fit_end_list)
        weight_list.append(weight)
    weights1 = pd.Series(weight_list)
    weights1.index = fit_end_list
    weights1 = weights1.reindex(forecast_df.index, method='ffill')

    replace_nan = lambda x: [0.5, 0.5] if isinstance(x, float) else x
    weights1 = weights1.apply(replace_nan)
    weight_df = pd.DataFrame(weights1.tolist())

    smoothed_weights = weight_df.ewm(span=125).mean()
    smoothed_weights.rename(columns={0: 'ewmac32', 1: 'ewmac8'}, inplace=True)
    weights = smoothed_weights[['ewmac8', 'ewmac32']]

    weights.index = forecast_df.index
    forecast_div_multiplier = get_div_mult(instrument, weights)
    combined_forecast = (weights * forecast_df).sum(axis=1) * forecast_div_multiplier
    capped_combined_forecast = combined_forecast.clip(20, -20)

    avg_position = calculate_avg_position(instrument)

    avg_position = avg_position.reindex(capped_combined_forecast.index, method='ffill')
    subsystem_position_raw = avg_position * capped_combined_forecast / 10.0

    buffered_position = apply_buffered_position(instrument, subsystem_position_raw)

    daily_pnl = calcuate_instrument_pnl(instrument, buffered_position)

    return daily_pnl


def main(my_config):
    instruments = my_config.instruments

    pnl_list = [process_instrument_pnl(instrument) for instrument in instruments]
    pnl_df = pd.concat(pnl_list, axis=1)
    pnl_df.columns = instruments

    weight = caculate_instrument_weights(pnl_df)
    print(weight)

    return


def get_weight(data_for_analysis, date_period, fit_end_list, strategy_number=2):
    fit_end = fit_end_list[date_period]

    span = len(my_config.instruments) * 50000
    min_periods_corr = len(my_config.instruments) * 10
    min_periods = len(my_config.instruments) * 5
    stdev_list = get_stdev_estimator(data_for_analysis, fit_end, span, min_periods)
    ave_stdev = np.nanmean(stdev_list)
    norm_stdev = [ave_stdev] * strategy_number

    norm_factor = [stdev / ave_stdev for stdev in stdev_list]
    mean_list = get_mean_estimator(data_for_analysis, fit_end, span, min_periods)  # mean list 没问题
    norm_mean = [a / b for a, b in zip(mean_list, norm_factor)]

    corr_matrix = get_corr_estimator(data_for_analysis, fit_end, span, min_periods_corr)  # corr matrix 没问题

    sigma = np.diag(norm_stdev).dot(corr_matrix).dot(np.diag(norm_stdev))  # sigma 没问题
    mus = np.array(norm_mean, ndmin=2).transpose()  # mus 没问题
    start_weights = np.array([1 / strategy_number] * strategy_number)
    bounds = [(0.0, 1.0)] * strategy_number

    cdict = [{"type": "eq", "fun": addem}]
    ans = minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', constraints=cdict, bounds=bounds, tol=0.00001)
    weight = ans['x']
    return weight


def calculate_avg_position(instrument_code, capital=1000000, perc_vol_target=16):
    block_move_price = get_block_move_price(instrument_code)
    block_value = get_block_value(instrument_code, block_move_price)

    daily_carry_price = get_raw_carry_data(instrument_code)
    price = get_daily_price(instrument_code)
    daily_returns = price.diff()
    vol = calculate_mixed_volatility(daily_returns, slow_vol_years=10)
    (daily_carry_price, vol) = daily_carry_price.align(vol, join='right')
    perc_vol = 100.0 * (vol / daily_carry_price.ffill().abs())

    instr_currency_vol = block_value.ffill() * perc_vol
    instr_value_vol = instr_currency_vol.ffill()

    annual_cash_vol_target = capital * perc_vol_target / 100
    daily_cash_vol_target = annual_cash_vol_target / 16
    vol_scalar = daily_cash_vol_target / instr_value_vol
    return vol_scalar


def apply_buffered_position(instrument, subsystem_position):
    vol_scalar = calculate_avg_position(instrument)
    vol_scalar = vol_scalar.reindex(subsystem_position.index).ffill()
    avg_position = vol_scalar * 1.0 * 1.0  # 乘的是instr weight和idm, 目前为default 1
    buffer_size = 0.10
    buffer = avg_position * buffer_size
    top_pos = subsystem_position.ffill() + buffer.ffill()
    bottom_pos = subsystem_position.ffill() - buffer.ffill()
    subsystem_position = subsystem_position.round()
    top_pos = top_pos.round()
    bottom_pos = bottom_pos.round()
    current_position = subsystem_position.values[0]
    if np.isnan(current_position):
        current_position = 0.0
    buffered_position_list = [current_position]
    for index in range(len(subsystem_position))[1:]:
        current_position = apply_buffer_for_single_period(current_position, subsystem_position.values[index],
                                                          top_pos.values[index], bottom_pos.values[index])
        buffered_position_list.append(current_position)
    buffered_position = pd.Series(buffered_position_list, index=subsystem_position.index)
    return buffered_position


def apply_buffer_for_single_period(last_position, optimal_position, top_pos, bot_pos, trade_to_edge=True):
    if last_position > top_pos:
        return top_pos if trade_to_edge else optimal_position
    elif last_position < bot_pos:
        return bot_pos if trade_to_edge else optimal_position
    else:
        return last_position


def calcuate_instrument_pnl(instrument, position):
    price = get_daily_price(instrument)
    block_move_price = get_block_move_price(instrument)
    fx = pd.Series(1.0, index=price.index)
    pnl_in_points = calculate_pandl(positions=position, prices=price)
    pnl_in_ccy = pnl_in_points * block_move_price
    fx_aligned = fx.reindex(pnl_in_ccy.index, method="ffill")
    pnl = pnl_in_ccy * fx_aligned
    pnl.index = pd.to_datetime(pnl.index)
    daily_pnl = pnl.resample("B").sum()
    return daily_pnl


def caculate_instrument_weights(pnl_df):
    daily_pnl = pnl_df.resample("1B").sum()
    daily_pnl[daily_pnl == 0.0] = np.nan

    instrument_number = len(daily_pnl.columns)
    weekly_ret = daily_pnl.resample('W').sum()  # SP500_micro 的一些数值不对，其他的都能对的上。怀疑是不是一些nan被填充了
    fit_end = weekly_ret.index[-1]
    norm_stdev, _ = get_stdev_estimator_for_instrument_weight(weekly_ret, fit_end)
    norm_mean = [0.5 * asset_stdev for asset_stdev in norm_stdev]
    mus = np.array(norm_mean, ndmin=2).transpose()  # mus 没问题
    corr = get_corr_estimator_for_instrument_weight(weekly_ret, fit_end)
    sigma = np.diag(norm_stdev).dot(corr).dot(np.diag(norm_stdev))
    start_weights = np.array([1 / instrument_number] * instrument_number)
    bounds = [(0.0, 1.0)] * instrument_number
    cdict = [{"type": "eq", "fun": addem}]
    ans = minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', constraints=cdict, bounds=bounds, tol=0.00001)
    weight = ans['x']
    return weight


############################################################################################# 分割线


def addem(weights):
    return 1.0 - sum(weights)


def neg_SR(weights, sigma, mus):
    estimated_returns = np.dot(weights, mus)[0]
    stdev = weights.dot(sigma).dot(weights.transpose()) ** 0.5
    sr = -estimated_returns / stdev
    return sr


############################################################################################# 分割线


def calculate_pandl(positions: pd.Series, prices: pd.Series):
    pos_series = positions.groupby(positions.index).last()
    both_series = pd.concat([pos_series, prices], axis=1)
    both_series.columns = ["positions", "prices"]
    both_series = both_series.ffill()
    price_returns = both_series.prices.diff()
    returns = both_series.positions.shift(1) * price_returns
    returns[returns.isna()] = 0.0
    return returns


#################################################################################################
# FIXME: 逻辑有误
#################################################################################################

def get_block_value(instrument_code, block_move_price):
    daily_carry_price = get_raw_carry_data(instrument_code)
    block_value = daily_carry_price.ffill() * block_move_price * 0.01
    return block_value


if __name__ == '__main__':
    main(my_config)
