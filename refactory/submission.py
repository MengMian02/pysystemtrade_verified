import numpy as np
import pandas as pd
from datetime import datetime

from refactory.data_source import get_instrument_info, get_daily_price, get_spread_cost, get_roll_parameters, \
    get_raw_carry_data
from sysdata.config.configdata import Config
from sysquant.returns import dictOfReturnsForOptimisation
from sysquant.returns import dictOfReturnsForOptimisationWithCosts
from systems.accounts.curves.account_curve import accountCurve
from systems.accounts.curves.account_curve_group import accountCurveGroup
from systems.accounts.curves.dict_of_account_curves import dictOfAccountCurves
from systems.accounts.pandl_calculators.pandl_SR_cost import pandlCalculationWithSRCosts
from systems.accounts.pandl_calculators.pandl_generic_costs import GROSS_CURVE


# 传入品种代码
def raw_price_and_vol(instrument_code, span=35, min_periods=10, vol_floor=True,
                      floor_min_quant=0.05, floor_min_periods=100, floor_days=500):
    price = get_daily_price(instrument_code)
    vol = price.ewm(adjust=True, span=span, min_periods=min_periods).std()
    vol_abs_min: float = 0.0000000001
    vol[vol < vol_abs_min] = vol_abs_min
    # 给 vol 设最低值
    if vol_floor:
        vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(q=floor_min_quant)
        vol_min.iloc[0] = 0.0
        vol_min.ffill(inplace=True)
        vol = np.maximum(vol, vol_min)
    return price, vol


# 策略
def ewmac_forecast(instrument_code, Lfast, Lslow, min_periods=1):
    price, vol = raw_price_and_vol(instrument_code)
    fast_ewm = price.ewm(span=Lfast, min_periods=min_periods).mean()
    slow_ewm = price.ewm(span=Lslow, min_periods=min_periods).mean()
    raw_ewm = fast_ewm - slow_ewm
    raw_forecast = raw_ewm / vol.ffill()
    raw_forecast[raw_forecast == 0] = np.nan
    return raw_forecast


# Forecast scalar
from copy import copy


def get_forecast_scalar(instrument_code, Lfast, Lslow, window=250000,
                        min_period=500, target_abs_forecast=10, backfill=True):
    forecast = ewmac_forecast(instrument_code, Lfast, Lslow)
    forecast_copy = copy(forecast)
    forecast_copy[forecast_copy == 0.0] = np.nan
    forecast_copy = forecast_copy.abs()
    ave_abs_value = forecast_copy.rolling(window=window, min_periods=min_period).mean()
    scaling_factor = target_abs_forecast / ave_abs_value
    if backfill:
        scaling_factor = scaling_factor.bfill()
    return scaling_factor


def get_capped_forecast(instrument_code, Lfast, Lslow, upper_cap=20):
    # Return the capped and scaled forecast
    scaled_forecast = (get_forecast_scalar(instrument_code, Lfast, Lslow)
                       * ewmac_forecast(instrument_code, Lfast, Lslow))
    lower_cap = -upper_cap
    capped_scaled_forecast = scaled_forecast.clip(lower=lower_cap, upper=upper_cap)
    return capped_scaled_forecast


# 算每天的收益
def get_daily_returns(instrument_code):
    price = get_daily_price(instrument_code)
    daily_returns = price.diff()
    return daily_returns


# 长期和短期波动进行权重处理
def mixed_vol_calc(daily_returns, days=35, min_periods=10, slow_vol_years=20,
                   proportion_of_slow_vol=0.3, vol_abs_min=0.0000000001,
                   vol_multiplier=1.0, backfill=False):
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


############################################################################################# 分割线


# 在设下起始资金量，目标波动的情况下，计算每天收益
def pandl_for_instrument_forecast(forecast, capital, risk_target,
                                  daily_returns_volatility, target_abs_forecast,
                                  instrument_code):
    normalised_forecast = forecast / target_abs_forecast

    daily_risk_target = risk_target / (256 ** 0.5)
    daily_cash_vol_target = daily_risk_target * capital

    instr_object = get_instrument_info(instrument_code)  # 基础品种信息
    block_move_price = instr_object.meta_data.Pointsize
    instr_currency_vol = daily_returns_volatility * block_move_price
    ave_notional_position = daily_cash_vol_target / instr_currency_vol

    aligned_ave = ave_notional_position.reindex(normalised_forecast.index, method='ffill')
    notional_position = aligned_ave * normalised_forecast
    return notional_position, ave_notional_position, block_move_price


my_config = Config()
my_config.instruments = ["CORN", "SOFR", "SP500_micro", 'US10']


def get_pandl_forecast():
    pandl_forecast = {}
    for instrument in my_config.instruments:
        pandl = get_returns_for_optimisation(instrument)
        pandl_forecast[instrument] = pandl
    pandl_forecast = dictOfReturnsForOptimisationWithCosts(pandl_forecast)
    return pandl_forecast


def get_returns_for_optimisation(instrument_code, capital=1000000, risk_target=0.16,
                                 target_abs_forecast=10):
    price, vol = raw_price_and_vol(instrument_code)  # price 也没有问题
    daily_returns = get_daily_returns(instrument_code)
    daily_returns_volatility = mixed_vol_calc(daily_returns, slow_vol_years=10)  # daily returns vol 没有问题
    start_date = pd.to_datetime(daily_returns_volatility.index[0])
    end_date = datetime.now()
    bdate_range = pd.bdate_range(start_date, end_date)

    fx = pd.Series(1.0, index=bdate_range)  # 虽然没有任何影响，但不能删，在作者定义的类中需要用到, fx 没有问题

    dict_of_pandl_by_rule = {}
    forecast = get_forecast_as_list(instrument_code)
    forecast_name = ['ewmac32', 'ewmac8']
    i = 0
    for forecast in forecast:
        notional_position, ave_notional_position, value_per_point = (
            pandl_for_instrument_forecast(forecast=forecast,
                                          capital=capital,
                                          risk_target=risk_target,
                                          daily_returns_volatility=daily_returns_volatility,
                                          target_abs_forecast=target_abs_forecast,
                                          instrument_code=instrument_code))
        pandl_calculator = pandlCalculationWithSRCosts(price,
                                                       SR_cost=0,
                                                       positions=notional_position,
                                                       fx=fx,
                                                       daily_returns_volatility=daily_returns_volatility,
                                                       average_position=ave_notional_position,
                                                       capital=capital,
                                                       value_per_point=value_per_point,
                                                       delayfill=True)
        account_curve = accountCurve(pandl_calculator)
        dict_of_pandl_by_rule[forecast_name[i]] = account_curve
        i += 1

    dict_of_pandl_by_rule = dictOfAccountCurves(dict_of_pandl_by_rule)

    account_curve = accountCurveGroup(dict_of_pandl_by_rule, capital, weighted=False)

    account_curve = getattr(account_curve, 'gross').to_frame()
    account_curve = account_curve.resample('1B').sum()
    account_curve[account_curve == 0.0] = np.nan

    return account_curve


def get_forecast_as_list(instrument_code):  # 需自行调整的函数
    ewmac32 = get_capped_forecast(instrument_code, 32, 128)
    ewmac8 = get_capped_forecast(instrument_code, 8, 32)
    forecast = [ewmac32, ewmac8]
    return forecast


def get_net_returns():
    gross_returns_dict = {}
    for instrument in my_config.instruments:
        gross_returns_dict[instrument] = get_returns_for_optimisation(instrument)
    gross_returns_dict = dictOfReturnsForOptimisation(gross_returns_dict)
    net_returns = gross_returns_dict.single_resampled_set_of_returns('W')
    return net_returns


############################################################################################# 分割线

from sysquant.fitting_dates import fitDates, listOfFittingDates


def generate_fitting_dates(interval_frequency='365D'):
    net_returns = get_net_returns()
    start_date = net_returns.index[0]
    end_date = net_returns.index[-1]
    # 根据数据起始和结束日期，生成各period的开始日期，且从后朝前看
    start_dates_per_period = list(pd.date_range(end_date, start_date, freq='-' + interval_frequency))
    start_dates_per_period.reverse()
    # Rolling 方法
    periods = []
    fit_end_list = []
    for periods_index in range(len(start_dates_per_period))[1:-1]:
        period_start = start_dates_per_period[periods_index]
        period_end = start_dates_per_period[periods_index + 1]
        fit_start = start_date
        fit_end = start_dates_per_period[periods_index]
        fit_end_list.append(fit_end)
        fit_date = fitDates(fit_start, fit_end, period_start, period_end)
        periods.append(fit_date)

    periods = [fitDates(start_date, start_date, start_date, start_dates_per_period[1], no_data=True)] + periods

    return listOfFittingDates(periods), fit_end_list


def get_corr_estimator(data_for_analysis, fit_end, span=50000, min_periods=10):
    raw_corr = data_for_analysis.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(
        pairwise=True)  # span 和min_periods 都是config 里面的4倍，因为4个instruments
    columns = data_for_analysis.columns
    size_of_matrix = len(columns)
    corr_matrix_values = (raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(
        size_of_matrix).values)  # 截取fit_period之前的数据
    return corr_matrix_values


def get_mean_estimator(data_for_analysis, fit_end, span=50000, min_periods=10):
    mean = data_for_analysis.ewm(span=span, min_periods=min_periods).mean()  # 逻辑还是config 的4倍
    last_index = data_for_analysis.index[data_for_analysis.index < fit_end].size - 1
    mean = mean.iloc[last_index]
    annualised_mean_estimate = {}
    for rule_name, mean_value in mean.items():
        annualised_mean_estimate[rule_name] = mean_value * 365.25 / 7.0
    mean_list = [value for value in annualised_mean_estimate.values()]
    return mean_list


def get_stdev_estimator(data_for_analysis, fit_end, span=50000, min_periods=10):
    stdev = data_for_analysis.ewm(span=span, min_periods=min_periods).std()
    last_index = data_for_analysis.index[data_for_analysis.index < fit_end].size - 1
    stdev = stdev.iloc[last_index]
    annualised_stdev_estimate = {}
    for rule_name, std_value in stdev.items():
        annualised_stdev_estimate[rule_name] = std_value * ((365.25 / 7.0) ** 0.5)
    stdev_list = [value for value in annualised_stdev_estimate.values()]
    return stdev_list


def addem(weights):
    return 1.0 - sum(weights)


def neg_SR(weights, sigma, mus):
    estimated_returns = np.dot(weights, mus)[0]
    stdev = weights.dot(sigma).dot(weights.transpose()) ** 0.5

    sr = -estimated_returns / stdev
    return sr


from scipy.optimize import minimize


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


def replace_nan(x):
    if isinstance(x, float):
        return [0.5, 0.5]
    else:
        return x


def get_smoothed_weights(instrument_code, data_for_analysis):
    fit_period, fit_end_list = generate_fitting_dates()
    weight_list = []
    for i in range(len(fit_period) - 1):
        weight = get_weight(data_for_analysis, i, fit_end_list)
        weight_list.append(weight)

    weights = pd.Series(weight_list)
    weights.index = fit_end_list

    ewmac32 = get_capped_forecast(instrument_code, 32, 128)
    ewmac8 = get_capped_forecast(instrument_code, 8, 32)
    forecast = pd.concat([ewmac32, ewmac8], axis=1)
    weights = weights.reindex(forecast.index, method='ffill')
    weights = weights.apply(replace_nan)
    df = pd.DataFrame(weights.tolist())
    smoothed_weights = df.ewm(span=125).mean()
    return smoothed_weights


############################################################################################# 分割线

def get_concat_forecast(instrument, end='2023-09-03'):
    forecast32 = get_capped_forecast(instrument, 32, 128)
    forecast32.name = 'ewmac32'
    forecast8 = get_capped_forecast(instrument, 8, 32)
    forecast8.name = 'ewmac8'
    forecast = pd.concat([forecast32, forecast8], axis=1)
    forecast.index = pd.to_datetime(forecast.index)
    weekly_index = pd.date_range(start=forecast.index[0], end=end, freq='W')  # 结束日期需要自行根据品种设定
    forecast = forecast.reindex(weekly_index, method='ffill')
    return forecast


def corr_over_time(data_for_analysis, fit_end):
    raw_corr = data_for_analysis.ewm(span=250, min_periods=20, ignore_na=True).corr(pairwise=True)
    columns = data_for_analysis.columns
    size_of_matrix = len(columns)
    corr_matrix_values = (raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(
        size_of_matrix).values)
    return corr_matrix_values


def generate_fitting_dates_for_divmult(instrument_code, interval_frequency='365D'):
    forecast = get_concat_forecast(instrument_code)
    start_date = forecast.index[0]
    end_date = forecast.index[-1]
    # 根据数据起始和结束日期，生成各period的开始日期，且从后朝前看
    start_dates_per_period = list(pd.date_range(end_date, start_date, freq='-' + interval_frequency))
    start_dates_per_period.reverse()

    # Rolling 方法
    periods = []
    fit_end_list = []
    for periods_index in range(len(start_dates_per_period))[1:-1]:
        period_start = start_dates_per_period[periods_index]
        period_end = start_dates_per_period[periods_index + 1]
        fit_start = start_date
        fit_end = start_dates_per_period[periods_index]  # 之前的fit end 是 period end, 所以就造成了fit 和 use 的一部分重叠
        fit_end_list.append(fit_end)
        fit_date = fitDates(fit_start, fit_end, period_start, period_end)
        periods.append(fit_date)

    return listOfFittingDates(periods), fit_end_list


def get_div_mult(instrument_code, weights):
    fit_period_divmult, fit_end_list_divmult = generate_fitting_dates_for_divmult(instrument_code)
    forecast32 = get_capped_forecast(instrument_code, 32, 128)
    forecast = get_concat_forecast(instrument_code)

    weights.index = forecast32.index

    ref_periods = fit_end_list_divmult
    # get weights
    weights = weights.rename(columns={0: 'ewmac32', 1: 'ewmac8'})
    corr_list = []
    for i in range(len(fit_end_list_divmult)):
        fit_end = fit_end_list_divmult[i]
        corr_matrix = corr_over_time(forecast, fit_end)
        corr_list.append(corr_matrix)
    div_mult = []
    for corrmatrix, start_of_period in zip(corr_list, ref_periods):
        weight_slice = weights[:start_of_period]
        weight_np = np.array(weight_slice.iloc[-1])
        variance = weight_np.dot(corrmatrix).dot(weight_np.transpose())
        risk = variance ** 0.5
        dm = np.min([1 / risk, 2.5])
        div_mult.append(dm)
    div_mult_df = pd.Series(div_mult, index=ref_periods)
    div_mult_df_daily = div_mult_df.reindex(weights.index, method='ffill')
    div_mult_df_daily[div_mult_df_daily.isna()] = 1.0
    div_mult_df_smoothed = div_mult_df_daily.ewm(span=125).mean()

    return div_mult_df_smoothed


############################################################################################# 分割线


def get_price_vol(instrument_code):
    daily_carry_price = get_raw_carry_data(instrument_code)
    daily_returns = get_daily_returns(instrument_code)
    vol = mixed_vol_calc(daily_returns, slow_vol_years=10)
    (daily_carry_price, vol) = daily_carry_price.align(vol, join='right')
    perc_vol = 100.0 * (vol / daily_carry_price.ffill().abs())
    return perc_vol


def get_block_value(instrument_code):
    instr_object = get_instrument_info(instrument_code)  # 基础品种信息
    daily_carry_price = get_raw_carry_data(instrument_code)
    block_move_price = instr_object.meta_data.Pointsize
    block_value = daily_carry_price.ffill() * block_move_price * 0.01
    return block_value, block_move_price


def get_instrument_value_vol(instrument_code):
    block_value, block_move_price = get_block_value(instrument_code)
    perc_vol = get_price_vol(instrument_code)
    instr_currency_vol = block_value.ffill() * perc_vol
    instr_value_vol = instr_currency_vol.ffill()
    return instr_value_vol


def get_avg_position_at_subsystem_level(instrument_code, capital=1000000, perc_vol_target=16):
    instr_value_vol = get_instrument_value_vol(instrument_code)
    annual_cash_vol_target = capital * perc_vol_target / 100
    daily_cash_vol_target = annual_cash_vol_target / 16
    vol_scalar = daily_cash_vol_target / instr_value_vol
    return vol_scalar


def get_combined_forecast(instrument_code):
    net_returns = get_net_returns()
    forecast8 = get_capped_forecast(instrument_code, 8, 32)
    forecast8 = forecast8.rename('ewmac8')
    forecast32 = get_capped_forecast(instrument_code, 32, 128)
    forecast32 = forecast32.rename('ewmac32')
    df = pd.concat([forecast8, forecast32], axis=1)  # Forecast 没有问题

    weights = get_smoothed_weights(instrument_code, net_returns)
    weights.rename(columns={0: 'ewmac32', 1: 'ewmac8'}, inplace=True)
    weights = weights[['ewmac8', 'ewmac32']]
    weights.index = df.index
    forecast_div_multiplier = get_div_mult(instrument_code, weights)

    raw_weighted_forecast = weights * df
    raw_weighted_forecast = raw_weighted_forecast.sum(axis=1)
    combined_forecast = raw_weighted_forecast * forecast_div_multiplier
    capped_combined_forecast = combined_forecast.clip(20, -20)
    return capped_combined_forecast


def get_subsystem_position(instrument_code):
    forecast32 = get_capped_forecast(instrument_code, 32, 128)
    capped_combined_forecast = get_combined_forecast(instrument_code)

    vol = get_avg_position_at_subsystem_level(instrument_code)
    vol = vol.reindex(forecast32.index, method='ffill')

    subsystem_position_raw = vol * capped_combined_forecast / 10.0
    return subsystem_position_raw


############################################################################################# 分割线
from sysobjects.instruments import instrumentCosts


def get_raw_cost_data(instrument_code):
    instrument_data = get_instrument_info(instrument_code)
    spread_costs = get_spread_cost(instrument_code)
    instrument_meta_data = instrument_data.meta_data
    instrument_costs = instrumentCosts.from_meta_data_and_spread_cost(instrument_meta_data, spread_costs)
    return instrument_costs


def get_buffered_positions(instrument_code):
    subsystem_position = get_subsystem_position(instrument_code)
    vol_scalar = get_avg_position_at_subsystem_level(instrument_code)

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
        if trade_to_edge:
            return top_pos
        else:
            return optimal_position
    elif last_position < bot_pos:
        if trade_to_edge:
            return bot_pos
        else:
            return optimal_position
    else:
        return last_position


from systems.accounts.pandl_calculators.pandl_cash_costs import pandlCalculationWithCashCostsAndFills


def get_stdev_estimator_for_instrument_weight(data_for_analysis, fit_end):
    span = 50000
    min_periods = 5
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


def get_mean_estimator_for_instrument_weight(data_for_analysis, fit_end):
    span = 50000
    min_periods = 5
    mean = data_for_analysis.ewm(span=span, min_periods=min_periods).mean()  # 逻辑还是config 的4倍
    last_index = data_for_analysis.index[data_for_analysis.index < fit_end].size - 1
    mean = mean.iloc[last_index]
    annualised_mean_estimate = {}
    for rule_name, mean_value in mean.items():
        annualised_mean_estimate[rule_name] = mean_value * 365.25 / 7.0
    mean_list = [value for value in annualised_mean_estimate.values()]
    return mean_list


def get_corr_estimator_for_instrument_weight(data_for_analysis, fit_end):
    span = 500000
    min_periods = 10
    raw_corr = data_for_analysis.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(
        pairwise=True)  # span 和min_periods 都是config 里面的4倍，因为4个instruments
    columns = data_for_analysis.columns
    size_of_matrix = len(columns)
    corr_matrix_values = (raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(
        size_of_matrix).values)  # 截取fit_period之前的数据
    corr_matrix_values = [[max(0, item) for item in sublist] for sublist in corr_matrix_values]
    return corr_matrix_values


############################################################################################# 分割线

def get_turnover_for_forecast(instrument_code, Lfast, Lslow):
    forecast = get_capped_forecast(instrument_code, Lfast, Lslow)
    daily_forecast = forecast.resample('1B').last()
    daily_y = pd.Series(np.full(daily_forecast.shape[0], 10.0), daily_forecast.index)
    daily_forecast_normalised_for_y = daily_forecast / daily_y.ffill()
    avg_daily = float(daily_forecast_normalised_for_y.diff().abs().mean())
    turnover = avg_daily * 256
    return turnover


def get_turnover(x_series, y, smooth_y_days=250):
    daily_x = x_series.resample('1B').last()
    if isinstance(y, float) or isinstance(y, int):
        daily_y = pd.Series(np.full(daily_x.shape[0], float(y)), daily_x.index)
    else:
        daily_y = y.reindex(daily_x.index, method='ffill')
        daily_y = daily_y.ewm(smooth_y_days, min_periods=2).mean()
    daily_x_normalised_for_y = daily_x / daily_y.ffill()
    avg_daily = float(daily_x_normalised_for_y.diff().abs().mean())
    turnover = avg_daily * 256
    return turnover


def get_turnover_dict_for_single_rule(Lfast, Lslow):
    turnover_dict_single_rule = {}
    for instrument in my_config.instruments:
        turnover = get_turnover_for_forecast(instrument, Lfast, Lslow)
        turnover_dict_single_rule[instrument] = turnover
    return turnover_dict_single_rule


def get_turnover_dict_for_rules(rule_list=[['ewmac32', 32, 128], ['ewmac8', 8, 32]]):
    turnover_dict = {}
    for rule in rule_list:
        turnover_dict_single_rule = get_turnover_dict_for_single_rule(rule[1], rule[2])
        turnover_dict[rule[0]] = turnover_dict_single_rule
    return turnover_dict


def get_portfolio_turnover():
    turnover_list = []
    for instrument in my_config.instruments:
        subsystem_position = get_subsystem_position(instrument)
        vol_scalar = get_avg_position_at_subsystem_level(instrument)
        turnover = get_turnover(subsystem_position, vol_scalar)
        turnover_list.append(turnover)
    return turnover_list


def pandl_calculator_for_subsystem_with_cash_costs(instrument_code):
    raw_costs = get_raw_cost_data(instrument_code)
    buffered_position = get_buffered_positions(instrument_code)

    price = get_daily_price(instrument_code)
    block_value, block_move_size = get_block_value(instrument_code)
    roll_parameters = get_roll_parameters(instrument_code)
    rolls_per_year = roll_parameters.rolls_per_year_in_hold_cycle()
    capital = 1000000.0
    fx = pd.Series(1.0, index=price.index)

    pandl_calculator = pandlCalculationWithCashCostsAndFills(price, raw_costs=raw_costs,
                                                             positions=buffered_position, capital=capital,
                                                             value_per_point=block_move_size,
                                                             vol_normalise_currency_costs=True,
                                                             rolls_per_year=rolls_per_year, delayfill=True,
                                                             roundpositions=True, fx=fx)
    return pandl_calculator


def caculate_instrument_weights(daily_ret):
    instrument_number = len(daily_ret.columns)
    weekly_ret = daily_ret.resample('W').sum()  # SP500_micro 的一些数值不对，其他的都能对的上。怀疑是不是一些nan被填充了
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


def iter_method(gross, instrument):
    curve = gross.dict_of_account_curves[instrument]
    calculator = curve.pandl_calculator_with_costs
    pnl = calculator.as_pd_series(percent=False, curve_type=GROSS_CURVE)

    pnl.index = pd.to_datetime(pnl.index)
    daily_pnl = pnl.resample("B").sum()
    return daily_pnl


def get_instrument_weight(my_config):
    instruments = my_config.instruments

    dict_of_account_curves = {}
    for instrument in instruments:
        pandl_calculator = pandl_calculator_for_subsystem_with_cash_costs(instrument)
        dict_of_account_curves[instrument] = accountCurve(pandl_calculator)
    dict_of_account_curves = dictOfAccountCurves(dict_of_account_curves)

    capital = 1000000

    gross = accountCurveGroup(
        dict_of_account_curves,
        capital=capital,
        curve_type=GROSS_CURVE,
        weighted=False)
    data_as_list = [iter_method(gross, instrument) for instrument in instruments]


    data_as_pd = pd.concat(data_as_list, axis=1)
    data_as_pd.columns = instruments
    curves = data_as_pd.resample("1B").sum()
    curves[curves == 0.0] = np.nan

    weight = caculate_instrument_weights(curves)

    print(weight)

    return weight


if __name__ == '__main__':
    get_instrument_weight(my_config)
