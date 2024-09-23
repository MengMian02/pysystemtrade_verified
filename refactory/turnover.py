import numpy as np
import pandas as pd

from refactory.submission import get_capped_forecast, my_config, calculate_avg_position, get_net_returns, \
    get_div_mult, generate_fitting_dates, get_weight, replace_nan


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
        vol_scalar = calculate_avg_position(instrument)
        turnover = get_turnover(subsystem_position, vol_scalar)
        turnover_list.append(turnover)
    return turnover_list


def get_subsystem_position(instrument_code):
    forecast32 = get_capped_forecast(instrument_code, 32, 128)
    capped_combined_forecast = get_combined_forecast(instrument_code)

    vol = calculate_avg_position(instrument_code)
    vol = vol.reindex(forecast32.index, method='ffill')

    subsystem_position_raw = vol * capped_combined_forecast / 10.0
    return subsystem_position_raw


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
