import numpy as np
import pandas as pd

from refactory.submission import get_capped_forecast, my_config, get_subsystem_position, \
    get_avg_position_at_subsystem_level


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
