import numpy as np
from scipy.optimize import minimize


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