from sysdata.sim.csv_futures_sim_data import csvFuturesSimData

source_data = csvFuturesSimData()


def get_instrument_info(instrument_code):
    return source_data.db_futures_instrument_data.get_instrument_data(instrument_code)


def get_daily_price(instrument_code):
    return source_data.daily_prices(instrument_code)


def get_spread_cost(instrument_code):
    return source_data.db_spread_cost_data.get_spread_cost(instrument_code)


def get_roll_parameters(instrument_code):
    return source_data.db_roll_parameters.get_roll_parameters(instrument_code)
