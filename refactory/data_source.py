import pandas as pd

from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
from sysobjects.instruments import instrumentCosts

source_data = csvFuturesSimData()


def get_instrument_info(instrument_code):
    return source_data.db_futures_instrument_data.get_instrument_data(instrument_code)


def get_daily_price(instrument_code):
    prices = source_data.daily_prices(instrument_code)
    prices.index = pd.to_datetime(prices.index)
    return prices


def get_spread_cost(instrument_code):
    return source_data.db_spread_cost_data.get_spread_cost(instrument_code)


def get_roll_parameters(instrument_code):
    return source_data.db_roll_parameters.get_roll_parameters(instrument_code)


def get_raw_carry_data(instrument_code):
    filename = '..\\data\\futures\\multiple_prices_csv\\' + instrument_code + '.csv'
    carry_data = pd.read_csv(filename)
    carry_price = carry_data['PRICE']
    carry_price.index = carry_data['DATETIME']
    carry_price.index = pd.to_datetime(carry_price.index)
    daily_carry_price = carry_price.resample('1B').last()
    return daily_carry_price


def get_rolls_per_year(instrument):
    roll_parameters = get_roll_parameters(instrument)
    rolls_per_year = roll_parameters.rolls_per_year_in_hold_cycle()
    return rolls_per_year


def get_raw_cost_data(instrument_code):
    instrument_data = get_instrument_info(instrument_code)
    spread_costs = get_spread_cost(instrument_code)
    instrument_meta_data = instrument_data.meta_data
    instrument_costs = instrumentCosts.from_meta_data_and_spread_cost(instrument_meta_data, spread_costs)
    return instrument_costs


def get_block_move_price(instrument_code):
    instr_object = get_instrument_info(instrument_code)  # 基础品种信息
    block_move_price = instr_object.meta_data.Pointsize
    return block_move_price
