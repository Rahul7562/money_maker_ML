from binance.client import Client
from binance.enums import HistoricalKlinesType
import pandas as pd


def get_historical_klines_df(symbol, interval, start_str, end_str=None, futures=True):
    client = Client()
    klines_type = HistoricalKlinesType.FUTURES if futures else HistoricalKlinesType.SPOT
    klines = client.get_historical_klines(
        symbol, interval, start_str, end_str, klines_type=klines_type
    )
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
    ]
    features = ["open","high","low","close","volume"]
    df = pd.DataFrame(klines, columns=cols)
    df[["open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"]] = df[["open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"]].astype(float)
    return df[["open_time","open","high","low","close","volume"]]


def data(input_symbol):
    print(get_historical_klines_df(input_symbol, Client.KLINE_INTERVAL_1HOUR, "3 day ago UTC", futures=True))


