import binance
from binance.client import Client
from requests import get

api_key = 'your_api_key'
api_secret  = 'your_api_secret'
client = Client(api_key, api_secret)
# Get account information
account_info = client.get_account()
print(account_info)

get balance = client.get_asset_balance(asset='BTC')
print(get_balance)  