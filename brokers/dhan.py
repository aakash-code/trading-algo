import os
import pandas as pd
from dhanhq import dhanhq
from .base import BrokerBase

class DhanBroker(BrokerBase):
    def __init__(self):
        super().__init__()
        self.client_id = os.environ.get("DHAN_CLIENT_ID")
        self.access_token = os.environ.get("DHAN_ACCESS_TOKEN")
        self.dhan = None
        self.instruments_df = None
        self.on_ticks = None
        self.on_connect = None
        self.on_order_update = None
        self.ws = None

        if self.client_id and self.access_token:
            self.dhan = dhanhq(self.client_id, self.access_token)
            self.authenticate()

    def authenticate(self):
        if not self.dhan:
            return None
        try:
            response = self.dhan.get_fund_limits()
            if response and response.get('status') == 'success':
                self.authenticated = True
                print("Dhan authentication successful.")
                return self.access_token
        except Exception as e:
            print(f"Dhan authentication check failed: {e}")
        self.authenticated = False
        return None

    def download_instruments(self, exchange="NFO"):
        if not self.authenticated:
            print("Cannot download instruments, not authenticated.")
            return

        exchange_map = {"NFO": "NSE_FNO", "NSE": "NSE_EQ"}
        dhan_exchange = exchange_map.get(exchange, exchange)

        securities = self.dhan.get_securities(dhan_exchange)
        if securities and securities.get('status') == 'success' and 'data' in securities:
            df = pd.DataFrame(securities['data'])
            column_mapping = {
                'SEM_TRADING_SYMBOL': 'tradingsymbol',
                'SEM_SM_INSTRUMENT_TYPE': 'instrument_type',
                'SEM_EXCH_SEGMENT': 'segment',
                'SEM_STRIKE_PRICE': 'strike',
                'SEM_INSTRUMENT_NAME': 'name',
                'SEM_EXPIRY_DATE': 'expiry',
                'SEM_LOT_UNITS': 'lot_size',
                'SEM_SECURITY_ID': 'instrument_token'
            }
            df.rename(columns=column_mapping, inplace=True)
            df.loc[df['segment'] == 'NSE_FNO', 'segment'] = 'NFO-OPT'
            df['instrument_token'] = pd.to_numeric(df['instrument_token'])
            df['strike'] = pd.to_numeric(df['strike'])
            self.instruments_df = df
            print(f"Dhan instruments for {exchange} downloaded successfully.")
        else:
            print(f"Failed to download Dhan instruments for {exchange}.")

    def get_quote(self, symbol_code: str):
        if not self.authenticated:
            return None

        if "NIFTY 50" in symbol_code:
            security_id = '26000' # NIFTY 50 Index
        else:
            if self.instruments_df is None:
                self.download_instruments()

            parts = symbol_code.split(':', 1)
            tradingsymbol = parts[1].strip() if len(parts) == 2 else parts[0]

            instrument_row = self.instruments_df[self.instruments_df['tradingsymbol'] == tradingsymbol]
            if instrument_row.empty:
                return None
            security_id = str(instrument_row.iloc[0]['instrument_token'])

        response = self.dhan.get_quotes([security_id])
        if response and response.get('status') == 'success' and 'data' in response:
            quote_data = response['data'][security_id]
            return {
                symbol_code: {
                    'last_price': quote_data.get('ltp'),
                    'instrument_token': int(security_id)
                }
            }
        return None

    def place_order(self, symbol, quantity, transaction_type, order_type, exchange, product, price=0, variety="REGULAR", tag=None):
        if not self.authenticated or self.instruments_df is None:
            return -1

        instrument_row = self.instruments_df[self.instruments_df['tradingsymbol'] == symbol]
        if instrument_row.empty:
            return -1
        security_id = str(instrument_row.iloc[0]['instrument_token'])

        exchange_map = {'NFO': 'NSE_FNO'}
        dhan_exchange = exchange_map.get(exchange, exchange)

        product_type_map = {'NRML': 'MARGIN', 'MIS': 'INTRADAY'}
        dhan_product = product_type_map.get(product, 'MARGIN')

        order_price = price if price is not None else 0
        response = self.dhan.place_order(
            security_id=security_id, exchange_segment=dhan_exchange, transaction_type=transaction_type,
            quantity=quantity, order_type=order_type, product_type=dhan_product, price=order_price, tag=tag
        )

        if response and response.get('status') == 'success' and response.get('data', {}).get('orderId'):
            return response['data']['orderId']
        else:
            print(f"Dhan order placement failed: {response}")
            return -1

    def connect_websocket(self):
        # This method is adjusted to be called from survivor.py, which will then set callbacks and subscribe.
        # For Dhan, subscription happens at connection time, so we need a different flow.
        # The main loop in survivor.py will be slightly adjusted for this.
        pass

    def subscribe_and_connect(self, index_symbol):
        if not self.authenticated or not all([self.on_ticks, self.on_connect]):
            return

        quote_data = self.get_quote(index_symbol)
        if not quote_data:
            return
        token = quote_data[index_symbol]['instrument_token']

        instruments_to_subscribe = [('13', str(token))] # 13 for IDX

        def on_ticks_internal(ticks):
            formatted_ticks = [{'instrument_token': int(ticks['security_id']), 'last_price': ticks['ltp']}]
            self.on_ticks(self, formatted_ticks)

        def on_order_internal(order_update):
            if self.on_order_update:
                self.on_order_update(self, order_update)

        self.ws = self.dhan.dhan_feed(
            self.client_id, self.access_token, instruments_to_subscribe,
            on_ticks=on_ticks_internal, on_order=on_order_internal
        )
        self.on_connect(self, "Dhan websocket connected.")