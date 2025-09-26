from dhanhq import dhanhq
from .base import BrokerBase
import pandas as pd

class DhanBroker(BrokerBase):
    def __init__(self, client_id, access_token):
        super().__init__()
        self.dhan = dhanhq(client_id, access_token)
        self.instruments_df = None
        self.authenticate() # Ensure authentication is called on initialization

    def authenticate(self):
        try:
            profile = self.dhan.get_profile()
            if profile.get('status') == 'success':
                self.authenticated = True
                print("Dhan authentication successful.")
            else:
                self.authenticated = False
                print(f"Dhan authentication failed: {profile}")
        except Exception as e:
            self.authenticated = False
            print(f"An error occurred during Dhan authentication: {e}")

    def download_instruments(self):
        """
        Downloads all tradable instruments from Dhan for relevant segments
        and stores them in a DataFrame.
        """
        try:
            segments = ['NFO', 'IDX', 'EQ']
            all_instruments = []
            for segment in segments:
                try:
                    instruments = self.dhan.get_all_scrips_for_exchange(exchange=segment)
                    all_instruments.extend(instruments)
                except Exception as e:
                    print(f"Could not fetch instruments for segment {segment}: {e}")

            df = pd.DataFrame(all_instruments)

            column_mapping = {
                'trading_symbol': 'tradingsymbol',
                'security_id': 'instrument_token',
                'lot_size': 'lot_size',
                'instrument_type': 'instrument_type',
                'strike_price': 'strike',
                'expiry_date': 'expiry',
                'exchange': 'exchange_segment'
            }
            df.rename(columns=column_mapping, inplace=True)

            # Create a standard tradingsymbol for indices for easier lookup
            idx_mask = df['exchange_segment'] == 'IDX'
            # Use the correct column name from Dhan's API response
            df.loc[idx_mask, 'tradingsymbol'] = 'NSE:' + df.loc[idx_mask, 'SEM_INSTRUMENT_NAME']

            # Standardize stock symbols to include the exchange
            eq_mask = df['exchange_segment'] == 'EQ'
            df.loc[eq_mask, 'tradingsymbol'] = 'NSE:' + df.loc[eq_mask, 'tradingsymbol']


            self.instruments_df = df
            print("Instruments downloaded from Dhan.")
        except Exception as e:
            print(f"Error downloading instruments from Dhan: {e}")
            self.instruments_df = pd.DataFrame()

    def get_instruments(self):
        if self.instruments_df is None:
            self.download_instruments()
        return self.instruments_df

    def get_quote(self, symbols):
        """
        Gets the quote for a list of symbols by looking up their security_id.
        """
        if not isinstance(symbols, list):
            symbols = [symbols]

        if self.instruments_df is None:
            self.download_instruments()

        quotes = {}
        for symbol in symbols:
            instrument = self.instruments_df[self.instruments_df['tradingsymbol'] == symbol]
            if instrument.empty:
                print(f"Instrument not found for symbol: {symbol}")
                continue

            security_id = str(instrument.iloc[0]['instrument_token'])
            exchange_segment = instrument.iloc[0]['exchange_segment']

            try:
                # Use the correct positional argument for security_id
                quote_data = self.dhan.quote(security_id, exchange_segment=exchange_segment)
                if quote_data.get('status') == 'success':
                    data = quote_data['data']
                    quotes[symbol] = {
                        'instrument_token': security_id,
                        'last_price': data['last_price'],
                        'oi': data.get('open_interest', 0)
                    }
            except Exception as e:
                print(f"Error fetching quote from Dhan for {symbol}: {e}")

        return quotes