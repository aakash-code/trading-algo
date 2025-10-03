import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from brokers.base import BrokerBase
import pandas as pd
from threading import Thread

from logger import logger

try:
    from dhanhq import dhanhq, marketfeed
except ImportError:
    logger.error("dhanhq package not installed. Please run: pip install dhanhq")
    raise

load_dotenv()


class DhanBroker(BrokerBase):
    """
    Dhan broker implementation for trading and market data.

    Features:
    - REST API for orders, quotes, and historical data
    - WebSocket for live market feed
    - Support for NSE, BSE, MCX, NFO segments

    Authentication:
    - Requires client_id and access_token from Dhan
    - No TOTP required - simpler authentication than Zerodha/Fyers
    """

    def __init__(self):
        super().__init__()
        self.client_id, self.access_token = self.authenticate()
        self.dhan = dhanhq(self.client_id, self.access_token)
        self.dhan_ws = None
        self.symbols = []
        self.instruments_df = None

    def authenticate(self):
        """
        Authenticate with Dhan API.

        Returns:
            tuple: (client_id, access_token)
        """
        client_id = os.getenv('BROKER_ID')
        access_token = os.getenv('BROKER_API_KEY')

        if not all([client_id, access_token]):
            raise Exception("Missing BROKER_ID or BROKER_API_KEY in environment variables.")

        logger.info(f"Dhan authentication successful for client: {client_id}")
        return client_id, access_token

    def get_orders(self):
        """Get all orders for the day."""
        try:
            return self.dhan.get_order_list()
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return None

    def get_quote(self, symbol, exchange="NSE_EQ"):
        """
        Get quote for a symbol.

        Args:
            symbol: Security ID or trading symbol
            exchange: Exchange segment (NSE_EQ, NSE_FNO, BSE_EQ, etc.)

        Returns:
            dict: Quote data with last_price, volume, etc.
        """
        try:
            # If symbol contains ":", parse it (e.g., "NSE:NIFTY 50")
            if ":" in symbol:
                parts = symbol.split(":")
                exchange_map = {
                    "NSE": "NSE_EQ",
                    "NFO": "NSE_FNO",
                    "BSE": "BSE_EQ"
                }
                exchange = exchange_map.get(parts[0], "NSE_EQ")
                symbol_name = parts[1].strip()

                # For indices like "NIFTY 50", get the security ID
                if symbol_name == "NIFTY 50":
                    security_id = "13"  # Dhan security ID for NIFTY 50
                elif symbol_name == "BANKNIFTY":
                    security_id = "25"  # Dhan security ID for BANKNIFTY
                else:
                    # Try to find security_id from instruments
                    security_id = self._get_security_id(symbol_name, exchange)
            else:
                security_id = symbol

            # Get quote data
            response = self.dhan.ohlc_data(
                securities={exchange: [security_id]}
            )

            if response and response.get('data'):
                # Transform Dhan response to match Zerodha format
                quote_data = response['data'][exchange][security_id]

                # Create standardized response
                formatted_symbol = f"{exchange}:{security_id}"
                return {
                    formatted_symbol: {
                        'instrument_token': security_id,
                        'last_price': quote_data.get('close', quote_data.get('last_price', 0)),
                        'volume': quote_data.get('volume', 0),
                        'ohlc': {
                            'open': quote_data.get('open', 0),
                            'high': quote_data.get('high', 0),
                            'low': quote_data.get('low', 0),
                            'close': quote_data.get('close', 0)
                        }
                    }
                }
            else:
                logger.error(f"No quote data received for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def _get_security_id(self, symbol, exchange):
        """
        Find security ID from symbol name.

        Args:
            symbol: Trading symbol
            exchange: Exchange segment

        Returns:
            str: Security ID
        """
        # This would require downloading instruments and mapping
        # For now, return the symbol itself
        logger.warning(f"Security ID lookup not implemented. Using symbol: {symbol}")
        return symbol

    def place_order(self, symbol, quantity, price, transaction_type, order_type,
                   variety, exchange, product, tag="Unknown"):
        """
        Place an order on Dhan.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price (ignored for MARKET orders)
            transaction_type: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            variety: Order variety (not used in Dhan)
            exchange: Exchange segment
            product: Product type (INTRADAY, CNC, etc.)
            tag: Order tag

        Returns:
            str: Order ID if successful, -1 otherwise
        """
        try:
            # Map exchanges
            exchange_map = {
                "NSE": self.dhan.NSE,
                "NFO": self.dhan.NFO,
                "BSE": self.dhan.BSE,
                "MCX": self.dhan.MCX
            }
            dhan_exchange = exchange_map.get(exchange, self.dhan.NSE)

            # Map transaction types
            dhan_transaction = self.dhan.BUY if transaction_type == "BUY" else self.dhan.SELL

            # Map order types
            dhan_order_type = self.dhan.MARKET if order_type == "MARKET" else self.dhan.LIMIT

            # Map product types
            product_map = {
                "NRML": self.dhan.CARRY_FORWARD,
                "MIS": self.dhan.INTRA,
                "CNC": self.dhan.CNC
            }
            dhan_product = product_map.get(product, self.dhan.INTRA)

            # Get security ID
            security_id = self._get_security_id(symbol, exchange)

            logger.info(f"Placing Dhan order: {symbol} {transaction_type} {quantity} @ {price}")

            response = self.dhan.place_order(
                security_id=security_id,
                exchange_segment=dhan_exchange,
                transaction_type=dhan_transaction,
                quantity=quantity,
                order_type=dhan_order_type,
                product_type=dhan_product,
                price=price if order_type == "LIMIT" else 0,
                tag=tag
            )

            if response and response.get('data', {}).get('orderId'):
                order_id = response['data']['orderId']
                logger.info(f"Dhan order placed successfully: {order_id}")
                return order_id
            else:
                logger.error(f"Dhan order placement failed: {response}")
                return -1

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return -1

    def get_positions(self):
        """Get current positions."""
        try:
            return self.dhan.get_positions()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return None

    def download_instruments(self):
        """
        Download instrument master from Dhan.

        Note: Dhan doesn't provide a comprehensive instruments CSV like Zerodha.
        You may need to maintain your own mapping or use their security master.
        """
        try:
            # Dhan doesn't have a direct instruments download API
            # You would need to implement this based on your requirements
            logger.warning("Dhan instruments download not fully implemented")

            # Create a minimal DataFrame structure for compatibility
            self.instruments_df = pd.DataFrame(columns=[
                'instrument_token', 'exchange_token', 'tradingsymbol',
                'name', 'last_price', 'expiry', 'strike', 'tick_size',
                'lot_size', 'instrument_type', 'segment', 'exchange'
            ])

        except Exception as e:
            logger.error(f"Error downloading instruments: {e}")
            self.instruments_df = pd.DataFrame()

    def get_instruments(self):
        """Get instruments DataFrame."""
        return self.instruments_df

    # WebSocket callbacks
    def on_ticks(self, ws, ticks):
        """
        WebSocket tick callback.
        Override this method in your implementation.
        """
        logger.debug(f"Dhan ticks: {ticks}")

    def on_connect(self, ws, response):
        """
        WebSocket connection callback.
        Override this method in your implementation.
        """
        logger.info(f"Dhan WebSocket connected: {response}")

    def on_order_update(self, ws, data):
        """
        Order update callback.
        Override this method in your implementation.
        """
        logger.info(f"Dhan order update: {data}")

    def on_close(self, ws, code, reason):
        """WebSocket close callback."""
        logger.info(f"Dhan WebSocket closed: {code} - {reason}")

    def on_error(self, ws, code, reason):
        """WebSocket error callback."""
        logger.error(f"Dhan WebSocket error: {code} - {reason}")

    def connect_websocket(self):
        """
        Connect to Dhan WebSocket for live market data.

        Note: Implement the market feed connection based on symbols
        """
        try:
            # Create instruments list for market feed
            # Format: [(exchange, security_id, subscription_type)]
            instruments = []
            for symbol in self.symbols:
                # Parse symbol and create instrument tuple
                # Example: (marketfeed.NSE, "1333", marketfeed.Ticker)
                instruments.append((marketfeed.NSE, symbol, marketfeed.Quote))

            # Initialize market feed
            self.dhan_ws = marketfeed.DhanFeed(
                self.client_id,
                self.access_token,
                instruments
            )

            # Start connection in separate thread
            def run_feed():
                while True:
                    try:
                        self.dhan_ws.run_forever()
                        response = self.dhan_ws.get_data()
                        if response:
                            # Call the on_ticks callback
                            self.on_ticks(self.dhan_ws, response)
                    except Exception as e:
                        logger.error(f"Error in market feed: {e}")

            feed_thread = Thread(target=run_feed, daemon=True)
            feed_thread.start()

            logger.info("Dhan WebSocket connected successfully")

        except Exception as e:
            logger.error(f"Error connecting Dhan WebSocket: {e}")

    def symbols_to_subscribe(self, symbols):
        """Set symbols for WebSocket subscription."""
        self.symbols = symbols
