import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from brokers.base import BrokerBase
import pandas as pd
from threading import Thread
import json

from logger import logger

try:
    import upstox_client
    from upstox_client import Configuration, ApiClient, LoginApi, OrderApi, MarketQuoteApi
    from upstox_client.rest import ApiException
except ImportError:
    logger.error("upstox-python-sdk package not installed. Please run: pip install upstox-python-sdk")
    raise

load_dotenv()


class UpstoxBroker(BrokerBase):
    """
    Upstox broker implementation for trading and market data.

    Features:
    - REST API for orders, quotes, and historical data
    - WebSocket V3 for live market data streaming
    - Support for NSE, BSE, MCX, NFO segments

    Authentication:
    - Uses OAuth2 flow
    - Requires API key, API secret, and redirect URI
    - Generates access token via authorization code
    """

    def __init__(self, without_oauth=False):
        super().__init__()
        self.without_oauth = without_oauth
        self.access_token = self.authenticate()
        self.configuration = Configuration()
        self.configuration.access_token = self.access_token
        self.api_client = ApiClient(self.configuration)

        # Initialize API instances
        self.order_api = OrderApi(self.api_client)
        self.quote_api = MarketQuoteApi(self.api_client)

        self.upstox_ws = None
        self.symbols = []
        self.instruments_df = None

    def authenticate(self):
        """
        Authenticate with Upstox API using OAuth2.

        Returns:
            str: Access token
        """
        api_key = os.getenv('BROKER_API_KEY')
        api_secret = os.getenv('BROKER_API_SECRET')
        redirect_uri = os.getenv('BROKER_REDIRECT_URI', 'http://localhost')

        if not api_key:
            raise Exception("Missing BROKER_API_KEY in environment variables.")

        if self.without_oauth:
            # Manual OAuth flow
            print(f"Please visit this URL to authorize:")
            print(f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}")
            print("\nAfter authorization, paste the authorization code from the redirect URL:")
            auth_code = input("Authorization Code: ")

            # Exchange auth code for access token
            configuration = Configuration()
            api_client = ApiClient(configuration)
            login_api = LoginApi(api_client)

            try:
                api_response = login_api.token(
                    api_version='2.0',
                    code=auth_code,
                    client_id=api_key,
                    client_secret=api_secret,
                    redirect_uri=redirect_uri,
                    grant_type='authorization_code'
                )
                access_token = api_response.access_token
                logger.info("Upstox authentication successful")
                return access_token

            except ApiException as e:
                logger.error(f"Exception when calling LoginApi->token: {e}")
                raise
        else:
            # Try to get from environment variable directly
            access_token = os.getenv('BROKER_ACCESS_TOKEN')
            if not access_token:
                raise Exception(
                    "No BROKER_ACCESS_TOKEN found. Please set without_oauth=True for manual login."
                )
            logger.info("Using access token from environment")
            return access_token

    def get_orders(self):
        """Get all orders for the day."""
        try:
            api_response = self.order_api.get_order_book(api_version='2.0')
            return api_response
        except ApiException as e:
            logger.error(f"Error fetching orders: {e}")
            return None

    def get_quote(self, symbol, exchange="NSE_EQ"):
        """
        Get quote for a symbol.

        Args:
            symbol: Instrument token (e.g., "NSE_EQ|INE669E01016")
            exchange: Exchange segment (optional, for compatibility)

        Returns:
            dict: Quote data with last_price, volume, etc.
        """
        try:
            # Format symbol for Upstox
            # Upstox uses format: "NSE_EQ|INE669E01016"
            if ":" in symbol:
                # Convert from "NSE:NIFTY 50" to Upstox format
                parts = symbol.split(":")
                exchange_map = {
                    "NSE": "NSE_INDEX",  # For indices
                    "NFO": "NSE_FO",
                    "BSE": "BSE_EQ"
                }
                exchange_prefix = exchange_map.get(parts[0], "NSE_EQ")
                symbol_name = parts[1].strip()

                # Special handling for indices
                if symbol_name == "NIFTY 50":
                    instrument_key = "NSE_INDEX|Nifty 50"
                elif symbol_name == "BANKNIFTY":
                    instrument_key = "NSE_INDEX|Nifty Bank"
                else:
                    # For stocks, need to convert to instrument token
                    instrument_key = f"{exchange_prefix}|{symbol_name}"
            else:
                instrument_key = symbol

            # Get quote
            api_response = self.quote_api.get_full_market_quote(
                instrument_key=instrument_key,
                api_version='2.0'
            )

            if api_response and api_response.data:
                quote_data = api_response.data.get(instrument_key, {})

                # Transform to standardized format
                ohlc = quote_data.get('ohlc', {})
                return {
                    instrument_key: {
                        'instrument_token': instrument_key,
                        'last_price': quote_data.get('last_price', 0),
                        'volume': quote_data.get('volume', 0),
                        'ohlc': {
                            'open': ohlc.get('open', 0),
                            'high': ohlc.get('high', 0),
                            'low': ohlc.get('low', 0),
                            'close': ohlc.get('close', 0)
                        }
                    }
                }
            else:
                logger.error(f"No quote data received for {symbol}")
                return None

        except ApiException as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def place_order(self, symbol, quantity, price, transaction_type, order_type,
                   variety, exchange, product, tag="Unknown"):
        """
        Place an order on Upstox.

        Args:
            symbol: Instrument token (e.g., "NSE_EQ|INE669E01016")
            quantity: Order quantity
            price: Order price (ignored for MARKET orders)
            transaction_type: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            variety: Order variety (not directly used in Upstox)
            exchange: Exchange segment
            product: Product type (D for delivery, I for intraday)
            tag: Order tag

        Returns:
            str: Order ID if successful, -1 otherwise
        """
        try:
            # Map product types
            product_map = {
                "NRML": "D",  # Delivery
                "MIS": "I",   # Intraday
                "CNC": "D"    # Delivery
            }
            upstox_product = product_map.get(product, "I")

            # Get instrument token
            instrument_token = symbol  # Assume symbol is already in correct format

            logger.info(f"Placing Upstox order: {symbol} {transaction_type} {quantity} @ {price}")

            # Create order request
            body = upstox_client.PlaceOrderRequest(
                quantity=quantity,
                product=upstox_product,
                validity="DAY",
                price=float(price) if order_type == "LIMIT" else 0,
                tag=tag,
                instrument_token=instrument_token,
                order_type=order_type,
                transaction_type=transaction_type,
                disclosed_quantity=0,
                trigger_price=0,
                is_amo=False
            )

            # Place order
            api_response = self.order_api.place_order(body=body, api_version='2.0')

            if api_response and api_response.data:
                order_id = api_response.data.order_id
                logger.info(f"Upstox order placed successfully: {order_id}")
                return order_id
            else:
                logger.error(f"Upstox order placement failed: {api_response}")
                return -1

        except ApiException as e:
            logger.error(f"Error placing order: {e}")
            return -1

    def get_positions(self):
        """Get current positions."""
        try:
            from upstox_client import PortfolioApi
            portfolio_api = PortfolioApi(self.api_client)
            api_response = portfolio_api.get_positions(api_version='2.0')
            return api_response
        except ApiException as e:
            logger.error(f"Error fetching positions: {e}")
            return None

    def download_instruments(self):
        """
        Download instrument master from Upstox.

        Upstox provides instruments via their master contract API.
        """
        try:
            # Upstox doesn't have a simple CSV download
            # You need to use their instruments API or download CSV manually
            logger.warning("Upstox instruments download - using manual implementation")

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
        logger.debug(f"Upstox ticks: {ticks}")

    def on_connect(self, ws, response):
        """
        WebSocket connection callback.
        Override this method in your implementation.
        """
        logger.info(f"Upstox WebSocket connected")

    def on_order_update(self, ws, data):
        """
        Order update callback.
        Override this method in your implementation.
        """
        logger.info(f"Upstox order update: {data}")

    def on_close(self, ws, code, reason):
        """WebSocket close callback."""
        logger.info(f"Upstox WebSocket closed")

    def on_error(self, ws, error):
        """WebSocket error callback."""
        logger.error(f"Upstox WebSocket error: {error}")

    def connect_websocket(self):
        """
        Connect to Upstox WebSocket V3 for live market data.
        """
        try:
            # Initialize market data streamer
            self.upstox_ws = upstox_client.MarketDataStreamerV3(
                self.api_client,
                self.symbols,
                "full"  # Mode: ltpc, full, option_greeks
            )

            # Set up event handlers
            def on_message(message):
                """Handle incoming WebSocket messages."""
                try:
                    # Parse and forward to callback
                    self.on_ticks(self.upstox_ws, message)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

            def on_error(error):
                """Handle WebSocket errors."""
                self.on_error(self.upstox_ws, error)

            def on_close():
                """Handle WebSocket close."""
                self.on_close(self.upstox_ws, None, None)

            def on_open():
                """Handle WebSocket open."""
                self.on_connect(self.upstox_ws, None)

            # Attach event handlers
            self.upstox_ws.on("message", on_message)
            self.upstox_ws.on("error", on_error)
            self.upstox_ws.on("close", on_close)
            self.upstox_ws.on("open", on_open)

            # Connect in separate thread
            def run_streamer():
                try:
                    self.upstox_ws.connect()
                except Exception as e:
                    logger.error(f"Error in WebSocket streamer: {e}")

            ws_thread = Thread(target=run_streamer, daemon=True)
            ws_thread.start()

            logger.info("Upstox WebSocket V3 connected successfully")

        except Exception as e:
            logger.error(f"Error connecting Upstox WebSocket: {e}")

    def symbols_to_subscribe(self, symbols):
        """Set symbols for WebSocket subscription."""
        self.symbols = symbols
