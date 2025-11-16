"""
OpenAlgo Survivor Options Trading Strategy

This is the OpenAlgo-compatible version of the Survivor strategy.
Connects to OpenAlgo server for broker-agnostic trading.

Key Differences from Original:
- Uses OpenAlgo client instead of custom BrokerGateway
- Simplified broker abstraction
- Standardized order placement format
- Compatible with any broker supported by OpenAlgo

Requirements:
- pip install openalgo
- OpenAlgo server running (default: http://127.0.0.1:5000)
- Valid API key for OpenAlgo server
"""

import os
import sys
import time
import yaml
import argparse
from datetime import datetime
from queue import Queue
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from logger import logger
from dispatcher import DataDispatcher
from orders import OrderTracker

warnings.filterwarnings("ignore")


class OpenAlgoSurvivorStrategy:
    """
    Survivor Options Trading Strategy - OpenAlgo Version

    This strategy implements a systematic approach to options trading based on price movements
    of the NIFTY index. The core concept is to sell options (both PE and CE) when the underlying
    index moves beyond certain thresholds, capturing premium decay while managing risk through
    dynamic gap adjustments.

    STRATEGY OVERVIEW:
    ==================

    1. **Dual-Side Trading**: The strategy monitors both upward and downward movements:
       - PE (Put) Trading: Triggered when NIFTY price moves UP beyond pe_gap threshold
       - CE (Call) Trading: Triggered when NIFTY price moves DOWN beyond ce_gap threshold

    2. **Gap-Based Execution**:
       - Maintains reference points (nifty_pe_last_value, nifty_ce_last_value)
       - Executes trades when price deviates beyond configured gaps
       - Uses multipliers to scale position sizes based on gap magnitude

    3. **Dynamic Strike Selection**:
       - Selects option strikes based on symbol_gap from current price
       - Adjusts strikes if option premium is below minimum threshold
       - Ensures adequate liquidity and pricing

    4. **Reset Mechanism**:
       - Automatically adjusts reference points when market moves favorably
       - Prevents excessive accumulation of positions
       - Maintains strategy responsiveness to market conditions

    CONFIGURATION PARAMETERS:
    ========================

    Core Parameters:
    - symbol_initials: Option series identifier (e.g., 'NIFTY25JAN30')
    - index_symbol: Underlying index for tracking (e.g., 'NIFTY 50')

    Gap Parameters:
    - pe_gap/ce_gap: Price movement thresholds to trigger trades
    - pe_symbol_gap/ce_symbol_gap: Strike distance from current price
    - pe_reset_gap/ce_reset_gap: Favorable movement thresholds for reference reset

    Quantity & Risk:
    - pe_quantity/ce_quantity: Base quantities for each trade
    - min_price_to_sell: Minimum option premium threshold
    - sell_multiplier_threshold: Maximum position scaling limit

    OpenAlgo Specific:
    - api_key: Your OpenAlgo API key
    - host: OpenAlgo server URL (default: http://127.0.0.1:5000)
    - exchange: Exchange code (NSE, NFO, etc.)
    """

    def __init__(self, openalgo_client, config, order_tracker, instruments_df):
        """
        Initialize the OpenAlgo Survivor Strategy

        Args:
            openalgo_client: OpenAlgo API client instance
            config: Configuration dictionary
            order_tracker: OrderTracker instance for order management
            instruments_df: Pre-loaded instruments DataFrame
        """
        # Assign config values as instance variables with 'strat_var_' prefix
        for k, v in config.items():
            setattr(self, f'strat_var_{k}', v)

        # External dependencies
        self.client = openalgo_client
        self.symbol_initials = self.strat_var_symbol_initials
        self.order_tracker = order_tracker

        # Use pre-loaded instruments
        self.instruments = instruments_df
        self.instruments = self.instruments[
            self.instruments['symbol'].str.contains(self.symbol_initials)
        ]

        if self.instruments.shape[0] == 0:
            logger.error(f"No instruments found for {self.symbol_initials}")
            logger.error(f"Instrument {self.symbol_initials} not found. Please check the symbol initials")
            return

        self.strike_difference = None
        self._initialize_state()
        self.lot_size = self.instruments['lot_size'].iloc[0]

        # Calculate and store strike difference for the option series
        self.strike_difference = self._get_strike_difference(self.symbol_initials)
        logger.info(f"Strike difference for {self.symbol_initials} is {self.strike_difference}")

    def _nifty_quote(self):
        """Get current quote for NIFTY index using OpenAlgo"""
        try:
            symbol = self.strat_var_index_symbol
            exchange = self.strat_var_exchange

            # OpenAlgo quotes API call
            response = self.client.quotes(symbol=symbol, exchange=exchange)

            if response and 'data' in response:
                # Extract LTP from response
                ltp = float(response['data'].get('ltp', 0))

                # Create a simple quote object
                class Quote:
                    def __init__(self, last_price):
                        self.last_price = last_price

                return Quote(ltp)
            else:
                logger.error(f"Invalid quote response for {symbol}: {response}")
                return None

        except Exception as e:
            logger.error(f"Error getting quote for {self.strat_var_index_symbol}: {e}")
            return None

    def _initialize_state(self):
        """Initialize strategy state with starting reference values"""
        # Initialize reset flags - these track when reset conditions are triggered
        self.pe_reset_gap_flag = 0  # Set to 1 when PE trade is executed
        self.ce_reset_gap_flag = 0  # Set to 1 when CE trade is executed

        # Get current market data for initialization
        current_quote = self._nifty_quote()

        if current_quote is None:
            logger.error("Failed to get initial quote. Using default values.")
            self.nifty_pe_last_value = 24000  # Fallback value
            self.nifty_ce_last_value = 24000  # Fallback value
            return

        # Initialize PE reference value
        if self.strat_var_pe_start_point == 0:
            # Use current market price as starting reference
            self.nifty_pe_last_value = current_quote.last_price
            logger.debug(f"Nifty PE Start Point is 0, so using LTP: {self.nifty_pe_last_value}")
        else:
            # Use configured starting point
            self.nifty_pe_last_value = self.strat_var_pe_start_point

        # Initialize CE reference value
        if self.strat_var_ce_start_point == 0:
            # Use current market price as starting reference
            self.nifty_ce_last_value = current_quote.last_price
            logger.debug(f"Nifty CE Start Point is 0, so using LTP: {self.nifty_ce_last_value}")
        else:
            # Use configured starting point
            self.nifty_ce_last_value = self.strat_var_ce_start_point

        logger.info(f"Nifty PE Start Value during initialization: {self.nifty_pe_last_value}, "
                   f"Nifty CE Start Value during initialization: {self.nifty_ce_last_value}")

    def _get_strike_difference(self, symbol_initials):
        """Calculate the strike difference for the option series"""
        if self.strike_difference is not None:
            return self.strike_difference

        # Filter for CE instruments to calculate strike difference
        ce_instruments = self.instruments[
            self.instruments['symbol'].str.contains(symbol_initials) &
            self.instruments['symbol'].str.endswith('CE')
        ]

        if ce_instruments.shape[0] < 2:
            logger.error(f"Not enough CE instruments found for {symbol_initials} to calculate strike difference")
            return 0

        # Sort by strike
        ce_instruments_sorted = ce_instruments.sort_values('strike')
        # Take the top 2
        top2 = ce_instruments_sorted.head(2)
        # Calculate the difference
        self.strike_difference = abs(top2.iloc[1]['strike'] - top2.iloc[0]['strike'])
        return self.strike_difference

    def on_ticks_update(self, ticks):
        """
        Main strategy execution method called on each tick update

        Args:
            ticks (dict): Market data containing 'last_price' or 'ltp'
        """
        current_price = ticks['last_price'] if 'last_price' in ticks else ticks.get('ltp', 0)

        if current_price == 0:
            logger.warning("Received invalid price (0), skipping tick")
            return

        # Process trading opportunities for both sides
        self._handle_pe_trade(current_price)  # Handle Put option opportunities
        self._handle_ce_trade(current_price)  # Handle Call option opportunities

        # Apply reset logic to adjust reference values
        self._reset_reference_values(current_price)

    def _check_sell_multiplier_breach(self, sell_multiplier):
        """
        Risk management check for position scaling

        Args:
            sell_multiplier (int): The calculated multiplier for position sizing

        Returns:
            bool: True if multiplier exceeds threshold, False otherwise
        """
        if sell_multiplier > self.strat_var_sell_multiplier_threshold:
            logger.warning(f"Sell multiplier {sell_multiplier} breached the threshold {self.strat_var_sell_multiplier_threshold}")
            return True
        return False

    def _handle_pe_trade(self, current_price):
        """
        Handle PE (Put) option trading logic

        Args:
            current_price (float): Current NIFTY index price
        """
        # No action needed if price hasn't moved up sufficiently
        if current_price <= self.nifty_pe_last_value:
            self._log_stable_market(current_price)
            return

        # Calculate price difference and check if it exceeds gap threshold
        price_diff = round(current_price - self.nifty_pe_last_value, 0)
        if price_diff > self.strat_var_pe_gap:
            # Calculate multiplier for position sizing
            sell_multiplier = int(price_diff / self.strat_var_pe_gap)

            # Risk check: Ensure multiplier doesn't exceed threshold
            if self._check_sell_multiplier_breach(sell_multiplier):
                return

            # Update reference value based on executed gaps
            self.nifty_pe_last_value += self.strat_var_pe_gap * sell_multiplier

            # Calculate total quantity to trade
            total_quantity = sell_multiplier * self.strat_var_pe_quantity

            # Find suitable PE option with adequate premium
            temp_gap = self.strat_var_pe_symbol_gap
            while True:
                # Find PE instrument at specified gap from current price
                instrument = self._find_nifty_symbol_from_gap("PE", current_price, gap=temp_gap)
                if not instrument:
                    logger.warning("No suitable instrument found for PE with gap %s", temp_gap)
                    return

                # Get current quote using OpenAlgo
                symbol = instrument['symbol']
                quote_price = self._get_option_quote(symbol)

                if quote_price is None:
                    logger.warning(f"Failed to get quote for {symbol}")
                    return

                # Check if premium meets minimum threshold
                if quote_price < self.strat_var_min_price_to_sell:
                    logger.info(f"Last price {quote_price} is less than min price to sell {self.strat_var_min_price_to_sell}")
                    # Try closer strike if premium is too low
                    temp_gap -= self.lot_size
                    continue

                # Execute the trade
                logger.info(f"Execute PE sell @ {symbol} × {total_quantity}, Market Price")
                self._place_order(symbol, total_quantity)

                # Set reset flag to enable reset logic
                self.pe_reset_gap_flag = 1
                break

    def _handle_ce_trade(self, current_price):
        """
        Handle CE (Call) option trading logic

        Args:
            current_price (float): Current NIFTY index price
        """
        # No action needed if price hasn't moved down sufficiently
        if current_price >= self.nifty_ce_last_value:
            self._log_stable_market(current_price)
            return

        # Calculate price difference and check if it exceeds gap threshold
        price_diff = round(self.nifty_ce_last_value - current_price, 0)
        if price_diff > self.strat_var_ce_gap:
            # Calculate multiplier for position sizing
            sell_multiplier = int(price_diff / self.strat_var_ce_gap)

            # Risk check: Ensure multiplier doesn't exceed threshold
            if self._check_sell_multiplier_breach(sell_multiplier):
                return

            # Update reference value based on executed gaps
            self.nifty_ce_last_value -= self.strat_var_ce_gap * sell_multiplier

            # Calculate total quantity to trade
            total_quantity = sell_multiplier * self.strat_var_ce_quantity

            # Find suitable CE option with adequate premium
            temp_gap = self.strat_var_ce_symbol_gap
            while True:
                # Find CE instrument at specified gap from current price
                instrument = self._find_nifty_symbol_from_gap("CE", current_price, gap=temp_gap)
                if not instrument:
                    logger.warning("No suitable instrument found for CE with gap %s", temp_gap)
                    return

                # Get current quote using OpenAlgo
                symbol = instrument['symbol']
                quote_price = self._get_option_quote(symbol)

                if quote_price is None:
                    logger.warning(f"Failed to get quote for {symbol}")
                    return

                # Check if premium meets minimum threshold
                if quote_price < self.strat_var_min_price_to_sell:
                    logger.info(f"Last price {quote_price} is less than min price to sell {self.strat_var_min_price_to_sell}, trying next strike")
                    # Try closer strike if premium is too low
                    temp_gap -= self.lot_size
                    continue

                # Execute the trade
                logger.info(f"Execute CE sell @ {symbol} × {total_quantity}, Market Price")
                self._place_order(symbol, total_quantity)

                # Set reset flag to enable reset logic
                self.ce_reset_gap_flag = 1
                break

    def _reset_reference_values(self, current_price):
        """
        Reset reference values when market moves favorably

        Args:
            current_price (float): Current NIFTY index price
        """
        # PE Reset Logic: Reset when price drops significantly below PE reference
        if (self.nifty_pe_last_value - current_price) > self.strat_var_pe_reset_gap and self.pe_reset_gap_flag:
            logger.info(f"Resetting PE value from {self.nifty_pe_last_value} to {current_price + self.strat_var_pe_reset_gap}")
            # Reset PE reference to current price plus reset gap
            self.nifty_pe_last_value = current_price + self.strat_var_pe_reset_gap

        # CE Reset Logic: Reset when price rises significantly above CE reference
        if (current_price - self.nifty_ce_last_value) > self.strat_var_ce_reset_gap and self.ce_reset_gap_flag:
            logger.info(f"Resetting CE value from {self.nifty_ce_last_value} to {current_price - self.strat_var_ce_reset_gap}")
            # Reset CE reference to current price minus reset gap
            self.nifty_ce_last_value = current_price - self.strat_var_ce_reset_gap

    def _find_nifty_symbol_from_gap(self, option_type, ltp, gap):
        """
        Find the most suitable option instrument based on strike distance from current price

        Args:
            option_type (str): 'PE' or 'CE'
            ltp (float): Last traded price of the underlying
            gap (int): Distance from current price to target strike

        Returns:
            dict: Instrument details or None if not found
        """
        # Convert gap to symbol_gap based on option type
        if option_type == "PE":
            symbol_gap = -gap  # Negative for PE (below current price)
        else:
            symbol_gap = gap   # Positive for CE (above current price)

        # Calculate target strike price
        target_strike = ltp + symbol_gap

        # Filter instruments for matching criteria
        df = self.instruments[
            (self.instruments['symbol'].str.contains(self.strat_var_symbol_initials)) &
            (self.instruments['instrument_type'] == option_type) &
            (self.instruments['segment'] == "NFO-OPT")
        ]

        if df.empty:
            return None

        # Find closest strike within acceptable tolerance
        df['target_strike_diff'] = (df['strike'] - target_strike).abs()

        # Filter to strikes within half strike difference (tolerance for rounding)
        tolerance = self._get_strike_difference(self.strat_var_symbol_initials) / 2
        df = df[df['target_strike_diff'] <= tolerance]

        if df.empty:
            logger.error(f"No instrument found for {self.strat_var_symbol_initials} {option_type} "
                        f"within {tolerance} of {target_strike}")
            return None

        # Return the closest match
        best = df.sort_values('target_strike_diff').iloc[0]
        return best.to_dict()

    def _get_option_quote(self, symbol):
        """
        Get quote for an option symbol using OpenAlgo

        Args:
            symbol (str): Option symbol

        Returns:
            float: Last traded price or None if failed
        """
        try:
            response = self.client.quotes(
                symbol=symbol,
                exchange=self.strat_var_exchange
            )

            if response and 'data' in response:
                return float(response['data'].get('ltp', 0))
            else:
                logger.error(f"Invalid quote response for {symbol}: {response}")
                return None

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    def _place_order(self, symbol, quantity):
        """
        Execute order placement through OpenAlgo

        Args:
            symbol (str): Trading symbol for the option
            quantity (int): Number of lots/shares to trade
        """
        try:
            # Place order using OpenAlgo client
            response = self.client.placeorder(
                strategy=self.strat_var_tag,  # Strategy identifier
                symbol=symbol,
                action="SELL",  # OpenAlgo uses BUY/SELL
                exchange=self.strat_var_exchange,
                price_type="MARKET",
                product=self.strat_var_product_type,  # MIS, CNC, NRML
                quantity=quantity
            )

            logger.debug(f"Order placement response: {response}")

            # Check response status
            if response and response.get('status') == 'success':
                order_id = response.get('orderid', 'unknown')
                logger.info(f"Order placed successfully: {order_id} - SELL {symbol} × {quantity}")

                # Track the order
                order_details = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "transaction_type": "SELL",
                    "quantity": quantity,
                    "price": None,  # Market order
                    "timestamp": datetime.now().isoformat(),
                }

                # Optionally add to order tracker
                # self.order_tracker.add_order(order_details)

                logger.info(f"Survivor order tracked: {order_id} - SELL {symbol} × {quantity}")
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                logger.error(f"Order placement failed for {symbol} × {quantity}: {error_msg}")

        except Exception as e:
            logger.error(f"Exception during order placement for {symbol} × {quantity}: {e}", exc_info=True)

    def _log_stable_market(self, current_val):
        """Log current market state when no trading action is taken"""
        logger.info(
            f"{self.strat_var_symbol_initials} Nifty under control. "
            f"PE = {self.nifty_pe_last_value}, "
            f"CE = {self.nifty_ce_last_value}, "
            f"Current = {current_val}, "
            f"CE Gap = {self.strat_var_ce_gap}, "
            f"PE Gap = {self.strat_var_pe_gap}"
        )


# =============================================================================
# MAIN SCRIPT EXECUTION FOR OPENALGO VERSION
# =============================================================================

if __name__ == "__main__":
    import logging
    from brokers import BrokerGateway  # For downloading instruments only

    logger.setLevel(logging.INFO)

    # Load configuration
    config_file = os.path.join(os.path.dirname(__file__), "configs/survivor_openalgo.yml")
    if not os.path.exists(config_file):
        config_file = os.path.join(os.path.dirname(__file__), "configs/survivor.yml")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)['default']

    # Get OpenAlgo credentials from environment
    api_key = os.getenv("OPENALGO_API_KEY")
    host = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

    if not api_key:
        logger.error("OPENALGO_API_KEY not set in environment variables")
        logger.error("Please set it using: export OPENALGO_API_KEY='your_api_key_here'")
        sys.exit(1)

    logger.info(f"Initializing OpenAlgo client with host: {host}")

    # Initialize OpenAlgo client
    try:
        client = api(api_key=api_key, host=host)
        logger.info("OpenAlgo client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAlgo client: {e}")
        sys.exit(1)

    # Download instruments using broker gateway (one-time operation)
    logger.info("Downloading instruments list...")
    broker_name = os.getenv("BROKER_NAME", "zerodha")
    broker = BrokerGateway.from_name(broker_name)
    broker.download_instruments()
    instruments = broker.get_instruments()
    logger.info(f"Downloaded {len(instruments)} instruments")

    # Create order tracker
    order_tracker = OrderTracker()

    # Initialize strategy
    strategy = OpenAlgoSurvivorStrategy(client, config, order_tracker, instruments)

    # Setup data dispatcher
    dispatcher = DataDispatcher()
    dispatcher.register_main_queue(Queue())

    # Define websocket callbacks
    def on_ticks(ws, ticks):
        logger.debug("Received ticks: {}".format(ticks))
        if isinstance(ticks, list):
            dispatcher.dispatch(ticks)
        else:
            if "symbol" in ticks:
                dispatcher.dispatch(ticks)

    def on_connect(ws, response):
        logger.info("Websocket connected successfully: {}".format(response))

    def on_order_update(ws, data):
        logger.info(f"Order update received: {data}")

    # Assign callbacks
    broker.on_ticks = on_ticks
    broker.on_connect = on_connect
    broker.on_order_update = on_order_update

    # Start websocket
    instrument_token = config['index_symbol']
    broker.connect_websocket(on_ticks=on_ticks, on_connect=on_connect)
    broker.symbols_to_subscribe([instrument_token])
    broker.connect_order_websocket(on_order_update=on_order_update)
    time.sleep(10)

    logger.info("Starting trading loop...")

    # Main trading loop
    try:
        while True:
            try:
                tick_data = dispatcher._main_queue.get()

                if isinstance(tick_data, list):
                    symbol_data = tick_data[0]
                else:
                    symbol_data = tick_data

                if isinstance(symbol_data, dict) and ('last_price' in symbol_data or 'ltp' in symbol_data):
                    strategy.on_ticks_update(symbol_data)

            except KeyboardInterrupt:
                logger.info("SHUTDOWN REQUESTED - Stopping strategy...")
                break

            except Exception as tick_error:
                logger.error(f"Error processing tick data: {tick_error}", exc_info=True)
                continue

    except Exception as fatal_error:
        logger.error("FATAL ERROR in main trading loop:")
        logger.error(f"Error: {fatal_error}")
        import traceback
        traceback.print_exc()

    finally:
        logger.info("STRATEGY SHUTDOWN COMPLETE")
