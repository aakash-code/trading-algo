"""
OpenAlgo Wave Trading Strategy (Simplified)

This is a simplified OpenAlgo-compatible version of the Wave strategy.
The full Wave strategy includes complex Greeks calculations and delta-neutral hedging
which may require additional adaptations for OpenAlgo.

This version maintains the core wave trading logic:
- Buy on dips, sell on rises
- Dynamic gap scaling based on position imbalance
- Position tracking and management

Requirements:
- pip install openalgo
- OpenAlgo server running (default: http://127.0.0.1:5000)
- Valid API key for OpenAlgo server

Note: For advanced features like Greeks-based delta management,
you may need to use the original wave.py with custom broker integration.
"""

import os
import sys
import time
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from logger import logger
from orders import OrderTracker

load_dotenv()


class OpenAlgoWaveStrategy:
    """
    Simplified Wave Trading Strategy - OpenAlgo Version

    Core concept: Buy on price dips, sell on price rises with dynamic gap scaling

    Features:
    - Wave-based positioning: Create buy/sell orders based on price movements
    - Dynamic gap scaling: Adjust gaps based on position imbalance
    - Position tracking: Monitor and manage positions through OpenAlgo
    """

    def __init__(self, config: Dict, openalgo_client, order_tracker=None):
        """
        Initialize the OpenAlgo Wave Strategy

        Args:
            config: Configuration dictionary
            openalgo_client: OpenAlgo API client instance
            order_tracker: OrderTracker instance for order management
        """
        # Core configuration
        self.config = config
        self.client = openalgo_client
        self.symbol_name = config.get("symbol_name", None)
        self.exchange = config.get("exchange", "NFO")
        self.buy_gap = float(config["buy_gap"])
        self.sell_gap = float(config["sell_gap"])
        self.cool_off_time = int(config["cool_off_time"])
        self.buy_quantity = int(config["buy_quantity"])
        self.sell_quantity = int(config["sell_quantity"])
        self.product_type = config.get("product_type", "NRML")
        self.tag = config.get("tag", "WAVE_OPENALGO")
        self.lot_size = int(config.get("lot_size", 1))

        # Order tracking
        self.order_tracker = order_tracker

        # System state
        self.scraper_last_price = 0
        self.already_executing_order = 0
        self.initial_positions = {}
        self.orders = {}  # Active orders tracking

        # Price tracking
        self.prev_wave_sell_price = None
        self.prev_wave_buy_price = None
        self.prev_quote_price = None

        # Generate multiplier scale for gap scaling
        self.multiplier_scale = self._generate_multiplier_scale()

        # Initialize position tracking
        self.initial_positions['position'] = self._get_position_for_symbol()

        # Get initial market price
        initial_quote = self._get_quote()
        if initial_quote:
            self.scraper_last_price = initial_quote
        else:
            logger.error("Failed to get initial quote")

        logger.info(f"System initialized for {self.symbol_name}")
        logger.info(f"Initial position: {self.initial_positions['position']}, Last Price: {self.scraper_last_price}")

    def _generate_multiplier_scale(self, levels: int = 10) -> Dict[str, List[float]]:
        """Generate multiplier scale for dynamic gap scaling based on position imbalance"""
        buy_scale = [1.3, 1.7, 2.5, 3, 10, 10, 10, 15, 15, 15]
        sell_scale = [1.3, 1.7, 2.5, 3, 10, 10, 10, 15, 15, 15]

        multiplier_scale = {"0": [1.0, 1.0]}  # Neutral position

        for i in range(1, levels + 1):
            multiplier_scale[str(i)] = [buy_scale[i - 1], 1.0]
            multiplier_scale[str(-i)] = [1.0, sell_scale[i - 1]]

        return multiplier_scale

    def _get_quote(self) -> float:
        """Get current quote for the trading symbol using OpenAlgo"""
        try:
            response = self.client.quotes(
                symbol=self.symbol_name,
                exchange=self.exchange
            )

            if response and 'data' in response:
                return float(response['data'].get('ltp', 0))
            else:
                logger.error(f"Invalid quote response for {self.symbol_name}: {response}")
                return 0

        except Exception as e:
            logger.error(f"Error getting quote for {self.symbol_name}: {e}")
            return 0

    def _get_position_for_symbol(self) -> int:
        """Get current position quantity for the trading symbol using OpenAlgo"""
        try:
            response = self.client.positionbook()

            if not response or 'data' not in response:
                logger.warning("No position data available")
                return 0

            positions = response['data']

            for position in positions:
                if position.get('symbol') == self.symbol_name:
                    net_qty = int(position.get('netqty', 0))
                    logger.info(f"Symbol: {self.symbol_name} | Current Position: {net_qty}")
                    return net_qty

            return 0

        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return 0

    def _get_scaled_gaps(self, current_diff_scale: float) -> Tuple[float, float]:
        """Calculate scaled gaps based on position imbalance"""
        diff_key = str(int(current_diff_scale))

        if diff_key not in self.multiplier_scale:
            mult = [100.0, 1.0] if current_diff_scale > 0 else [1.0, 100.0]
        else:
            mult = self.multiplier_scale[diff_key]

        return round(self.buy_gap * mult[0], 1), round(self.sell_gap * mult[1], 1)

    def _get_best_buy_sell_price(self, buy_price_1: float, buy_price_2: float,
                                sell_price_1: float, sell_price_2: float) -> Dict[str, float]:
        """Get the best prices for buy (lower) and sell (higher) orders"""
        return {'buy': min(buy_price_1, buy_price_2), 'sell': max(sell_price_1, sell_price_2)}

    def _prepare_final_prices(self, scaled_buy_gap: float, scaled_sell_gap: float) -> Dict[str, float]:
        """Prepare final order prices with cool-off period"""
        price = self._get_quote()
        self.prev_quote_price = price

        best_prices = self._get_best_buy_sell_price(
            price - scaled_buy_gap, self.scraper_last_price - scaled_buy_gap,
            price + scaled_sell_gap, self.scraper_last_price + scaled_sell_gap
        )

        time.sleep(self.cool_off_time)

        price_after_wait = self._get_quote()
        return self._get_best_buy_sell_price(
            best_prices['buy'], price_after_wait - scaled_buy_gap,
            best_prices['sell'], price_after_wait + scaled_sell_gap
        )

    def _execute_orders(self, final_buy_price: float, final_sell_price: float,
                       restrict_buy_order: int, restrict_sell_order: int) -> None:
        """Execute buy and sell orders using OpenAlgo"""
        sell_order_id = None

        logger.info(f"Executing orders for {self.symbol_name} | Restrictions - Buy: {restrict_buy_order}, Sell: {restrict_sell_order}")

        # Place sell order if not restricted
        if restrict_sell_order == 0:
            try:
                response = self.client.placeorder(
                    strategy=self.tag,
                    symbol=self.symbol_name,
                    action="SELL",
                    exchange=self.exchange,
                    price_type="LIMIT",
                    product=self.product_type,
                    quantity=self.sell_quantity,
                    price=final_sell_price
                )

                if response and response.get('status') == 'success':
                    sell_order_id = response.get('orderid', 'unknown')
                    logger.info(f"Placed SELL order {sell_order_id} for {self.sell_quantity} @ {final_sell_price}")
                    self.add_order_to_list(sell_order_id, final_sell_price, self.sell_quantity, "SELL", self.symbol_name)
                else:
                    error_msg = response.get('message', 'Unknown error') if response else 'No response'
                    logger.error(f"Sell order failed: {error_msg}")

            except Exception as e:
                logger.error(f"Exception placing sell order: {e}", exc_info=True)

        # Place buy order if not restricted (and sell order succeeded or was restricted)
        if (restrict_sell_order == 1 or sell_order_id is not None) and restrict_buy_order == 0:
            try:
                response = self.client.placeorder(
                    strategy=self.tag,
                    symbol=self.symbol_name,
                    action="BUY",
                    exchange=self.exchange,
                    price_type="LIMIT",
                    product=self.product_type,
                    quantity=self.buy_quantity,
                    price=final_buy_price
                )

                if response and response.get('status') == 'success':
                    buy_order_id = response.get('orderid', 'unknown')
                    logger.info(f"Placed BUY order {buy_order_id} for {self.buy_quantity} @ {final_buy_price}")
                    self.add_order_to_list(buy_order_id, final_buy_price, self.buy_quantity, "BUY", self.symbol_name)
                else:
                    error_msg = response.get('message', 'Unknown error') if response else 'No response'
                    logger.error(f"Buy order failed: {error_msg}")

            except Exception as e:
                logger.error(f"Exception placing buy order: {e}", exc_info=True)

    def add_order_to_list(self, order_id, price, quantity, transaction_type, symbol):
        """Add order to tracking list"""
        now = datetime.now()

        # Create order details for OrderTracker
        order_details = {
            'order_id': order_id,
            'price': price,
            'quantity': quantity,
            'transaction_type': transaction_type,
            'symbol': symbol,
            'hour': now.hour,
            'min': now.minute,
            'second': now.second,
            'time': f"{now.hour}:{now.minute}:{now.second}",
            'timestamp': now.isoformat()
        }

        # Add to OrderTracker if available
        if self.order_tracker:
            self.order_tracker.add_order(order_details)

        # Keep local reference
        self.orders[order_id] = order_details
        logger.info("Current Orders List: {}".format(self.orders))
        self.print_current_status()

    def get_current_position_difference(self) -> float:
        """Calculate position difference from initial position"""
        current_net = self._get_position_for_symbol()
        quantity = self.sell_quantity if self.sell_quantity != 0 else 1

        return (current_net - self.initial_positions['position']) / quantity

    def place_wave_order(self) -> None:
        """Main function to execute wave trading strategy"""
        if self.already_executing_order > 0:
            logger.info("Order execution already in progress.")
            return

        self.already_executing_order = 1

        try:
            logger.info("--- Starting New Wave Order Cycle ---")

            # For simplified version, we don't have complex delta restrictions
            restrict_buy_order, restrict_sell_order = 0, 0

            current_diff_scale = self.get_current_position_difference()

            logger.info(f"Position Imbalance: {current_diff_scale:.2f}")

            scaled_buy_gap, scaled_sell_gap = self._get_scaled_gaps(current_diff_scale)
            logger.info(f"Scaled Gaps | Buy: {scaled_buy_gap}, Sell: {scaled_sell_gap}")

            final_prices = self._prepare_final_prices(scaled_buy_gap, scaled_sell_gap)
            logger.info(f"Final Prices -> Buy: {final_prices['buy']:.2f}, Sell: {final_prices['sell']:.2f}")

            # Incorporate previous wave prices if available
            if self.prev_wave_buy_price is not None:
                final_prices = self._get_best_buy_sell_price(
                    final_prices['buy'], self.prev_wave_buy_price,
                    final_prices['sell'], self.prev_wave_sell_price
                )

            logger.info(f"Adjusted Prices -> Buy: {final_prices['buy']:.2f}, Sell: {final_prices['sell']:.2f}")

            self._execute_orders(final_prices['buy'], final_prices['sell'], restrict_buy_order, restrict_sell_order)

            self.prev_wave_buy_price = final_prices['buy']
            self.prev_wave_sell_price = final_prices['sell']

            logger.info(f"Previous Wave Prices -> Buy: {self.prev_wave_buy_price}, Sell: {self.prev_wave_sell_price}")

            logger.info("--- End of Wave Order Cycle ---")
            time.sleep(3)

        except Exception as e:
            logger.error(f"Error in wave order execution: {e}", exc_info=True)
        finally:
            self.already_executing_order = 0

    def check_is_any_order_active(self) -> bool:
        """Check if any orders are currently active"""
        logger.info("Current Order List - {}".format(self.orders))
        return len(self.orders) > 0

    def print_current_status(self) -> None:
        """Print current system status"""
        current_position = self._get_position_for_symbol()

        # Use OrderTracker for status if available
        if self.order_tracker:
            additional_info = {
                'symbol': self.symbol_name,
                'initial_position': self.initial_positions['position'],
                'current_position': current_position
            }
            self.order_tracker.print_status(additional_info)
        else:
            logger.info("="*50)
            logger.info(f"STATUS as of {time.ctime()}")
            logger.info(f"Symbol: {self.symbol_name}, Initial Position: {self.initial_positions['position']}, Current Position: {current_position}")
            logger.info(f"Active Orders Tracked: {len(self.orders)}")
            if self.orders:
                logger.info(f"Tracked Orders: {self.orders}")
            logger.info("="*50)


# =============================================================================
# MAIN SCRIPT EXECUTION FOR OPENALGO VERSION
# =============================================================================

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    # Load configuration
    config_file = os.path.join(os.path.dirname(__file__), "configs/wave_openalgo.yml")
    if not os.path.exists(config_file):
        config_file = os.path.join(os.path.dirname(__file__), "configs/wave.yml")

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

    # Create order tracker
    order_tracker = OrderTracker()

    # Initialize strategy
    strategy = OpenAlgoWaveStrategy(config, client, order_tracker)

    logger.info("Wave strategy initialized. Ready to place orders.")
    logger.info("Call strategy.place_wave_order() to execute wave trading logic.")

    # Example: Place initial wave order
    # strategy.place_wave_order()

    # For continuous operation, you would typically run this in a loop
    # or trigger based on market events/conditions
