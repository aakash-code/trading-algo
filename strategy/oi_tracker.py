import os
import sys
import time
import yaml
import argparse
from datetime import datetime, timedelta
from collections import deque

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
import beepy as bp

# Imports are resolved by the __main__ block
from logger import logger
from brokers.zerodha import ZerodhaBroker
from brokers.dhan import DhanBroker

class OITrackerStrategy:
    def __init__(self, broker, config):
        self.broker = broker
        self.config = config
        self.console = Console()
        self.instruments_df = None
        self.live_data_buffer = {}
        self.alerts_triggered = 0
        self.signals = {}

        self.broker.download_instruments()
        self.instruments_df = self.broker.get_instruments()
        if self.instruments_df is None or self.instruments_df.empty:
            logger.error("Failed to download instruments. Exiting.")
            sys.exit(1)

    def get_atm_strikes(self, ltp, strike_diff, num_strikes=5):
        atm_strike = round(ltp / strike_diff) * strike_diff
        return [atm_strike + i * strike_diff for i in range(- (num_strikes // 2), (num_strikes // 2) + 1)]

    def run(self):
        with Live(self.generate_layout(), screen=True, auto_refresh=False) as live:
            while True:
                try:
                    all_symbols_to_fetch = set()
                    watchlists = [w for w in self.config['watchlists'] if w['enabled']]

                    for watchlist in watchlists:
                        quote = self.broker.get_quote(watchlist['underlying_symbol'])
                        if not quote:
                            logger.warning(f"Could not fetch quote for {watchlist['underlying_symbol']}. Skipping.")
                            continue

                        ltp = quote[watchlist['underlying_symbol']]['last_price']
                        atm_strikes = self.get_atm_strikes(ltp, watchlist['strike_difference'])

                        call_symbols = [f"{watchlist['option_symbol_prefix']}{strike}CE" for strike in atm_strikes]
                        put_symbols = [f"{watchlist['option_symbol_prefix']}{strike}PE" for strike in atm_strikes]

                        all_symbols_to_fetch.update(call_symbols)
                        all_symbols_to_fetch.update(put_symbols)
                        all_symbols_to_fetch.add(watchlist['underlying_symbol'])

                    current_data = self._fetch_live_data(list(all_symbols_to_fetch))
                    self._update_data_buffer(current_data)
                    self._generate_signals()

                    self.alerts_triggered = 0
                    layout = self.generate_layout(current_data)
                    live.update(layout, refresh=True)

                    if self.alerts_triggered > self.config['alert_threshold']:
                        bp.beep(sound='ping')

                    time.sleep(self.config['refresh_interval'])
                except KeyboardInterrupt:
                    logger.info("Stopping OI Tracker...")
                    break
                except Exception as e:
                    logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
                    time.sleep(self.config['refresh_interval'])

    def _fetch_live_data(self, symbols):
        live_data = {}
        try:
            quotes = self.broker.get_quote(symbols)
            for symbol, data in quotes.items():
                live_data[symbol] = {
                    'timestamp': datetime.now(),
                    'oi': data.get('oi', 0),
                    'price': data['last_price']
                }
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
        return live_data

    def _update_data_buffer(self, current_data):
        max_buffer_size = 3 * 60
        for symbol, data in current_data.items():
            if symbol not in self.live_data_buffer:
                self.live_data_buffer[symbol] = deque(maxlen=max_buffer_size)
            self.live_data_buffer[symbol].append(data)

    def _generate_signals(self):
        self.signals = {}
        for watchlist in self.config['watchlists']:
            if not watchlist['enabled']:
                continue

            underlying_symbol = watchlist['underlying_symbol']
            interval = watchlist['signal_generation_interval']

            _, underlying_pct = self.calculate_change(underlying_symbol, interval, is_underlying=True)

            atm_strikes = self.get_atm_strikes(self.live_data_buffer[underlying_symbol][-1]['price'], watchlist['strike_difference'])
            call_symbols = [f"{watchlist['option_symbol_prefix']}{strike}CE" for strike in atm_strikes]
            put_symbols = [f"{watchlist['option_symbol_prefix']}{strike}PE" for strike in atm_strikes]

            for symbol in call_symbols + put_symbols:
                if symbol not in self.live_data_buffer:
                    continue

                _, oi_pct = self.calculate_change(symbol, interval)

                if "CE" in symbol:
                    if oi_pct > watchlist['call_buy_oi_change_p_threshold'] and underlying_pct > watchlist['price_change_p_for_confirmation']:
                        self.signals[symbol] = "[bold green]BUY[/bold green]"
                    elif oi_pct < watchlist['call_sell_oi_change_p_threshold'] and underlying_pct < -watchlist['price_change_p_for_confirmation']:
                        self.signals[symbol] = "[bold red]SELL[/bold red]"
                elif "PE" in symbol:
                    if oi_pct > watchlist['put_buy_oi_change_p_threshold'] and underlying_pct < -watchlist['price_change_p_for_confirmation']:
                        self.signals[symbol] = "[bold green]BUY[/bold green]"
                    elif oi_pct < watchlist['put_sell_oi_change_p_threshold'] and underlying_pct > watchlist['price_change_p_for_confirmation']:
                        self.signals[symbol] = "[bold red]SELL[/bold red]"

    def generate_layout(self, current_data=None):
        if current_data is None:
            current_data = {}

        main_layout = Layout()
        watchlists = [w for w in self.config['watchlists'] if w['enabled']]

        for watchlist in watchlists:
            underlying_symbol = watchlist['underlying_symbol']
            ltp = current_data.get(underlying_symbol, {}).get('price', 0)
            atm_strikes = self.get_atm_strikes(ltp, watchlist['strike_difference'])
            option_prefix = watchlist['option_symbol_prefix']

            call_symbols = [f"{option_prefix}{strike}CE" for strike in atm_strikes]
            put_symbols = [f"{option_prefix}{strike}PE" for strike in atm_strikes]

            table_call = self.create_options_table(f"{underlying_symbol} Calls", call_symbols, current_data)
            table_put = self.create_options_table(f"{underlying_symbol} Puts", put_symbols, current_data)
            table_underlying = self.create_underlying_table(underlying_symbol, [underlying_symbol], current_data)

            grid = Table.grid(expand=True)
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            grid.add_row(Panel(table_call), Panel(table_put), Panel(table_underlying))
            main_layout.split_column(grid)

        return main_layout

    def create_options_table(self, title, symbols, current_data):
        table = Table(title=title)
        headers = ["Symbol", "Signal", "Current OI/Price"] + [f"{i}m Δ" for i in self.config['intervals']]
        for h in headers:
            table.add_column(h, justify="right")

        for symbol in symbols:
            row_data = self.get_row_data(symbol, current_data)
            table.add_row(*row_data)
        return table

    def create_underlying_table(self, title, symbols, current_data):
        table = Table(title=title)
        headers = ["Symbol", "Current Price"] + [f"{i}m Δ" for i in self.config['intervals']]
        for h in headers:
            table.add_column(h, justify="right")

        for symbol in symbols:
            row_data = self.get_row_data(symbol, current_data, is_underlying=True)
            table.add_row(*row_data)
        return table

    def get_row_data(self, symbol, current_data, is_underlying=False):
        row = [symbol]
        current = current_data.get(symbol, {})

        if is_underlying:
            row.append(f"{current.get('price', 0):.2f}")
        else:
            row.insert(1, self.signals.get(symbol, "---"))
            row.append(f"{current.get('oi', 0)} ({current.get('price', 0):.2f})")

        for interval in self.config['intervals']:
            change_str, _ = self.calculate_change(symbol, interval, is_underlying)
            # Use global alert percentages for highlighting
            _, is_alert = self.calculate_change(symbol, interval, is_underlying, use_global_alert_config=True)
            if is_alert:
                self.alerts_triggered += 1
                row.append(f"[bold red]{change_str}[/bold red]")
            else:
                row.append(change_str)
        return row

    def calculate_change(self, symbol, interval_minutes, is_underlying=False, use_global_alert_config=False):
        if symbol not in self.live_data_buffer or len(self.live_data_buffer[symbol]) < 2:
            return "N/A", 0.0

        now = datetime.now()
        past_time = now - timedelta(minutes=interval_minutes)

        buffer = self.live_data_buffer[symbol]
        current_data = buffer[-1]

        past_data = min(buffer, key=lambda x: abs(x['timestamp'] - past_time))

        if not past_data:
            return "N/A", 0.0

        if is_underlying:
            price_change = current_data['price'] - past_data['price']
            price_change_pct = (price_change / past_data['price']) * 100 if past_data['price'] != 0 else 0
            alert = abs(price_change_pct) > self.config['nifty_alert_percentage']
            return f"{price_change:+.2f} ({price_change_pct:+.2f}%)", price_change_pct
        else:
            oi_change = current_data['oi'] - past_data['oi']
            oi_change_pct = (oi_change / past_data['oi']) * 100 if past_data['oi'] != 0 else 0
            alert = abs(oi_change_pct) > self.config['oi_alert_percentage']
            return f"{oi_change:+} ({oi_change_pct:+.2f}%)", oi_change_pct

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from logger import logger
    from brokers.zerodha import ZerodhaBroker
    from brokers.dhan import DhanBroker

    parser = argparse.ArgumentParser(description="OI Tracker Strategy")
    parser.add_argument('--broker', type=str, default='zerodha', choices=['zerodha', 'dhan'], help='Broker to use')
    parser.add_argument('--config', type=str, default='strategy/configs/oi_tracker.yml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.broker == 'zerodha':
        broker = ZerodhaBroker(without_totp=True)
    elif args.broker == 'dhan':
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        if not client_id or not access_token:
            raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set for Dhan broker")
        broker = DhanBroker(client_id, access_token)
    else:
        raise ValueError("Invalid broker specified")

    strategy = OITrackerStrategy(broker, config)
    strategy.run()