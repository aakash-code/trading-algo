# OpenAlgo Trading Strategies

This repository now includes OpenAlgo-compatible versions of the trading strategies, allowing you to trade with any broker supported by OpenAlgo.

## Table of Contents

- [What is OpenAlgo?](#what-is-openalgo)
- [Why Use OpenAlgo?](#why-use-openalgo)
- [Installation](#installation)
- [Setup](#setup)
- [Available Strategies](#available-strategies)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Differences from Original Strategies](#differences-from-original-strategies)
- [Troubleshooting](#troubleshooting)

## What is OpenAlgo?

OpenAlgo is a unified trading platform that provides a broker-agnostic API for algorithmic trading in India. Instead of writing separate code for each broker (Zerodha, Fyers, Angel One, etc.), you write once and trade with any supported broker through OpenAlgo.

**OpenAlgo Documentation:** https://docs.openalgo.in/

## Why Use OpenAlgo?

### Benefits

1. **Broker Independence**: Switch brokers without changing your strategy code
2. **Unified API**: One consistent API instead of learning multiple broker APIs
3. **Simplified Development**: Less code, fewer dependencies, easier maintenance
4. **Multi-Broker Support**: Trade with Zerodha, Fyers, Angel One, Upstox, and more
5. **Centralized Management**: Manage all your trading through a single interface

### Original vs OpenAlgo Comparison

| Aspect | Original (This Repo) | OpenAlgo Version |
|--------|---------------------|------------------|
| **Broker Support** | Zerodha, Fyers (custom code per broker) | Any OpenAlgo-supported broker |
| **Code Complexity** | Custom BrokerGateway abstraction layer | Simple OpenAlgo client |
| **Dependencies** | kiteconnect, fyers-apiv3, custom broker code | Just `openalgo` package |
| **Switching Brokers** | Requires code changes | Just reconfigure OpenAlgo server |
| **Order Placement** | `broker.place_order(OrderRequest(...))` | `client.placeorder(strategy="...", ...)` |

## Installation

### 1. Install OpenAlgo Python Library

```bash
pip install openalgo
```

### 2. Set Up OpenAlgo Server

Follow the official OpenAlgo setup guide: https://docs.openalgo.in/getting-started/installation

The OpenAlgo server acts as a bridge between your Python strategies and your broker. You configure your broker credentials in the OpenAlgo web interface (not in your code).

### 3. Clone This Repository

```bash
git clone <repository-url>
cd trading-algo
pip install -r requirements.txt
```

## Setup

### 1. Configure OpenAlgo Server

1. Access OpenAlgo web interface (typically http://localhost:5000)
2. Configure your broker credentials
3. Generate an API key for your strategy

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.openalgo.example .env
```

Edit `.env` and add your OpenAlgo credentials:

```bash
# OpenAlgo API credentials
OPENALGO_API_KEY=your_api_key_here
OPENALGO_HOST=http://127.0.0.1:5000

# Broker for downloading instruments (optional)
BROKER_NAME=zerodha
```

### 3. Configure Strategy Parameters

Edit the strategy configuration files:

- **Survivor Strategy**: `strategy/configs/survivor_openalgo.yml`
- **Wave Strategy**: `strategy/configs/wave_openalgo.yml`

Update parameters like:
- Symbol names (e.g., NIFTY25JAN30)
- Gap thresholds
- Position sizes
- Risk management limits

## Available Strategies

### 1. OpenAlgo Survivor Strategy

**File:** `strategy/openalgo_survivor.py`
**Config:** `strategy/configs/survivor_openalgo.yml`

Systematic options selling strategy based on NIFTY index movements.

**Features:**
- Sells PE options when NIFTY rises beyond threshold
- Sells CE options when NIFTY falls beyond threshold
- Dynamic strike selection based on premium requirements
- Automatic reference value reset mechanism
- Position scaling with multiplier system

**When to Use:**
- Range-bound or moderately trending markets
- Premium collection strategies
- Low-frequency trading (a few trades per day)

### 2. OpenAlgo Wave Strategy (Simplified)

**File:** `strategy/openalgo_wave.py`
**Config:** `strategy/configs/wave_openalgo.yml`

Simplified wave trading strategy for directional positions.

**Features:**
- Buy on price dips, sell on price rises
- Dynamic gap scaling based on position imbalance
- Position tracking and management
- Cooldown period between orders

**When to Use:**
- Trending markets
- Mean-reversion strategies
- Higher-frequency trading

**Note:** This is a simplified version without advanced Greeks calculations. For delta-neutral hedging and complex Greeks management, use the original `wave.py` with custom broker integration.

## Quick Start

### Survivor Strategy

```python
# Run the OpenAlgo Survivor strategy
python strategy/openalgo_survivor.py
```

The strategy will:
1. Connect to OpenAlgo server
2. Download instrument list
3. Initialize with current NIFTY price
4. Start monitoring for trading opportunities
5. Place orders automatically when conditions are met

### Wave Strategy

```python
# Run the OpenAlgo Wave strategy
python strategy/openalgo_wave.py
```

Or use it programmatically:

```python
from openalgo import api
from strategy.openalgo_wave import OpenAlgoWaveStrategy
import yaml

# Load configuration
with open('strategy/configs/wave_openalgo.yml', 'r') as f:
    config = yaml.safe_load(f)['default']

# Initialize OpenAlgo client
client = api(api_key='your_api_key', host='http://127.0.0.1:5000')

# Create strategy instance
strategy = OpenAlgoWaveStrategy(config, client)

# Place a wave order
strategy.place_wave_order()
```

## Configuration

### Survivor Strategy Configuration

Key parameters in `survivor_openalgo.yml`:

```yaml
# Core parameters
index_symbol: "NIFTY 50"              # Underlying index
symbol_initials: "NIFTY25JAN30"       # Option series

# Gap parameters (trigger thresholds)
pe_gap: 20                            # Points move up to trigger PE sell
ce_gap: 20                            # Points move down to trigger CE sell

# Position sizing
pe_quantity: 75                       # Base PE quantity
ce_quantity: 75                       # Base CE quantity

# Strike selection
pe_symbol_gap: 200                    # PE strike distance from spot
ce_symbol_gap: 200                    # CE strike distance from spot

# Risk management
min_price_to_sell: 15                 # Minimum premium threshold
sell_multiplier_threshold: 5          # Max position scaling

# OpenAlgo specific
exchange: "NFO"                       # Exchange code
product_type: "NRML"                  # Product type (MIS/CNC/NRML)
```

### Wave Strategy Configuration

Key parameters in `wave_openalgo.yml`:

```yaml
# Core parameters
symbol_name: "NIFTY25JANFUT"         # Trading symbol
exchange: "NFO"                       # Exchange

# Gap parameters
buy_gap: 25                           # Price drop to trigger buy
sell_gap: 25                          # Price rise to trigger sell
cool_off_time: 10                     # Cooldown (seconds)

# Position sizing
buy_quantity: 75                      # Buy order quantity
sell_quantity: 75                     # Sell order quantity
lot_size: 75                          # Instrument lot size

# OpenAlgo specific
product_type: "NRML"                  # Product type
```

## Usage Examples

### Example 1: Basic Survivor Strategy

```python
import os
from openalgo import api
from strategy.openalgo_survivor import OpenAlgoSurvivorStrategy
import yaml

# Load config
with open('strategy/configs/survivor_openalgo.yml', 'r') as f:
    config = yaml.safe_load(f)['default']

# Initialize OpenAlgo client
client = api(
    api_key=os.getenv('OPENALGO_API_KEY'),
    host=os.getenv('OPENALGO_HOST', 'http://127.0.0.1:5000')
)

# Download instruments (one-time)
from brokers import BrokerGateway
broker = BrokerGateway.from_name('zerodha')
broker.download_instruments()
instruments = broker.get_instruments()

# Create strategy
from orders import OrderTracker
tracker = OrderTracker()
strategy = OpenAlgoSurvivorStrategy(client, config, tracker, instruments)

# Main loop - process market ticks
while True:
    # Get tick data from your data source (websocket, etc.)
    tick = get_market_tick()  # Your implementation

    # Process tick
    strategy.on_ticks_update(tick)
```

### Example 2: Wave Strategy with Custom Logic

```python
from strategy.openalgo_wave import OpenAlgoWaveStrategy

# Create strategy (as above)
strategy = OpenAlgoWaveStrategy(config, client)

# Custom trading loop
while trading_hours():
    # Check if we should place orders
    current_position = strategy._get_position_for_symbol()

    # Place wave order if no active orders
    if not strategy.check_is_any_order_active():
        strategy.place_wave_order()

    # Wait before next check
    time.sleep(60)  # Check every minute
```

### Example 3: Testing with Paper Trading

```python
# For testing, you can configure OpenAlgo with paper trading
# or use the fyrodha broker (simulated broker in this repo)

# Set in .env:
# BROKER_NAME=fyrodha

# OpenAlgo strategies work the same way
# All orders go through OpenAlgo, which can be configured
# for paper trading or live trading
```

## Differences from Original Strategies

### Survivor Strategy

| Feature | Original | OpenAlgo Version |
|---------|----------|-----------------|
| Broker Integration | Custom BrokerGateway | OpenAlgo client |
| Order Placement | `broker.place_order(OrderRequest(...))` | `client.placeorder(strategy="...", ...)` |
| Quote Retrieval | `broker.get_quote(symbol)` | `client.quotes(symbol=..., exchange=...)` |
| Dependencies | kiteconnect/fyers-apiv3 | openalgo |
| Symbol Format | Exchange prefix (e.g., "NFO:SYMBOL") | Simple name (e.g., "SYMBOL") |
| Core Strategy Logic | **Identical** | **Identical** |

### Wave Strategy

| Feature | Original | OpenAlgo Version |
|---------|----------|-----------------|
| Greeks Calculation | ✅ Full (using mibian) | ❌ Not included (simplified) |
| Delta Management | ✅ Full delta-neutral hedging | ❌ Simplified |
| Position Tracking | ✅ Advanced with delta limits | ✅ Basic position tracking |
| Order Pairing | ✅ Complex associated orders | ✅ Simple buy/sell |
| Broker Integration | Custom BrokerGateway | OpenAlgo client |
| Core Wave Logic | ✅ Full | ✅ Core logic preserved |

**Recommendation:**
- Use `openalgo_wave.py` for basic wave trading
- Use original `wave.py` for advanced delta-neutral strategies

## Troubleshooting

### Common Issues

#### 1. "OPENALGO_API_KEY not set"

**Problem:** Environment variable not configured

**Solution:**
```bash
export OPENALGO_API_KEY='your_api_key_here'
# Or add to .env file
```

#### 2. "Connection refused to OpenAlgo server"

**Problem:** OpenAlgo server not running

**Solution:**
```bash
# Start OpenAlgo server (follow OpenAlgo docs)
# Verify it's running: http://localhost:5000
```

#### 3. "Invalid quote response"

**Problem:** Symbol name format incorrect

**Solution:**
- Check symbol format in OpenAlgo (usually no exchange prefix)
- Verify symbol exists in your broker's instruments
- Example: Use "NIFTY 50" not "NSE:NIFTY 50"

#### 4. "Order placement failed"

**Problem:** Broker not configured in OpenAlgo or insufficient funds

**Solution:**
- Verify broker credentials in OpenAlgo web interface
- Check account balance and margins
- Review OpenAlgo logs for detailed error messages

#### 5. "No instruments found"

**Problem:** Instrument download failed or symbol_initials incorrect

**Solution:**
```python
# Verify instruments downloaded
broker = BrokerGateway.from_name('zerodha')
broker.download_instruments()
instruments = broker.get_instruments()

# Check if symbol exists
nifty_options = instruments[instruments['symbol'].str.contains('NIFTY25')]
print(nifty_options.head())
```

### Getting Help

1. **OpenAlgo Documentation**: https://docs.openalgo.in/
2. **OpenAlgo Community**: Check Discord/Telegram (links in OpenAlgo docs)
3. **This Repository Issues**: For strategy-specific questions

## Advanced Topics

### Custom Broker Integration

While OpenAlgo versions are recommended for simplicity, you can still use the original strategies with custom broker integration if you need:

- Advanced Greeks calculations (Wave strategy)
- Direct broker API access
- Custom order management logic
- Broker-specific features

See the original `strategy/survivor.py` and `strategy/wave.py` for reference.

### Extending OpenAlgo Strategies

You can extend the OpenAlgo strategies with custom logic:

```python
from strategy.openalgo_survivor import OpenAlgoSurvivorStrategy

class MySurvivorStrategy(OpenAlgoSurvivorStrategy):
    def _handle_pe_trade(self, current_price):
        # Add custom pre-trade checks
        if self.custom_condition():
            super()._handle_pe_trade(current_price)

    def custom_condition(self):
        # Your custom logic
        return True
```

### Running Multiple Strategies

You can run multiple strategies simultaneously:

```python
# Strategy 1: Survivor for NIFTY weekly options
survivor_config = load_config('survivor_openalgo.yml')
survivor = OpenAlgoSurvivorStrategy(client, survivor_config, tracker1, instruments)

# Strategy 2: Wave for NIFTY futures
wave_config = load_config('wave_openalgo.yml')
wave = OpenAlgoWaveStrategy(wave_config, client, tracker2)

# Run both in parallel (use threading or async)
```

## License

Same as the main repository.

## Contributing

Contributions are welcome! Please:
1. Test thoroughly with paper trading first
2. Maintain compatibility with OpenAlgo API
3. Update documentation for any changes
4. Follow existing code style

## Disclaimer

**IMPORTANT:** These are algorithmic trading strategies that can result in financial losses.

- Always test with paper trading first
- Understand the strategy logic before using real money
- Use proper risk management
- Monitor your positions actively
- Past performance does not guarantee future results

The authors and contributors are not responsible for any financial losses incurred from using these strategies.
