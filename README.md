# trading-algo
Code for certain trading strategies
1. Survivor Algo is Live

## Disclaimer:
This algorithm is provided for **educational** and **informational purposes** only. Trading in financial markets involves substantial risk, and you may lose all or more than your initial investment. By using this algorithm, you acknowledge that all trading decisions are made at your own risk and discretion. The creators of this algorithm assume no liability or responsibility for any financial losses or damages incurred through its use. **Always do your own research and consult with a qualified financial advisor before trading.**


## Setup

### 1. Install Dependencies

To insall uv, use:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or


```bash
pip install uv
```

This uses `uv` for dependency management. Install dependencies:
```bash
uv sync
```

Or if you prefer using pip:

```bash
pip install -r requirements.txt  # You may need to generate this from pyproject.toml
```

### 2. Environment Configuration

1. Copy the sample environment file:
   ```bash
   cp .sample.env .env
   ```

2. Edit `.env` and fill in your broker credentials:
   ```bash
   # Broker Configuration - Supports Zerodha, Fyers, Dhan, Upstox
   BROKER_NAME=zerodha  # Options: zerodha, fyers, dhan, upstox

   # Common credentials (fill based on your broker)
   BROKER_API_KEY=<YOUR_API_KEY>
   BROKER_API_SECRET=<YOUR_API_SECRET>
   BROKER_ID=<YOUR_BROKER_ID>
   BROKER_PASSWORD=<YOUR_BROKER_PASSWORD>

   # For Zerodha/Fyers (TOTP authentication)
   BROKER_TOTP_ENABLE=false  # Set to 'true' for TOTP login
   BROKER_TOTP_KEY=<YOUR_TOTP_KEY>
   BROKER_TOTP_PIN=<YOUR_TOTP_PIN>
   BROKER_TOTP_REDIDRECT_URI=<YOUR_REDIRECT_URI>

   # For Upstox (OAuth2 authentication)
   BROKER_REDIRECT_URI=http://localhost
   BROKER_ACCESS_TOKEN=<YOUR_ACCESS_TOKEN>  # Optional
   ```

### 3. Running Strategies

Strategies should be placed in the `strategy/` folder.

#### Running the Survivor Strategy


**Basic usage (using default config):**
```bash
cd strategy/
python survivor.py
```

**With custom parameters:**
```bash
cd strategy/
python survivor.py \
    --symbol-initials NIFTY25JAN30 \
    --pe-gap 25 --ce-gap 25 \
    --pe-quantity 50 --ce-quantity 50 \
    --min-price-to-sell 15
```

**View current configuration:**
```bash
cd strategy/
python survivor.py --show-config
```

### 4. Available Brokers

All four major Indian brokers are supported with full trading and market data capabilities:

| Broker | Authentication | Market Data | Order Execution | Installation |
|--------|---------------|-------------|-----------------|--------------|
| **Zerodha** | TOTP / Manual Login | KiteTicker WebSocket | KiteConnect API | `pip install kiteconnect` |
| **Fyers** | TOTP (Required) | Fyers WebSocket | Fyers REST API | `pip install fyers-apiv3` |
| **Dhan** | API Key (Simple) | DhanHQ WebSocket | DhanHQ API | `pip install dhanhq` |
| **Upstox** | OAuth2 | Upstox WebSocket V3 | Upstox REST API | `pip install upstox-client` |

#### Broker-Specific Setup:

**Zerodha**
- Get API credentials from [Kite Connect](https://kite.trade/)
- Supports both TOTP and manual login flow
- Most comprehensive instrument data available

**Fyers**
- Get API credentials from [Fyers API](https://myapi.fyers.in/)
- TOTP authentication mandatory
- Requires redirect URI configuration

**Dhan**
- Get access token from [DhanHQ Developer](https://dhanhq.co/)
- Simplest authentication (just client ID + access token)
- No TOTP or OAuth complexity

**Upstox**
- Get API credentials from [Upstox Developer API](https://upstox.com/developer/)
- Uses OAuth2 flow
- WebSocket V3 with improved stability

### 5. Core Components

- `brokers/`: Broker implementations (Zerodha, Fyers, Dhan, Upstox)
- `dispatcher.py`: Data routing and queue management
- `orders.py`: Order management utilities
- `logger.py`: Logging configuration
- `strategy/`: Place your trading strategies here

### Example Usage

```python
import os
from brokers.zerodha import ZerodhaBroker
from brokers.fyers import FyersBroker
from brokers.dhan import DhanBroker
from brokers.upstox import UpstoxBroker

# Initialize broker based on environment
broker_name = os.getenv('BROKER_NAME', 'zerodha')

if broker_name == 'zerodha':
    broker = ZerodhaBroker(without_totp=True)  # Set to False for TOTP login
elif broker_name == 'fyers':
    broker = FyersBroker(symbols=['NSE:SBIN-EQ'])
elif broker_name == 'dhan':
    broker = DhanBroker()
elif broker_name == 'upstox':
    broker = UpstoxBroker(without_oauth=True)  # Set to False if using env token
else:
    raise ValueError(f"Unsupported broker: {broker_name}")

# Use broker for trading
broker.get_quote("NSE:NIFTY 50")
broker.place_order(...)
broker.connect_websocket()
```

For more details, check the individual broker implementations and example strategies in the `strategy/` folder.

## Web Dashboard

A modern web-based UI for monitoring and controlling your trading strategies.

### Features

- 📊 **Real-time monitoring** - Live market data and strategy status
- 🎮 **Strategy controls** - Start/stop strategies with one click
- ⚙️ **Configuration management** - Edit strategy parameters through UI
- 📋 **Order tracking** - View all orders and positions in real-time
- 📈 **Performance metrics** - Track P&L and statistics
- 🔄 **Live updates** - WebSocket-based real-time data streaming
- 🌐 **Multi-broker support** - Switch between brokers seamlessly

### Running the Dashboard

**Start the web server:**
```bash
python web_app.py
```

Or using uvicorn directly:
```bash
uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
```

**Access the dashboard:**
Open your browser and navigate to:
```
http://localhost:8000
```

### Dashboard Screenshots

The dashboard provides:
- **Control Panel**: Start/stop strategies, view status
- **Market Data**: Real-time NIFTY prices and changes
- **Orders Tab**: Complete order history with timestamps
- **Positions Tab**: Open positions with P&L tracking
- **Logs Tab**: Live system logs and events
- **Configuration Modal**: Edit strategy parameters on-the-fly

### API Endpoints

The web app exposes a REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get system status |
| `/api/config` | GET | Get current configuration |
| `/api/config` | POST | Update configuration |
| `/api/strategy/start` | POST | Start trading strategy |
| `/api/strategy/stop` | POST | Stop trading strategy |
| `/api/orders` | GET | Get all orders |
| `/api/positions` | GET | Get current positions |
| `/api/brokers` | GET | List available brokers |
| `/api/broker/switch` | POST | Switch broker |
| `/api/logs` | GET | Get system logs |
| `/ws` | WebSocket | Real-time updates |

### Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Real-time**: WebSocket
- **Styling**: Custom CSS with dark theme
- **Icons**: Font Awesome 6

### Development Mode

Run with auto-reload for development:
```bash
uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

For production, use a production-grade ASGI server:
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Gunicorn with Uvicorn workers:
```bash
gunicorn web_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
