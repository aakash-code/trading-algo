# Web Dashboard Quick Start Guide

## Overview

The Trading Algorithm Dashboard is a modern web interface built with **Tailwind CSS** for monitoring and controlling your trading strategies in real-time.

## Features ✨

- 🎨 **Modern UI** - Beautiful dark theme with Tailwind CSS
- 📊 **Real-time Data** - WebSocket-powered live updates
- 🎮 **Strategy Controls** - Start/stop strategies with one click
- ⚙️ **Live Configuration** - Update parameters without restarting
- 📋 **Order Tracking** - View all orders and positions
- 📈 **Market Data** - Live NIFTY prices and changes
- 🔔 **System Logs** - Real-time log streaming
- 🌐 **Multi-broker** - Switch between Zerodha, Fyers, Dhan, Upstox

## Installation

### 1. Install Dependencies

```bash
cd trading-algo
uv sync
```

Or with pip:
```bash
pip install fastapi uvicorn jinja2 websockets
```

### 2. Start the Web Server

```bash
python web_app.py
```

Or:
```bash
uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the Dashboard

Open your browser:
```
http://localhost:8000
```

## Dashboard Sections

### 📊 Header
- **Broker Selector**: Switch between different brokers
- **Connection Status**: Real-time WebSocket connection indicator

### 🎮 Strategy Controls
- **Status Indicator**: Shows if strategy is running or stopped
- **Start Button**: Launch your trading strategy
- **Stop Button**: Halt the running strategy
- **Configure Button**: Open configuration modal

### 📈 Stats Cards
- **NIFTY 50**: Live price and percentage change
- **Total Orders**: Count of all placed orders
- **Open Positions**: Current active positions
- **P&L Today**: Daily profit/loss

### 📋 Tabs

#### Orders Tab
- View all placed orders
- Real-time order updates
- Order details: Time, Symbol, Type, Quantity, Price, Status
- Refresh button for manual updates

#### Positions Tab
- Track open positions
- Real-time P&L calculation
- Position details: Symbol, Quantity, Avg Price, LTP, P&L
- Close position button

#### Logs Tab
- Live system logs
- Color-coded by level (INFO, SUCCESS, WARNING, ERROR)
- Scrollable log container
- Clear logs button

### ⚙️ Configuration Modal

Edit strategy parameters:
- **Option Series**: e.g., NIFTY25JAN30
- **PE/CE Gap**: Price movement thresholds
- **PE/CE Quantity**: Position sizes
- **Min Premium**: Minimum option premium
- **PE/CE Strike Gap**: Strike distance from spot

## API Endpoints

### Status & Info
```http
GET /api/status          # Get system status
GET /api/brokers         # List available brokers
GET /api/market-data     # Get current market data
```

### Configuration
```http
GET  /api/config         # Get current config
POST /api/config         # Update configuration
```

### Strategy Control
```http
POST /api/strategy/start # Start strategy
POST /api/strategy/stop  # Stop strategy
```

### Data
```http
GET /api/orders          # Get all orders
GET /api/positions       # Get current positions
GET /api/logs            # Get system logs
```

### Broker Management
```http
POST /api/broker/switch  # Switch broker
```

### WebSocket
```
ws://localhost:8000/ws   # Real-time updates
```

## WebSocket Messages

The dashboard receives real-time updates via WebSocket:

```javascript
// Message types
{
    "type": "initial_state",
    "data": { /* initial state */ }
}

{
    "type": "strategy_status",
    "data": { "running": true }
}

{
    "type": "market_data",
    "data": { /* market data */ }
}

{
    "type": "order_update",
    "data": { /* order details */ }
}

{
    "type": "log",
    "data": { "message": "...", "level": "INFO" }
}
```

## Customization

### Tailwind Configuration

Edit `templates/dashboard.html` to customize Tailwind:

```javascript
tailwind.config = {
    theme: {
        extend: {
            colors: {
                primary: '#2563eb',  // Change primary color
                success: '#10b981',  // Change success color
                // Add more colors...
            }
        }
    }
}
```

### Modify Styles

The dashboard uses Tailwind utility classes. To customize:

1. **Change colors**: Update color classes (e.g., `bg-blue-600` to `bg-purple-600`)
2. **Adjust spacing**: Modify padding/margin classes (e.g., `p-6` to `p-8`)
3. **Responsive design**: Add responsive prefixes (e.g., `md:grid-cols-4`)

## Development Tips

### Hot Reload

Run with `--reload` for automatic reloading during development:
```bash
uvicorn web_app:app --reload
```

### Debug Mode

Enable detailed logging in `web_app.py`:
```python
logger.setLevel(logging.DEBUG)
```

### Custom Endpoints

Add new endpoints in `web_app.py`:
```python
@app.get("/api/custom")
async def custom_endpoint():
    return {"data": "your data"}
```

## Production Deployment

### Using Uvicorn
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn
```bash
pip install gunicorn
gunicorn web_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Behind Nginx

Example Nginx configuration:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### SSL/HTTPS

For WebSocket over HTTPS (WSS):
```javascript
// app.js will auto-detect protocol
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
```

## Troubleshooting

### WebSocket Connection Failed
- Check if server is running
- Verify port 8000 is accessible
- Check firewall rules

### Orders Not Showing
- Ensure broker is configured in `.env`
- Check if `artifacts/orders_data.json` exists
- Verify broker authentication

### Configuration Not Saving
- Check file permissions on `strategy/configs/survivor.yml`
- Ensure strategy is stopped before updating config

### Logs Not Appearing
- Refresh the page
- Check browser console for errors
- Verify WebSocket connection

## Browser Support

Tested on:
- ✅ Chrome/Edge (Latest)
- ✅ Firefox (Latest)
- ✅ Safari (Latest)
- ⚠️ Internet Explorer (Not supported)

## Mobile Responsive

The dashboard is fully responsive:
- 📱 Mobile: Single column layout
- 📱 Tablet: 2-column grid
- 💻 Desktop: 4-column grid

## Security Notes

⚠️ **Important Security Considerations:**

1. **Never expose to public internet** without authentication
2. Use **HTTPS** in production
3. Add **authentication middleware** before deployment
4. Set **strong firewall rules**
5. Use **environment variables** for sensitive data

## Need Help?

- 📖 Check the main [README.md](README.md)
- 🐛 Report issues on GitHub
- 💬 Ask questions in discussions

## License

Same as the main project license.

---

**Happy Trading! 🚀📈**
