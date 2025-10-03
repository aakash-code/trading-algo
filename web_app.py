"""
Trading Algorithm Web Dashboard
================================
A FastAPI-based web interface for monitoring and controlling trading strategies.

Features:
- Real-time market data and strategy monitoring
- Strategy start/stop controls
- Configuration management
- Order and position tracking
- Live logs viewer
- Broker selection and management

Run with: uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import yaml
import asyncio
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="Trading Algorithm Dashboard",
    description="Web interface for trading strategies",
    version="1.0.0"
)

# Set up static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# =============================================================================
# GLOBAL STATE MANAGEMENT
# =============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.strategy_running = False
        self.strategy_process = None
        self.current_broker = os.getenv("BROKER_NAME", "zerodha")
        self.broker_connected = False  # NEW: Track broker connection status
        self.broker_status_message = "Not configured"  # NEW: Status message
        self.current_config = {}
        self.market_data = {}
        self.orders = []
        self.positions = []
        self.logs = []
        self.websocket_clients = []

    def add_log(self, message: str, level: str = "INFO"):
        """Add log message"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        # Keep only last 100 logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

state = AppState()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class BrokerConfig(BaseModel):
    broker_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    broker_id: Optional[str] = None
    totp_enabled: bool = False


class StrategyConfig(BaseModel):
    symbol_initials: str
    pe_gap: float
    ce_gap: float
    pe_quantity: int
    ce_quantity: int
    min_price_to_sell: float
    pe_symbol_gap: int = 200
    ce_symbol_gap: int = 200
    pe_reset_gap: float = 30
    ce_reset_gap: float = 30


class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    order_type: str
    transaction_type: str
    price: Optional[float] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config(config_name: str = "survivor") -> Dict:
    """Load strategy configuration from YAML"""
    try:
        config_file = Path(__file__).parent / "strategy" / "configs" / f"{config_name}.yml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('default', {})
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def save_config(config: Dict, config_name: str = "survivor") -> bool:
    """Save strategy configuration to YAML"""
    try:
        config_file = Path(__file__).parent / "strategy" / "configs" / f"{config_name}.yml"
        with open(config_file, 'w') as f:
            yaml.dump({'default': config}, f, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def load_orders() -> List[Dict]:
    """Load orders from JSON file"""
    try:
        orders_file = Path(__file__).parent / "artifacts" / "orders_data.json"
        if orders_file.exists():
            with open(orders_file, 'r') as f:
                orders_dict = json.load(f)
                return [{"order_id": k, **v} for k, v in orders_dict.items()]
        return []
    except Exception as e:
        logger.error(f"Error loading orders: {e}")
        return []


# =============================================================================
# WEBSOCKET MANAGER
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

manager = ConnectionManager()


# =============================================================================
# WEB ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "broker": state.current_broker,
        "strategy_running": state.strategy_running
    })


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "strategy_running": state.strategy_running,
        "broker": state.current_broker,
        "broker_connected": state.broker_connected,
        "broker_status": state.broker_status_message,
        "timestamp": datetime.now().isoformat(),
        "orders_count": len(state.orders),
        "positions_count": len(state.positions)
    }


@app.get("/api/config")
async def get_config():
    """Get current strategy configuration"""
    config = load_config()
    return {"config": config}


@app.post("/api/config")
async def update_config(config: Dict):
    """Update strategy configuration"""
    try:
        if save_config(config):
            state.current_config = config
            state.add_log("Configuration updated successfully", "INFO")
            await manager.broadcast({
                "type": "config_updated",
                "data": config
            })
            return {"status": "success", "message": "Configuration updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save configuration")
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategy/start")
async def start_strategy():
    """Start the trading strategy"""
    if state.strategy_running:
        return {"status": "error", "message": "Strategy is already running"}

    try:
        # In a real implementation, you would start the strategy in a separate thread/process
        # For now, we'll just update the state
        state.strategy_running = True
        state.add_log("Strategy started", "INFO")

        await manager.broadcast({
            "type": "strategy_status",
            "data": {"running": True}
        })

        return {"status": "success", "message": "Strategy started successfully"}
    except Exception as e:
        logger.error(f"Error starting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategy/stop")
async def stop_strategy():
    """Stop the trading strategy"""
    if not state.strategy_running:
        return {"status": "error", "message": "Strategy is not running"}

    try:
        state.strategy_running = False
        state.add_log("Strategy stopped", "INFO")

        await manager.broadcast({
            "type": "strategy_status",
            "data": {"running": False}
        })

        return {"status": "success", "message": "Strategy stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders")
async def get_orders():
    """Get all orders"""
    orders = load_orders()
    return {"orders": orders}


@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    # In real implementation, fetch from broker
    return {"positions": state.positions}


@app.get("/api/brokers")
async def get_brokers():
    """Get available brokers"""
    return {
        "brokers": [
            {
                "name": "zerodha",
                "display_name": "Zerodha",
                "auth_type": "TOTP/Manual",
                "status": "available"
            },
            {
                "name": "fyers",
                "display_name": "Fyers",
                "auth_type": "TOTP",
                "status": "available"
            },
            {
                "name": "dhan",
                "display_name": "Dhan",
                "auth_type": "API Key",
                "status": "available"
            },
            {
                "name": "upstox",
                "display_name": "Upstox",
                "auth_type": "OAuth2",
                "status": "available"
            }
        ],
        "current": state.current_broker
    }


@app.post("/api/broker/switch")
async def switch_broker(broker: BrokerConfig):
    """Switch to a different broker"""
    try:
        # Update environment variable
        os.environ["BROKER_NAME"] = broker.broker_name
        state.current_broker = broker.broker_name

        # Reset broker connection status
        state.broker_connected = False
        state.broker_status_message = "Switched, not authenticated yet"

        state.add_log(f"Switched to {broker.broker_name} broker", "INFO")

        await manager.broadcast({
            "type": "broker_changed",
            "data": {
                "broker": broker.broker_name,
                "connected": state.broker_connected,
                "status": state.broker_status_message
            }
        })

        return {"status": "success", "message": f"Switched to {broker.broker_name}"}
    except Exception as e:
        logger.error(f"Error switching broker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/broker/configure")
async def configure_broker(credentials: Dict):
    """Save broker credentials and test connection"""
    try:
        broker_name = credentials.get("broker_name")
        if not broker_name:
            raise HTTPException(status_code=400, detail="Broker name is required")

        # Update environment variables
        os.environ["BROKER_NAME"] = broker_name

        # Save credentials based on broker type
        if broker_name == "dhan":
            os.environ["BROKER_ID"] = credentials.get("client_id", "")
            os.environ["BROKER_API_KEY"] = credentials.get("access_token", "")
        elif broker_name == "zerodha":
            os.environ["BROKER_API_KEY"] = credentials.get("api_key", "")
            os.environ["BROKER_API_SECRET"] = credentials.get("api_secret", "")
            os.environ["BROKER_ID"] = credentials.get("client_id", "")
            os.environ["BROKER_PASSWORD"] = credentials.get("password", "")
            # Convert boolean to string
            totp_enable = "true" if credentials.get("totp_enable", False) else "false"
            os.environ["BROKER_TOTP_ENABLE"] = totp_enable
            if credentials.get("totp_key"):
                os.environ["BROKER_TOTP_KEY"] = credentials.get("totp_key", "")
        elif broker_name == "fyers":
            os.environ["BROKER_API_KEY"] = credentials.get("api_key", "")
            os.environ["BROKER_API_SECRET"] = credentials.get("api_secret", "")
            os.environ["BROKER_ID"] = credentials.get("client_id", "")
            os.environ["BROKER_TOTP_KEY"] = credentials.get("totp_key", "")
            os.environ["BROKER_TOTP_PIN"] = credentials.get("totp_pin", "")
            os.environ["BROKER_TOTP_REDIDRECT_URI"] = credentials.get("redirect_uri", "")
            os.environ["BROKER_TOTP_ENABLE"] = "true"
        elif broker_name == "upstox":
            os.environ["BROKER_API_KEY"] = credentials.get("api_key", "")
            os.environ["BROKER_API_SECRET"] = credentials.get("api_secret", "")
            os.environ["BROKER_REDIRECT_URI"] = credentials.get("redirect_uri", "http://localhost")
            if credentials.get("access_token"):
                os.environ["BROKER_ACCESS_TOKEN"] = credentials.get("access_token", "")

        # Update .env file
        env_file = Path(__file__).parent / ".env"
        env_content = []

        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.readlines()

        # Update or add BROKER_NAME
        updated = False
        for i, line in enumerate(env_content):
            if line.startswith("BROKER_NAME="):
                env_content[i] = f"BROKER_NAME={broker_name}\n"
                updated = True
                break
        if not updated:
            env_content.append(f"BROKER_NAME={broker_name}\n")

        # Update credentials based on broker
        totp_enable_str = "true" if credentials.get("totp_enable", False) else "false"

        cred_mapping = {
            "dhan": [
                ("BROKER_ID", credentials.get("client_id", "")),
                ("BROKER_API_KEY", credentials.get("access_token", ""))
            ],
            "zerodha": [
                ("BROKER_API_KEY", credentials.get("api_key", "")),
                ("BROKER_API_SECRET", credentials.get("api_secret", "")),
                ("BROKER_ID", credentials.get("client_id", "")),
                ("BROKER_PASSWORD", credentials.get("password", "")),
                ("BROKER_TOTP_ENABLE", totp_enable_str),
                ("BROKER_TOTP_KEY", credentials.get("totp_key", ""))
            ],
            "fyers": [
                ("BROKER_API_KEY", credentials.get("api_key", "")),
                ("BROKER_API_SECRET", credentials.get("api_secret", "")),
                ("BROKER_ID", credentials.get("client_id", "")),
                ("BROKER_TOTP_KEY", credentials.get("totp_key", "")),
                ("BROKER_TOTP_PIN", credentials.get("totp_pin", "")),
                ("BROKER_TOTP_REDIDRECT_URI", credentials.get("redirect_uri", "")),
                ("BROKER_TOTP_ENABLE", "true")
            ],
            "upstox": [
                ("BROKER_API_KEY", credentials.get("api_key", "")),
                ("BROKER_API_SECRET", credentials.get("api_secret", "")),
                ("BROKER_REDIRECT_URI", credentials.get("redirect_uri", "http://localhost")),
                ("BROKER_ACCESS_TOKEN", credentials.get("access_token", ""))
            ]
        }

        for key, value in cred_mapping.get(broker_name, []):
            updated = False
            for i, line in enumerate(env_content):
                if line.startswith(f"{key}="):
                    env_content[i] = f"{key}={value}\n"
                    updated = True
                    break
            if not updated:
                env_content.append(f"{key}={value}\n")

        # Write back to .env file
        with open(env_file, 'w') as f:
            f.writelines(env_content)

        # Try to test connection
        try:
            # Import broker dynamically
            if broker_name == "dhan":
                from brokers.dhan import DhanBroker
                broker = DhanBroker()
                state.broker_connected = True
                state.broker_status_message = "Connected successfully"
            elif broker_name == "zerodha":
                state.broker_connected = False
                state.broker_status_message = "Credentials saved. Start strategy to authenticate."
            elif broker_name == "fyers":
                state.broker_connected = False
                state.broker_status_message = "Credentials saved. Start strategy to authenticate."
            elif broker_name == "upstox":
                # For Upstox, check if access token is provided
                if credentials.get("access_token"):
                    # Try to validate the access token
                    from brokers.upstox import UpstoxBroker
                    broker = UpstoxBroker(without_oauth=False)
                    state.broker_connected = True
                    state.broker_status_message = "Connected successfully"
                else:
                    # Generate OAuth URL for user to authorize
                    api_key = credentials.get("api_key", "")
                    redirect_uri = credentials.get("redirect_uri", "http://localhost:8000")
                    oauth_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}/upstox/callback"

                    state.broker_connected = False
                    state.broker_status_message = "OAuth required"

                    return {
                        "status": "oauth_required",
                        "message": "Please complete OAuth authorization",
                        "oauth_url": oauth_url,
                        "connected": False
                    }

            state.current_broker = broker_name
            state.add_log(f"Broker credentials configured for {broker_name}", "INFO")

            await manager.broadcast({
                "type": "broker_status",
                "data": {
                    "connected": state.broker_connected,
                    "status": state.broker_status_message
                }
            })

            return {
                "status": "success",
                "message": f"{broker_name.capitalize()} configured successfully!",
                "connected": state.broker_connected
            }

        except Exception as conn_error:
            logger.error(f"Connection test failed: {conn_error}")
            state.broker_connected = False
            state.broker_status_message = f"Credentials saved but connection failed: {str(conn_error)}"
            return {
                "status": "warning",
                "message": f"Credentials saved but connection test failed: {str(conn_error)}",
                "connected": False
            }

    except Exception as e:
        logger.error(f"Error configuring broker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/broker/test-connection")
async def test_broker_connection():
    """Test broker connection with current credentials"""
    try:
        # Check if credentials are configured
        broker_name = os.getenv("BROKER_NAME", "zerodha")
        api_key = os.getenv("BROKER_API_KEY")

        if not api_key or api_key.startswith("<INPUT"):
            state.broker_connected = False
            state.broker_status_message = "Credentials not configured in .env"
            return {
                "status": "error",
                "message": "Please configure broker credentials in .env file",
                "connected": False
            }

        # Try to initialize broker (without actual connection for now)
        state.broker_status_message = f"Credentials found for {broker_name}"
        state.broker_connected = False  # Set to false until actual auth is tested

        await manager.broadcast({
            "type": "broker_status",
            "data": {
                "connected": state.broker_connected,
                "status": state.broker_status_message
            }
        })

        return {
            "status": "success",
            "message": f"Found credentials for {broker_name}. Start strategy to connect.",
            "connected": False
        }

    except Exception as e:
        logger.error(f"Error testing broker connection: {e}")
        state.broker_connected = False
        state.broker_status_message = f"Error: {str(e)}"
        return {
            "status": "error",
            "message": str(e),
            "connected": False
        }


@app.get("/upstox/callback")
async def upstox_oauth_callback(code: str = None):
    """Handle Upstox OAuth callback"""
    try:
        if not code:
            return HTMLResponse("""
                <html>
                <head><title>Upstox OAuth - Error</title></head>
                <body style="font-family: Arial; padding: 40px; text-align: center;">
                    <h2 style="color: #ef4444;">Authorization Failed</h2>
                    <p>No authorization code received from Upstox.</p>
                    <button onclick="window.close()" style="padding: 10px 20px; background: #3b82f6; color: white; border: none; border-radius: 5px; cursor: pointer;">Close Window</button>
                </body>
                </html>
            """)

        # Get credentials from environment
        api_key = os.getenv("BROKER_API_KEY")
        api_secret = os.getenv("BROKER_API_SECRET")
        redirect_uri = os.getenv("BROKER_REDIRECT_URI", "http://localhost:8000")

        # Exchange authorization code for access token
        import upstox_client
        from upstox_client import Configuration, ApiClient, LoginApi

        configuration = Configuration()
        api_client = ApiClient(configuration)
        login_api = LoginApi(api_client)

        api_response = login_api.token(
            api_version='2.0',
            code=code,
            client_id=api_key,
            client_secret=api_secret,
            redirect_uri=f"{redirect_uri}/upstox/callback",
            grant_type='authorization_code'
        )

        access_token = api_response.access_token

        # Save access token to environment and .env file
        os.environ["BROKER_ACCESS_TOKEN"] = access_token

        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.readlines()

            # Update BROKER_ACCESS_TOKEN
            updated = False
            for i, line in enumerate(env_content):
                if line.startswith("BROKER_ACCESS_TOKEN="):
                    env_content[i] = f"BROKER_ACCESS_TOKEN={access_token}\n"
                    updated = True
                    break

            if not updated:
                env_content.append(f"BROKER_ACCESS_TOKEN={access_token}\n")

            with open(env_file, 'w') as f:
                f.writelines(env_content)

        # Update state
        state.broker_connected = True
        state.broker_status_message = "Connected successfully"

        # Broadcast update
        await manager.broadcast({
            "type": "broker_status",
            "data": {
                "connected": True,
                "status": "Connected successfully"
            }
        })

        logger.info("Upstox OAuth completed successfully")

        # Return success page
        return HTMLResponse("""
            <html>
            <head>
                <title>Upstox OAuth - Success</title>
                <style>
                    body { font-family: Arial; padding: 40px; text-align: center; background: #0f172a; color: white; }
                    .success { color: #10b981; font-size: 48px; }
                    button { padding: 12px 24px; background: #10b981; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; margin-top: 20px; }
                    button:hover { background: #059669; }
                </style>
            </head>
            <body>
                <div class="success">✓</div>
                <h2>Authorization Successful!</h2>
                <p>Upstox broker connected successfully. You can close this window.</p>
                <button onclick="window.close()">Close Window</button>
                <script>
                    // Auto-close after 3 seconds
                    setTimeout(() => { window.close(); }, 3000);
                </script>
            </body>
            </html>
        """)

    except Exception as e:
        logger.error(f"Upstox OAuth callback error: {e}")
        return HTMLResponse(f"""
            <html>
            <head><title>Upstox OAuth - Error</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center; background: #0f172a; color: white;">
                <div style="color: #ef4444; font-size: 48px;">✗</div>
                <h2 style="color: #ef4444;">Authorization Failed</h2>
                <p>Error: {str(e)}</p>
                <button onclick="window.close()" style="padding: 12px 24px; background: #ef4444; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; margin-top: 20px;">Close Window</button>
            </body>
            </html>
        """)


@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """Get recent logs"""
    return {"logs": state.logs[-limit:]}


@app.get("/api/market-data")
async def get_market_data():
    """Get current market data"""
    return {"market_data": state.market_data}


# =============================================================================
# WEBSOCKET ENDPOINTS
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_json({
            "type": "initial_state",
            "data": {
                "strategy_running": state.strategy_running,
                "broker": state.current_broker,
                "orders_count": len(state.orders),
                "positions_count": len(state.positions)
            }
        })

        # Keep connection alive and listen for messages
        while True:
            data = await websocket.receive_text()
            # Echo back or process as needed
            await websocket.send_json({
                "type": "echo",
                "data": data
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def update_market_data():
    """Background task to update market data"""
    while True:
        try:
            # Simulate market data update
            # In real implementation, fetch from broker
            state.market_data = {
                "NIFTY": {
                    "last_price": 24500.50,
                    "change": 125.30,
                    "change_percent": 0.51
                },
                "timestamp": datetime.now().isoformat()
            }

            # Broadcast to all connected clients
            await manager.broadcast({
                "type": "market_data",
                "data": state.market_data
            })

        except Exception as e:
            logger.error(f"Error updating market data: {e}")

        await asyncio.sleep(1)  # Update every second


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting Trading Algorithm Web Dashboard")
    state.add_log("Web dashboard started", "INFO")

    # Load initial configuration
    state.current_config = load_config()

    # Start background tasks
    # asyncio.create_task(update_market_data())


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Trading Algorithm Web Dashboard")
    state.add_log("Web dashboard shutting down", "INFO")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
