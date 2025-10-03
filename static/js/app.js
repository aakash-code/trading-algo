// ============================================================================
// Trading Algorithm Dashboard - JavaScript
// ============================================================================

// Global state
let ws = null;
let currentConfig = {};
let isStrategyRunning = false;

// ============================================================================
// WebSocket Connection
// ============================================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = function() {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
        addLog('WebSocket connected', 'success');
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onclose = function() {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        addLog('WebSocket disconnected, attempting to reconnect...', 'warning');

        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        addLog('WebSocket error occurred', 'error');
    };
}

function handleWebSocketMessage(data) {
    console.log('WebSocket message:', data);

    switch(data.type) {
        case 'initial_state':
            updateInitialState(data.data);
            break;
        case 'strategy_status':
            updateStrategyStatus(data.data.running);
            break;
        case 'market_data':
            updateMarketData(data.data);
            break;
        case 'order_update':
            addOrderToTable(data.data);
            break;
        case 'log':
            addLog(data.data.message, data.data.level);
            break;
        case 'config_updated':
            currentConfig = data.data;
            break;
        case 'broker_changed':
            updateBrokerSelector(data.data.broker);
            if (data.data.connected !== undefined) {
                updateBrokerStatus(data.data.connected, data.data.status);
            }
            break;
        case 'broker_status':
            updateBrokerStatus(data.data.connected, data.data.status);
            break;
    }
}

function updateConnectionStatus(connected) {
    const statusBadge = document.getElementById('ws-connection-status');
    if (connected) {
        statusBadge.className = 'flex items-center gap-2 px-4 py-2 bg-green-500/10 border border-green-500 rounded-lg';
        statusBadge.innerHTML = '<i class="fas fa-wifi text-green-500 animate-pulse-ring"></i><span class="text-green-500 font-semibold text-sm">Dashboard Online</span>';
    } else {
        statusBadge.className = 'flex items-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500 rounded-lg';
        statusBadge.innerHTML = '<i class="fas fa-wifi text-red-500"></i><span class="text-red-500 font-semibold text-sm">Dashboard Offline</span>';
    }
}

function updateBrokerStatus(connected, message) {
    const statusBadge = document.getElementById('broker-status');
    if (connected) {
        statusBadge.className = 'flex items-center gap-2 px-4 py-2 bg-green-500/10 border border-green-500 rounded-lg';
        statusBadge.innerHTML = `<i class="fas fa-check-circle text-green-500"></i><span class="text-green-500 font-semibold text-sm">Broker: ${message || 'Connected'}</span>`;
    } else {
        statusBadge.className = 'flex items-center gap-2 px-4 py-2 bg-yellow-500/10 border border-yellow-500 rounded-lg';
        statusBadge.innerHTML = `<i class="fas fa-exclamation-circle text-yellow-500"></i><span class="text-yellow-500 font-semibold text-sm">Broker: ${message || 'Not Connected'}</span>`;
    }
}

// ============================================================================
// API Functions
// ============================================================================

async function apiRequest(endpoint, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(endpoint, options);
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('API request failed:', error);
        addLog(`API request failed: ${error.message}`, 'error');
        throw error;
    }
}

// ============================================================================
// Strategy Controls
// ============================================================================

async function startStrategy() {
    try {
        const result = await apiRequest('/api/strategy/start', 'POST');
        if (result.status === 'success') {
            addLog('Strategy started successfully', 'success');
            updateStrategyStatus(true);
        } else {
            addLog(result.message, 'error');
        }
    } catch (error) {
        addLog('Failed to start strategy', 'error');
    }
}

async function stopStrategy() {
    if (!confirm('Are you sure you want to stop the strategy?')) {
        return;
    }

    try {
        const result = await apiRequest('/api/strategy/stop', 'POST');
        if (result.status === 'success') {
            addLog('Strategy stopped successfully', 'success');
            updateStrategyStatus(false);
        } else {
            addLog(result.message, 'error');
        }
    } catch (error) {
        addLog('Failed to stop strategy', 'error');
    }
}

function updateStrategyStatus(running) {
    isStrategyRunning = running;
    const statusIndicator = document.getElementById('strategy-status');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');

    if (running) {
        statusIndicator.innerHTML = '<span class="w-3 h-3 rounded-full bg-green-500 animate-pulse-ring"></span><span class="text-lg font-semibold">Running</span>';
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } else {
        statusIndicator.innerHTML = '<span class="w-3 h-3 rounded-full bg-red-500 animate-pulse-ring"></span><span class="text-lg font-semibold">Stopped</span>';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// ============================================================================
// Configuration Management
// ============================================================================

async function showConfigModal() {
    // Load current config
    const result = await apiRequest('/api/config');
    currentConfig = result.config;

    // Populate form
    document.getElementById('symbol-initials').value = currentConfig.symbol_initials || '';
    document.getElementById('pe-gap').value = currentConfig.pe_gap || '';
    document.getElementById('ce-gap').value = currentConfig.ce_gap || '';
    document.getElementById('pe-quantity').value = currentConfig.pe_quantity || '';
    document.getElementById('ce-quantity').value = currentConfig.ce_quantity || '';
    document.getElementById('min-price').value = currentConfig.min_price_to_sell || '';
    document.getElementById('pe-symbol-gap').value = currentConfig.pe_symbol_gap || '';
    document.getElementById('ce-symbol-gap').value = currentConfig.ce_symbol_gap || '';

    // Show modal
    const modal = document.getElementById('config-modal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeConfigModal() {
    const modal = document.getElementById('config-modal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

async function saveConfig() {
    const config = {
        symbol_initials: document.getElementById('symbol-initials').value,
        pe_gap: parseFloat(document.getElementById('pe-gap').value),
        ce_gap: parseFloat(document.getElementById('ce-gap').value),
        pe_quantity: parseInt(document.getElementById('pe-quantity').value),
        ce_quantity: parseInt(document.getElementById('ce-quantity').value),
        min_price_to_sell: parseFloat(document.getElementById('min-price').value),
        pe_symbol_gap: parseInt(document.getElementById('pe-symbol-gap').value),
        ce_symbol_gap: parseInt(document.getElementById('ce-symbol-gap').value),
        // Add other fields from current config
        ...currentConfig
    };

    try {
        const result = await apiRequest('/api/config', 'POST', config);
        if (result.status === 'success') {
            addLog('Configuration saved successfully', 'success');
            closeConfigModal();
        } else {
            addLog('Failed to save configuration', 'error');
        }
    } catch (error) {
        addLog('Failed to save configuration', 'error');
    }
}

// ============================================================================
// Broker Management
// ============================================================================

async function switchBroker() {
    const brokerSelect = document.getElementById('broker-select');
    const selectedBroker = brokerSelect.value;

    try {
        const result = await apiRequest('/api/broker/switch', 'POST', {
            broker_name: selectedBroker
        });

        if (result.status === 'success') {
            addLog(`Switched to ${selectedBroker} broker`, 'success');
        } else {
            addLog(result.message, 'error');
        }
    } catch (error) {
        addLog('Failed to switch broker', 'error');
    }
}

function updateBrokerSelector(broker) {
    document.getElementById('broker-select').value = broker;
}

// ============================================================================
// Broker Configuration Modal
// ============================================================================

function showBrokerConfigModal() {
    const modal = document.getElementById('broker-config-modal');
    const currentBroker = document.getElementById('broker-select').value;

    // Set the broker selector to current broker
    document.getElementById('broker-config-select').value = currentBroker;

    // Show the appropriate form
    switchBrokerForm();

    // Show modal
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeBrokerConfigModal() {
    const modal = document.getElementById('broker-config-modal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

function switchBrokerForm() {
    const selectedBroker = document.getElementById('broker-config-select').value;

    // Hide all broker forms
    document.querySelectorAll('.broker-form').forEach(form => {
        form.classList.add('hidden');
    });

    // Show selected broker form
    const targetForm = document.getElementById(`${selectedBroker}-form`);
    if (targetForm) {
        targetForm.classList.remove('hidden');
    }
}

function toggleZerodhaTotp() {
    const totpEnabled = document.getElementById('zerodha-totp-enable').checked;
    const totpFields = document.getElementById('zerodha-totp-fields');

    if (totpEnabled) {
        totpFields.classList.remove('hidden');
    } else {
        totpFields.classList.add('hidden');
    }
}

async function saveBrokerCredentials() {
    const selectedBroker = document.getElementById('broker-config-select').value;
    let credentials = {
        broker_name: selectedBroker
    };

    // Collect credentials based on selected broker
    try {
        switch(selectedBroker) {
            case 'dhan':
                credentials.client_id = document.getElementById('dhan-client-id').value.trim();
                credentials.access_token = document.getElementById('dhan-access-token').value.trim();

                if (!credentials.client_id || !credentials.access_token) {
                    showNotification('error', 'Missing Fields', 'Please fill in all required fields for Dhan');
                    return;
                }
                break;

            case 'zerodha':
                credentials.api_key = document.getElementById('zerodha-api-key').value.trim();
                credentials.api_secret = document.getElementById('zerodha-api-secret').value.trim();
                credentials.client_id = document.getElementById('zerodha-client-id').value.trim();
                credentials.password = document.getElementById('zerodha-password').value.trim();
                credentials.totp_enable = document.getElementById('zerodha-totp-enable').checked;

                if (credentials.totp_enable) {
                    credentials.totp_key = document.getElementById('zerodha-totp-key').value.trim();
                }

                if (!credentials.api_key || !credentials.api_secret || !credentials.client_id || !credentials.password) {
                    showNotification('error', 'Missing Fields', 'Please fill in all required fields for Zerodha');
                    return;
                }
                break;

            case 'fyers':
                credentials.api_key = document.getElementById('fyers-api-key').value.trim();
                credentials.api_secret = document.getElementById('fyers-api-secret').value.trim();
                credentials.client_id = document.getElementById('fyers-client-id').value.trim();
                credentials.totp_key = document.getElementById('fyers-totp-key').value.trim();
                credentials.totp_pin = document.getElementById('fyers-totp-pin').value.trim();
                credentials.redirect_uri = document.getElementById('fyers-redirect-uri').value.trim();

                if (!credentials.api_key || !credentials.api_secret || !credentials.client_id ||
                    !credentials.totp_key || !credentials.totp_pin || !credentials.redirect_uri) {
                    showNotification('error', 'Missing Fields', 'Please fill in all required fields for Fyers');
                    return;
                }
                break;

            case 'upstox':
                credentials.api_key = document.getElementById('upstox-api-key').value.trim();
                credentials.api_secret = document.getElementById('upstox-api-secret').value.trim();
                credentials.redirect_uri = document.getElementById('upstox-redirect-uri').value.trim();
                credentials.access_token = document.getElementById('upstox-access-token').value.trim();

                if (!credentials.api_key || !credentials.api_secret) {
                    showNotification('error', 'Missing Fields', 'Please fill in API Key and API Secret for Upstox');
                    return;
                }
                break;
        }

        // Show loading state
        addLog(`Saving ${selectedBroker} credentials and testing connection...`, 'info');

        // Send credentials to backend
        const result = await apiRequest('/api/broker/configure', 'POST', credentials);

        if (result.status === 'success') {
            showNotification('success', 'Broker Connected!',
                `Successfully connected to ${selectedBroker.charAt(0).toUpperCase() + selectedBroker.slice(1)} broker`);
            addLog(`${selectedBroker} broker configured successfully`, 'success');

            // Close modal after success
            setTimeout(() => {
                closeBrokerConfigModal();
            }, 1500);

            // Update broker status
            updateBrokerStatus(true, 'Connected');

            // Update broker selector in header
            document.getElementById('broker-select').value = selectedBroker;
        } else if (result.status === 'oauth_required') {
            // Handle Upstox OAuth flow
            showNotification('warning', 'Authorization Required',
                'Opening Upstox authorization page. Please authorize the application.');
            addLog('Opening Upstox OAuth authorization window', 'info');

            // Open OAuth URL in popup
            const width = 600;
            const height = 700;
            const left = (screen.width / 2) - (width / 2);
            const top = (screen.height / 2) - (height / 2);

            const popup = window.open(
                result.oauth_url,
                'Upstox Authorization',
                `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
            );

            // Poll for popup close or listen for success
            const checkPopup = setInterval(() => {
                if (popup.closed) {
                    clearInterval(checkPopup);
                    addLog('OAuth window closed. Checking connection status...', 'info');

                    // Check broker status after popup closes
                    setTimeout(async () => {
                        const statusResult = await apiRequest('/api/status');
                        if (statusResult.broker_connected) {
                            showNotification('success', 'Broker Connected!',
                                'Successfully connected to Upstox broker');
                            updateBrokerStatus(true, 'Connected');
                            closeBrokerConfigModal();
                        } else {
                            showNotification('warning', 'Authorization Incomplete',
                                'Please complete the authorization process and try again.');
                        }
                    }, 1000);
                }
            }, 500);

            // Don't close modal - wait for OAuth completion
        } else {
            showNotification('error', 'Connection Failed', result.message || 'Failed to connect to broker');
            addLog(`Failed to configure ${selectedBroker}: ${result.message}`, 'error');
        }

    } catch (error) {
        showNotification('error', 'Error', 'Failed to save broker credentials. Please try again.');
        addLog(`Error configuring broker: ${error.message}`, 'error');
    }
}

function showNotification(type, title, message) {
    const popup = document.getElementById('notification-popup');
    const icon = document.getElementById('popup-icon');
    const titleEl = document.getElementById('popup-title');
    const messageEl = document.getElementById('popup-message');
    const container = popup.querySelector('div');

    // Set content
    titleEl.textContent = title;
    messageEl.textContent = message;

    // Set icon and colors based on type
    if (type === 'success') {
        icon.className = 'fas fa-check-circle text-green-500 text-2xl mt-1';
        container.className = 'bg-slate-900 border-2 border-green-500 rounded-xl p-4 shadow-2xl min-w-[300px] max-w-md';
    } else if (type === 'error') {
        icon.className = 'fas fa-times-circle text-red-500 text-2xl mt-1';
        container.className = 'bg-slate-900 border-2 border-red-500 rounded-xl p-4 shadow-2xl min-w-[300px] max-w-md';
    } else if (type === 'warning') {
        icon.className = 'fas fa-exclamation-triangle text-yellow-500 text-2xl mt-1';
        container.className = 'bg-slate-900 border-2 border-yellow-500 rounded-xl p-4 shadow-2xl min-w-[300px] max-w-md';
    }

    // Show popup
    popup.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        closeNotification();
    }, 5000);
}

function closeNotification() {
    const popup = document.getElementById('notification-popup');
    popup.classList.add('hidden');
}

// ============================================================================
// Data Display
// ============================================================================

async function refreshOrders() {
    try {
        const result = await apiRequest('/api/orders');
        displayOrders(result.orders);
        addLog('Orders refreshed', 'info');
    } catch (error) {
        addLog('Failed to refresh orders', 'error');
    }
}

function displayOrders(orders) {
    const tbody = document.getElementById('orders-tbody');

    if (!orders || orders.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="px-6 py-12 text-center text-slate-500">No orders yet</td></tr>';
        return;
    }

    tbody.innerHTML = orders.map(order => `
        <tr class="hover:bg-slate-800/50">
            <td class="px-6 py-4">${formatTime(order.timestamp)}</td>
            <td class="px-6 py-4">${order.symbol}</td>
            <td class="px-6 py-4"><span class="px-2 py-1 rounded ${order.transaction_type === 'BUY' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}">${order.transaction_type}</span></td>
            <td class="px-6 py-4">${order.quantity}</td>
            <td class="px-6 py-4">₹${order.price || 'Market'}</td>
            <td class="px-6 py-4"><span class="px-2 py-1 bg-green-500/20 text-green-500 rounded text-sm">Placed</span></td>
        </tr>
    `).join('');

    // Update count
    document.getElementById('orders-count').textContent = orders.length;
}

function addOrderToTable(order) {
    const tbody = document.getElementById('orders-tbody');

    // Remove empty state if present
    if (tbody.querySelector('td[colspan="6"]')) {
        tbody.innerHTML = '';
    }

    const row = document.createElement('tr');
    row.className = 'hover:bg-slate-800/50';
    row.innerHTML = `
        <td class="px-6 py-4">${formatTime(order.timestamp)}</td>
        <td class="px-6 py-4">${order.symbol}</td>
        <td class="px-6 py-4"><span class="px-2 py-1 rounded ${order.transaction_type === 'BUY' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}">${order.transaction_type}</span></td>
        <td class="px-6 py-4">${order.quantity}</td>
        <td class="px-6 py-4">₹${order.price || 'Market'}</td>
        <td class="px-6 py-4"><span class="px-2 py-1 bg-green-500/20 text-green-500 rounded text-sm">Placed</span></td>
    `;

    tbody.insertBefore(row, tbody.firstChild);

    // Update count
    const currentCount = parseInt(document.getElementById('orders-count').textContent);
    document.getElementById('orders-count').textContent = currentCount + 1;
}

async function refreshPositions() {
    try {
        const result = await apiRequest('/api/positions');
        displayPositions(result.positions);
        addLog('Positions refreshed', 'info');
    } catch (error) {
        addLog('Failed to refresh positions', 'error');
    }
}

function displayPositions(positions) {
    const tbody = document.getElementById('positions-tbody');

    if (!positions || positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="px-6 py-12 text-center text-slate-500">No open positions</td></tr>';
        return;
    }

    tbody.innerHTML = positions.map(position => `
        <tr class="hover:bg-slate-800/50">
            <td class="px-6 py-4">${position.symbol}</td>
            <td class="px-6 py-4">${position.quantity}</td>
            <td class="px-6 py-4">₹${position.avg_price}</td>
            <td class="px-6 py-4">₹${position.ltp}</td>
            <td class="px-6 py-4 ${position.pnl >= 0 ? 'text-green-500' : 'text-red-500'} font-semibold">
                ₹${position.pnl.toFixed(2)}
            </td>
            <td class="px-6 py-4">
                <button class="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm" onclick="closePosition('${position.symbol}')">
                    Close
                </button>
            </td>
        </tr>
    `).join('');

    // Update count
    document.getElementById('positions-count').textContent = positions.length;
}

function updateMarketData(data) {
    if (data.NIFTY) {
        document.getElementById('nifty-price').textContent = data.NIFTY.last_price.toFixed(2);

        const changePercent = data.NIFTY.change_percent;
        const changeElement = document.getElementById('nifty-change');
        changeElement.textContent = `${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
        changeElement.className = `text-sm font-semibold mt-2 ${changePercent >= 0 ? 'text-green-500' : 'text-red-500'}`;
    }
}

// ============================================================================
// Logs
// ============================================================================

function addLog(message, level = 'info') {
    const logsContainer = document.getElementById('logs-container');
    const logEntry = document.createElement('div');

    const levelColors = {
        'info': 'border-blue-500',
        'success': 'border-green-500',
        'warning': 'border-yellow-500',
        'error': 'border-red-500'
    };

    const levelTextColors = {
        'info': 'text-blue-500',
        'success': 'text-green-500',
        'warning': 'text-yellow-500',
        'error': 'text-red-500'
    };

    logEntry.className = `flex gap-3 p-3 bg-slate-800/50 border-l-4 ${levelColors[level.toLowerCase()] || levelColors['info']} rounded`;

    const timestamp = new Date().toLocaleTimeString();
    logEntry.innerHTML = `
        <span class="text-slate-400 font-semibold">[${timestamp}]</span>
        <span class="${levelTextColors[level.toLowerCase()] || levelTextColors['info']} font-bold min-w-[60px]">${level.toUpperCase()}</span>
        <span class="flex-1">${message}</span>
    `;

    logsContainer.insertBefore(logEntry, logsContainer.firstChild);

    // Keep only last 50 logs
    while (logsContainer.children.length > 50) {
        logsContainer.removeChild(logsContainer.lastChild);
    }
}

function clearLogs() {
    if (confirm('Clear all logs?')) {
        document.getElementById('logs-container').innerHTML = '';
        addLog('Logs cleared', 'info');
    }
}

// ============================================================================
// Tabs
// ============================================================================

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
    });

    // Deactivate all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'text-blue-500');
        btn.classList.add('text-slate-400', 'border-transparent');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.remove('hidden');

    // Activate corresponding button
    event.target.closest('.tab-btn').classList.remove('text-slate-400', 'border-transparent');
    event.target.closest('.tab-btn').classList.add('text-blue-500', 'border-blue-500');
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatTime(timestamp) {
    if (!timestamp) return '--';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

function updateInitialState(state) {
    isStrategyRunning = state.strategy_running;
    updateStrategyStatus(state.strategy_running);
    updateBrokerSelector(state.broker);

    // Update broker connection status
    if (state.broker_connected !== undefined) {
        updateBrokerStatus(state.broker_connected, state.broker_status || 'Not configured');
    }

    document.getElementById('orders-count').textContent = state.orders_count || 0;
    document.getElementById('positions-count').textContent = state.positions_count || 0;
}

// Check broker status on load
async function checkBrokerStatus() {
    try {
        const result = await apiRequest('/api/status');
        updateBrokerStatus(result.broker_connected, result.broker_status);
    } catch (error) {
        console.error('Failed to check broker status:', error);
    }
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Connect WebSocket
    connectWebSocket();

    // Load initial data
    refreshOrders();
    refreshPositions();
    checkBrokerStatus();  // Check broker status on load

    // Set up periodic refresh
    setInterval(refreshOrders, 30000);  // Refresh orders every 30 seconds
    setInterval(refreshPositions, 10000);  // Refresh positions every 10 seconds
    setInterval(checkBrokerStatus, 15000);  // Check broker status every 15 seconds

    addLog('Dashboard initialized', 'info');
});

// Close modal when clicking outside
window.onclick = function(event) {
    const configModal = document.getElementById('config-modal');
    const brokerModal = document.getElementById('broker-config-modal');

    if (event.target === configModal) {
        closeConfigModal();
    }

    if (event.target === brokerModal) {
        closeBrokerConfigModal();
    }
};
