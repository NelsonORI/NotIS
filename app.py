# ============================================================
# app.py — Flask Orchestrator + Dashboard API
# ============================================================

import os
import logging
import threading
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    LOG_LEVEL, LOG_FILE, ENFORCE_SESSION_FILTER,
    TRADING_SESSIONS,
)
from alpaca_stream import AlpacaCryptoStream
from data_processor import DataProcessor
from groq_analyst import GroqAnalyst
from risk_manager import RiskManager
from order_manager import OrderManager
from trade_logger import TradeLogger

# ----------------------------
# Environment Configuration
# ----------------------------
PORT = int(os.environ.get("PORT", 5000))
IS_RENDER = os.environ.get("RENDER", False)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Component Initialization
# ----------------------------
processor     = DataProcessor()
analyst       = GroqAnalyst()
risk_manager  = RiskManager()
order_manager = OrderManager(risk_manager)
trade_logger  = TradeLogger()

# ----------------------------
# Session Filter
# ----------------------------

def is_session_active() -> bool:
    """Return True if current UTC time is within London or NY session."""
    if not ENFORCE_SESSION_FILTER:
        return True
    now_hour = datetime.now(timezone.utc).hour
    for session in TRADING_SESSIONS.values():
        if session["start"] <= now_hour < session["end"]:
            return True
    return False

# ----------------------------
# Bar Handlers (called by stream)
# ----------------------------

async def on_htf_bar(pair, bar, htf_df, ltf_df):
    """
    Fires on every completed HTF candle (1HR or configured timeframe).
    Runs full analysis and submits order if conditions are met.
    """
    logger.info(f"[APP] HTF bar received: {pair}")

    session_active = is_session_active()
    if not session_active:
        logger.info(f"[APP] Outside trading session. Skipping analysis for {pair}.")
        return

    # Process indicators
    htf, ltf = processor.process(pair, htf_df, ltf_df)
    if htf is None or ltf is None:
        logger.warning(f"[APP] Insufficient data for {pair}. Skipping.")
        return

    # Groq analysis
    signal = analyst.analyze(pair, htf, ltf, session_active=session_active)

    # Log every signal (including WAITs — full audit trail)
    trade_logger.log_signal(signal)

    logger.info(
        f"[APP] Signal → {pair} | {signal.decision.value.upper()} | "
        f"Confidence: {signal.confidence:.0%} | R:R: {signal.risk_reward}"
    )
    logger.info(f"[APP] Reasoning: {signal.reasoning}")

    # Submit order if actionable
    order = order_manager.submit_order(signal)
    if order:
        trade_logger.log_order(order)

    # Sync any pending order statuses
    order_manager.sync_order_statuses()

    # Update daily stats
    balance = order_manager.get_account_balance()
    risk_manager.update_balance(balance)


async def on_ltf_bar(pair, bar, htf_df, ltf_df):
    """
    Fires on every completed LTF candle (5MIN/15MIN configured timeframe).
    Syncs order statuses to catch fills quickly.
    """
    order_manager.sync_order_statuses()


# ----------------------------
# Stream Thread
# ----------------------------

def start_stream():
    """Run the Alpaca stream in a background thread."""
    stream = AlpacaCryptoStream(
        on_htf_bar=on_htf_bar,
        on_ltf_bar=on_ltf_bar,
    )
    # Initialize daily stats before stream starts
    balance = order_manager.get_account_balance()
    risk_manager.start_day(balance)
    logger.info(f"[APP] Starting stream. Balance: ${balance:,.2f}")
    
    try:
        stream.start()
    except Exception as e:
        logger.error(f"[APP] Stream error: {e}")
        raise


# ----------------------------
# Flask API Routes
# ----------------------------

@app.route("/api/performance")
def api_performance():
    return jsonify(trade_logger.get_performance_summary())


@app.route("/api/orders")
def api_orders():
    return jsonify(trade_logger.get_all_orders())


@app.route("/api/orders/open")
def api_open_orders():
    return jsonify(trade_logger.get_open_orders())


@app.route("/api/signals")
def api_signals():
    return jsonify(trade_logger.get_all_signals(limit=50))


@app.route("/api/daily")
def api_daily():
    return jsonify(trade_logger.get_daily_stats())


@app.route("/api/risk")
def api_risk():
    return jsonify(risk_manager.get_daily_summary())


@app.route("/health")
def health_check():
    """Health check endpoint for Render monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "balance": order_manager.get_account_balance(),
        "open_trades": len(order_manager.get_open_orders())
    })


# ----------------------------
# Dashboard Route
# ----------------------------

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ----------------------------
# Dashboard HTML
# ----------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Forex Trading Bot — Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0d1117;
    color: #e6edf3;
    min-height: 100vh;
  }

  header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  header h1 { font-size: 18px; font-weight: 600; color: #58a6ff; }
  .live-badge {
    background: #238636;
    color: #fff;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
  }

  .container { padding: 28px 32px; max-width: 1400px; margin: 0 auto; }

  /* Stats grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }
  .stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 18px;
  }
  .stat-card .label {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 8px;
  }
  .stat-card .value {
    font-size: 26px;
    font-weight: 700;
    color: #e6edf3;
  }
  .stat-card .value.green { color: #3fb950; }
  .stat-card .value.red   { color: #f85149; }
  .stat-card .value.blue  { color: #58a6ff; }

  /* Tables */
  .section { margin-bottom: 32px; }
  .section h2 {
    font-size: 14px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 14px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    overflow: hidden;
    font-size: 13px;
  }
  th {
    background: #1c2128;
    color: #8b949e;
    text-align: left;
    padding: 10px 14px;
    font-weight: 500;
    font-size: 12px;
    border-bottom: 1px solid #30363d;
  }
  td {
    padding: 10px 14px;
    border-bottom: 1px solid #21262d;
    vertical-align: top;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge.buy    { background: #1a3a2a; color: #3fb950; }
  .badge.sell   { background: #3a1a1a; color: #f85149; }
  .badge.wait   { background: #2a2a1a; color: #d29922; }
  .badge.long   { background: #1a3a2a; color: #3fb950; }
  .badge.short  { background: #3a1a1a; color: #f85149; }
  .badge.filled   { background: #1a3050; color: #58a6ff; }
  .badge.pending  { background: #2a2a1a; color: #d29922; }
  .badge.cancelled{ background: #2a2020; color: #8b949e; }

  .reasoning-cell {
    max-width: 340px;
    font-size: 12px;
    color: #8b949e;
    line-height: 1.5;
  }
  .confluences {
    font-size: 11px;
    color: #3fb950;
    margin-top: 4px;
  }
  .warnings {
    font-size: 11px;
    color: #d29922;
    margin-top: 2px;
  }

  .no-data {
    text-align: center;
    color: #8b949e;
    padding: 32px;
    font-size: 13px;
  }

  .refresh-note {
    font-size: 11px;
    color: #8b949e;
    text-align: right;
    margin-bottom: 20px;
  }
</style>
</head>
<body>

<header>
  <h1>⚡ Crypto AI Trading Bot</h1>
  <span class="live-badge">PAPER TRADING</span>
</header>

<div class="container">
  <p class="refresh-note" id="last-updated">Loading...</p>

  <!-- Performance Stats -->
  <div class="stats-grid" id="stats-grid">
    <div class="stat-card"><div class="label">Total Trades</div><div class="value blue" id="total-trades">—</div></div>
    <div class="stat-card"><div class="label">Win Rate</div><div class="value" id="win-rate">—</div></div>
    <div class="stat-card"><div class="label">Total P&L</div><div class="value" id="total-pnl">—</div></div>
    <div class="stat-card"><div class="label">Avg R:R</div><div class="value blue" id="avg-rr">—</div></div>
    <div class="stat-card"><div class="label">Avg Confidence</div><div class="value blue" id="avg-conf">—</div></div>
    <div class="stat-card"><div class="label">Open Positions</div><div class="value" id="open-pos">—</div></div>
    <div class="stat-card"><div class="label">Daily P&L</div><div class="value" id="daily-pnl">—</div></div>
    <div class="stat-card"><div class="label">Status</div><div class="value" id="bot-status">—</div></div>
  </div>

  <!-- Open Orders -->
  <div class="section">
    <h2>Open Positions</h2>
    <table>
      <thead>
        <tr>
          <th>Pair</th><th>Side</th><th>Units</th>
          <th>Entry</th><th>SL</th><th>TP</th>
          <th>R:R</th><th>Status</th><th>Submitted</th>
        </tr>
      </thead>
      <tbody id="open-orders-body">
        <tr><td colspan="9" class="no-data">No open positions</td</tr>
      </tbody>
    </table>
  </div>

  <!-- Trade History -->
  <div class="section">
    <h2>Trade History</h2>
    <table>
      <thead>
        <tr>
          <th>Pair</th><th>Side</th><th>Entry</th><th>Fill</th>
          <th>SL</th><th>TP</th><th>P&L</th><th>R:R</th>
          <th>Confidence</th><th>Status</th><th>Reasoning</th>
        </tr>
      </thead>
      <tbody id="orders-body">
        <tr><td colspan="11" class="no-data">No trades yet</td</tr>
      </tbody>
    </table>
  </div>

  <!-- Recent AI Signals -->
  <div class="section">
    <h2>Recent AI Signals (incl. WAITs)</h2>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Pair</th><th>Decision</th>
          <th>Confidence</th><th>R:R</th><th>Reasoning</th>
        </tr>
      </thead>
      <tbody id="signals-body">
        <tr><td colspan="6" class="no-data">No signals yet</td</tr>
      </tbody>
    </table>
  </div>

</div>

<script>
  function fmt(val, decimals=2) {
    if (val === null || val === undefined) return '—';
    return parseFloat(val).toFixed(decimals);
  }

  function fmtTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString();
  }

  function badge(cls, text) {
    return `<span class="badge ${cls}">${text.toUpperCase()}</span>`;
  }

  function pnlClass(val) {
    if (!val) return '';
    return parseFloat(val) >= 0 ? 'green' : 'red';
  }

  async function fetchPerformance() {
    const res = await fetch('/api/performance');
    const d = await res.json();
    document.getElementById('total-trades').textContent = d.total_trades ?? '0';
    const wr = d.win_rate ?? 0;
    const wrEl = document.getElementById('win-rate');
    wrEl.textContent = wr + '%';
    wrEl.className = 'value ' + (wr >= 50 ? 'green' : 'red');
    const pnl = d.total_pnl ?? 0;
    const pnlEl = document.getElementById('total-pnl');
    pnlEl.textContent = '$' + fmt(pnl);
    pnlEl.className = 'value ' + (pnl >= 0 ? 'green' : 'red');
    document.getElementById('avg-rr').textContent = fmt(d.avg_rr) + 'R';
    document.getElementById('avg-conf').textContent = (d.avg_confidence ?? 0) + '%';
  }

  async function fetchRisk() {
    const res = await fetch('/api/risk');
    const d = await res.json();
    document.getElementById('open-pos').textContent = d.open_trades ?? '0';
    const dpnl = d.daily_pnl ?? 0;
    const dpEl = document.getElementById('daily-pnl');
    dpEl.textContent = '$' + fmt(dpnl);
    dpEl.className = 'value ' + (dpnl >= 0 ? 'green' : 'red');
    const halted = d.trading_halted;
    const statusEl = document.getElementById('bot-status');
    statusEl.textContent = halted ? 'HALTED' : 'ACTIVE';
    statusEl.className = 'value ' + (halted ? 'red' : 'green');
  }

  async function fetchOpenOrders() {
    const res = await fetch('/api/orders/open');
    const orders = await res.json();
    const tbody = document.getElementById('open-orders-body');
    if (!orders.length) {
      tbody.innerHTML = '<tr><td colspan="9" class="no-data">No open positions</td></tr>';
      return;
    }
    tbody.innerHTML = orders.map(o => `
      <tr>
        <td><strong>${o.pair}</strong></td>
        <td>${badge(o.side, o.side)}</td>
        <td>${parseInt(o.units).toLocaleString()}</td>
        <td>${fmt(o.entry_price, 5)}</td>
        <td>${fmt(o.stop_loss, 5)}</td>
        <td>${fmt(o.take_profit, 5)}</td>
        <td>${fmt(o.risk_reward)}R</td>
        <td>${badge(o.status, o.status)}</td>
        <td>${fmtTime(o.submitted_at)}</td>
      40
    `).join('');
  }

  async function fetchOrders() {
    const res = await fetch('/api/orders');
    const orders = await res.json();
    const tbody = document.getElementById('orders-body');
    if (!orders.length) {
      tbody.innerHTML = '<tr><td colspan="11" class="no-data">No trades yet</td></tr>';
      return;
    }
    tbody.innerHTML = orders.map(o => `
      <tr>
        <td><strong>${o.pair}</strong></td>
        <td>${badge(o.side, o.side)}</td>
        <td>${fmt(o.entry_price, 5)}</td>
        <td>${o.fill_price ? fmt(o.fill_price, 5) : '—'}</td>
        <td>${fmt(o.stop_loss, 5)}</td>
        <td>${fmt(o.take_profit, 5)}</td>
        <td class="${pnlClass(o.pnl)}">${o.pnl !== null ? '$' + fmt(o.pnl) : '—'}</td>
        <td>${fmt(o.risk_reward)}R</td>
        <td>${fmt(o.confidence * 100, 0)}%</td>
        <td>${badge(o.status, o.status)}</td>
        <td class="reasoning-cell">
          ${o.reasoning || '—'}
          ${o.confluences ? '<div class="confluences">✓ ' + JSON.parse(o.confluences || '[]').join(' · ') + '</div>' : ''}
          ${o.warnings ? '<div class="warnings">⚠ ' + JSON.parse(o.warnings || '[]').join(' · ') + '</div>' : ''}
        </td>
      60
    `).join('');
  }

  async function fetchSignals() {
    const res = await fetch('/api/signals');
    const signals = await res.json();
    const tbody = document.getElementById('signals-body');
    if (!signals.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="no-data">No signals yet</td></tr>';
      return;
    }
    tbody.innerHTML = signals.map(s => `
      <tr>
        <td style="white-space:nowrap;font-size:12px">${fmtTime(s.timestamp)}</td>
        <td><strong>${s.pair}</strong></td>
        <td>${badge(s.decision, s.decision)}</td>
        <td>${fmt(s.confidence * 100, 0)}%</td>
        <td>${s.risk_reward ? fmt(s.risk_reward) + 'R' : '—'}</td>
        <td class="reasoning-cell">${s.reasoning || '—'}</td>
      </tr>
    `).join('');
  }

  async function refresh() {
    await Promise.all([
      fetchPerformance(),
      fetchRisk(),
      fetchOpenOrders(),
      fetchOrders(),
      fetchSignals(),
    ]);
    document.getElementById('last-updated').textContent =
      'Last updated: ' + new Date().toLocaleTimeString();
  }

  refresh();
  setInterval(refresh, 15000); // Auto-refresh every 15 seconds
</script>
</body>
</html>
"""

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    # Start the Alpaca stream in a background thread
    stream_thread = threading.Thread(target=start_stream, daemon=True)
    stream_thread.start()

    # Get port from environment (for Render compatibility)
    port = int(os.environ.get("PORT", FLASK_PORT))
    
    if IS_RENDER:
        # Production mode on Render
        logger.info(f"[APP] Running in PRODUCTION mode on Render")
        logger.info(f"[APP] Dashboard running at http://0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        # Local development mode
        logger.info(f"[APP] Running in DEVELOPMENT mode")
        logger.info(f"[APP] Dashboard running at http://{FLASK_HOST}:{FLASK_PORT}")
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False)