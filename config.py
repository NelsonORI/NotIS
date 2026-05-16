# ============================================================
# config.py — Central configuration for Crypto Trading Bot
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Alpaca API Credentials
# ----------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "your_alpaca_api_key")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "your_alpaca_secret_key")

# Paper trading base URLs
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "wss://stream.data.alpaca.markets/v1beta1/crypto"  # Changed from forex to crypto

# ----------------------------
# Groq API Configuration
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ----------------------------
# Trading Parameters (Updated for Crypto)
# ----------------------------

# Crypto pairs to monitor (Alpaca crypto format)
CRYPTO_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD"]  # Changed from FOREX_PAIRS

# Timeframes (in minutes) - Crypto often uses shorter timeframes
HTF = 60       # 1 Hour — structure & breakout detection
LTF = 15       # 15 Min  — entry trigger (crypto moves faster)

# Number of candles to fetch for analysis
HTF_CANDLE_LIMIT = 100
LTF_CANDLE_LIMIT = 100  # Increased for crypto

# ----------------------------
# Indicator Settings (Same logic, works for crypto)
# ----------------------------

# Trend
EMA_FAST = 20
EMA_SLOW = 50
EMA_LTF_FAST = 9
EMA_LTF_SLOW = 21

# Volatility / Range (Crypto needs wider ATR)
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Momentum
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Crypto-specific adjustments
CRYPTO_VOLATILITY_MULTIPLIER = 1.5  # Crypto is more volatile

# ----------------------------
# Breakout Detection Settings
# ----------------------------

MIN_CONSOLIDATION_CANDLES = 5
CONSOLIDATION_ATR_MULTIPLIER = 0.6
BREAKOUT_ATR_MULTIPLIER = 0.8

# ----------------------------
# Risk Management (Crypto-specific)
# ----------------------------

ACCOUNT_RISK_PER_TRADE = 0.01    # 1% account risk per trade
MAX_OPEN_TRADES = 3
MIN_RISK_REWARD = 2.0
MAX_DAILY_DRAWDOWN = 0.05         # 5% for crypto (higher than forex)

# ----------------------------
# Session Filter (24/7 crypto trading - optional)
# ----------------------------
# Crypto trades 24/7, but you might want to avoid low-liquidity hours
TRADING_SESSIONS = {
    "asia": {"start": 0, "end": 8},
    "london": {"start": 8, "end": 16},
    "new_york": {"start": 13, "end": 22},
    "high_volatility": {"start": 13, "end": 16},  # NY-London overlap
}

ENFORCE_SESSION_FILTER = False  # Crypto is 24/7, so disable by default

# ----------------------------
# Database
# ----------------------------
DB_PATH = "trades.db"

# ----------------------------
# Flask
# ----------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# ----------------------------
# Logging
# ----------------------------
LOG_LEVEL = "INFO"
LOG_FILE = "trading_bot.log"