# ============================================================
# alpaca_stream.py — Live Crypto Data Feed via Alpaca CryptoDataStream SDK
# ============================================================

import asyncio
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Callable, Dict, List, Optional
from alpaca.data.live import CryptoDataStream
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models.bars import Bar
from alpaca.data.models.trades import Trade

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    CRYPTO_PAIRS,
    HTF,
    LTF,
    HTF_CANDLE_LIMIT,
    LTF_CANDLE_LIMIT,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Historical Bar Fetcher (Using SDK)
# -------------------------------------------------------

class HistoricalDataFetcher:
    """
    Fetches historical OHLCV bars for crypto using Alpaca's SDK
    """
    
    def __init__(self):
        # Crypto historical data client doesn't require API keys for public data
        # But we include them for paper trading compatibility
        self.client = CryptoHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )
    
    def _get_timeframe(self, minutes: int) -> TimeFrame:
        """Convert minutes to Alpaca TimeFrame object."""
        if minutes == 60:
            return TimeFrame.Hour
        elif minutes == 15:
            return TimeFrame.Minute
        elif minutes == 5:
            return TimeFrame.Minute
        elif minutes == 1:
            return TimeFrame.Minute
        elif minutes == 1440:  # Daily
            return TimeFrame.Day
        else:
            return TimeFrame.Minute
    
    def fetch_bars(
        self,
        pair: str,
        timeframe_minutes: int,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch historical bars for a crypto pair.
        
        Args:
            pair: e.g. "BTC/USD"
            timeframe_minutes: 15 or 60
            limit: number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate start time
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=timeframe_minutes * limit)
            
            # Get the base timeframe and multiplier
            base_timeframe = self._get_timeframe(timeframe_minutes)
            multiplier = timeframe_minutes if base_timeframe == TimeFrame.Minute else 1
            
            # Create request
            request = CryptoBarsRequest(
                symbol_or_symbols=pair,
                timeframe=base_timeframe,
                start=start,
                end=end,
                limit=limit,
                feed='us'  # 'us' for US crypto feed
            )
            
            # If using minute timeframe with multiplier, we need to handle differently
            if base_timeframe == TimeFrame.Minute and multiplier > 1:
                # Use raw timeframe string for custom minute bars
                request.timeframe = f"{multiplier}Min"
            
            # Get bars
            bars = self.client.get_crypto_bars(request)
            
            # Convert to DataFrame
            if bars and hasattr(bars, 'df') and not bars.df.empty:
                df = bars.df.copy()
                df.reset_index(inplace=True)
                df.rename(columns={
                    'timestamp': 'timestamp',
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }, inplace=True)
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"✅ Fetched {len(df)} bars for {pair} ({timeframe_minutes}min)")
                return df
            else:
                logger.warning(f"No bars returned for {pair}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch bars for {pair}: {e}")
            return pd.DataFrame()


# -------------------------------------------------------
# Live Bar Builder
# Builds OHLCV bars from streaming trades
# -------------------------------------------------------

class LiveBarBuilder:
    """
    Builds OHLCV bars from live streaming trades.
    Maintains separate bar builders per pair per timeframe.
    """
    
    def __init__(self, timeframe_minutes: int):
        self.timeframe_minutes = timeframe_minutes
        self.current_bars: Dict[str, dict] = {}
        self.completed_bars: Dict[str, List[dict]] = defaultdict(list)
        self.last_trade_times: Dict[str, datetime] = {}
    
    def _get_bar_timestamp(self, ts: datetime) -> datetime:
        """Round down timestamp to the nearest bar boundary."""
        if self.timeframe_minutes >= 60:
            # Hourly bars
            hours = self.timeframe_minutes // 60
            bar_start_hour = (ts.hour // hours) * hours
            return ts.replace(
                hour=bar_start_hour,
                minute=0,
                second=0,
                microsecond=0
            )
        else:
            # Minute bars
            total_minutes = ts.hour * 60 + ts.minute
            bar_start_minutes = (total_minutes // self.timeframe_minutes) * self.timeframe_minutes
            return ts.replace(
                hour=bar_start_minutes // 60,
                minute=bar_start_minutes % 60,
                second=0,
                microsecond=0,
            )
    
    def update(self, pair: str, price: float, volume: float, ts: datetime) -> Optional[dict]:
        """
        Feed a new trade. Returns a completed bar if the
        current bar period has closed, otherwise None.
        """
        bar_ts = self._get_bar_timestamp(ts)
        
        # Track last trade time for this pair
        self.last_trade_times[pair] = ts
        
        if pair not in self.current_bars:
            self.current_bars[pair] = {
                "timestamp": bar_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
            return None
        
        current = self.current_bars[pair]
        
        # New bar period started — close the current bar and open a new one
        if bar_ts > current["timestamp"]:
            completed = current.copy()
            self.completed_bars[pair].append(completed)
            
            self.current_bars[pair] = {
                "timestamp": bar_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
            return completed
        
        # Still in the same bar — update OHLC
        current["high"] = max(current["high"], price)
        current["low"] = min(current["low"], price)
        current["close"] = price
        current["volume"] += volume
        return None
    
    def get_current_bar(self, pair: str) -> Optional[dict]:
        """Get the current in-progress bar."""
        return self.current_bars.get(pair)


# -------------------------------------------------------
# Alpaca Crypto Stream (Using Official SDK)
# -------------------------------------------------------

class AlpacaCryptoStream:
    """
    Connects to Alpaca's crypto stream using the official SDK.
    Builds live OHLCV bars and triggers callbacks when:
      - A new HTF bar completes → strategy analysis
      - A new LTF bar completes → entry trigger check
    """
    
    def __init__(
        self,
        on_htf_bar: Callable,
        on_ltf_bar: Callable,
    ):
        self.on_htf_bar = on_htf_bar
        self.on_ltf_bar = on_ltf_bar
        
        self.htf_builder = LiveBarBuilder(HTF)
        self.ltf_builder = LiveBarBuilder(LTF)
        
        self.fetcher = HistoricalDataFetcher()
        
        # Store seeded historical data
        self.htf_history: Dict[str, pd.DataFrame] = {}
        self.ltf_history: Dict[str, pd.DataFrame] = {}
        
        # Track last bar timestamps to avoid duplicate triggers
        self.last_htf_trigger: Dict[str, datetime] = {}
        self.last_ltf_trigger: Dict[str, datetime] = {}
        
        self._running = False
        self._stream = None
        
    def seed_historical_data(self):
        """Pre-load historical bars for all pairs before live stream starts."""
        logger.info("Seeding historical data for all crypto pairs...")
        
        for pair in CRYPTO_PAIRS:
            logger.info(f"Fetching {pair}...")
            
            self.htf_history[pair] = self.fetcher.fetch_bars(
                pair, HTF, HTF_CANDLE_LIMIT
            )
            self.ltf_history[pair] = self.fetcher.fetch_bars(
                pair, LTF, LTF_CANDLE_LIMIT
            )
            
            # Initialize last trigger timestamps
            if not self.htf_history[pair].empty:
                self.last_htf_trigger[pair] = self.htf_history[pair].index[-1]
            if not self.ltf_history[pair].empty:
                self.last_ltf_trigger[pair] = self.ltf_history[pair].index[-1]
        
        logger.info("✅ Historical data seeding completed")
    
    def append_bar_to_history(
        self,
        pair: str,
        bar: dict,
        timeframe: str,
    ):
        """Append a newly completed live bar to the historical DataFrame."""
        new_row = pd.DataFrame(
            [{
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }],
            index=[bar["timestamp"]],
        )
        
        if timeframe == "htf":
            if pair in self.htf_history and not self.htf_history[pair].empty:
                self.htf_history[pair] = pd.concat(
                    [self.htf_history[pair], new_row]
                ).tail(HTF_CANDLE_LIMIT)
            else:
                self.htf_history[pair] = new_row
            self.last_htf_trigger[pair] = bar["timestamp"]
            
        elif timeframe == "ltf":
            if pair in self.ltf_history and not self.ltf_history[pair].empty:
                self.ltf_history[pair] = pd.concat(
                    [self.ltf_history[pair], new_row]
                ).tail(LTF_CANDLE_LIMIT)
            else:
                self.ltf_history[pair] = new_row
            self.last_ltf_trigger[pair] = bar["timestamp"]
    
    async def _trade_handler(self, trade: Trade):
        """
        Handle incoming trade data from the stream.
        This is called by the SDK for each trade.
        """
        try:
            pair = trade.symbol  # Format: "BTC/USD"
            
            if pair not in CRYPTO_PAIRS:
                return
            
            price = trade.price
            volume = trade.size
            timestamp = trade.timestamp
            
            # Update bar builders with trade data
            htf_bar = self.htf_builder.update(pair, price, volume, timestamp)
            ltf_bar = self.ltf_builder.update(pair, price, volume, timestamp)
            
            # HTF bar completed — trigger strategy analysis
            if htf_bar:
                self.append_bar_to_history(pair, htf_bar, "htf")
                logger.info(f"[HTF BAR CLOSED] {pair} @ {htf_bar['timestamp']} | Close: {htf_bar['close']:.2f}")
                
                # Get current DataFrames for this pair
                htf_df = self.htf_history.get(pair, pd.DataFrame())
                ltf_df = self.ltf_history.get(pair, pd.DataFrame())
                
                # Trigger the async callback
                await self.on_htf_bar(pair, htf_bar, htf_df, ltf_df)
            
            # LTF bar completed — check for entry trigger
            if ltf_bar:
                self.append_bar_to_history(pair, ltf_bar, "ltf")
                logger.info(f"[LTF BAR CLOSED] {pair} @ {ltf_bar['timestamp']} | Close: {ltf_bar['close']:.2f}")
                
                htf_df = self.htf_history.get(pair, pd.DataFrame())
                ltf_df = self.ltf_history.get(pair, pd.DataFrame())
                
                await self.on_ltf_bar(pair, ltf_bar, htf_df, ltf_df)
                
        except Exception as e:
            logger.error(f"Error in trade handler: {e}")
    
    async def _bar_handler(self, bar: Bar):
        """
        Handle incoming bar data (pre-built bars from Alpaca).
        This is a fallback in case we want to use Alpaca's bar stream.
        """
        try:
            pair = bar.symbol
            if pair not in CRYPTO_PAIRS:
                return
            
            # Convert Alpaca Bar to our format
            bar_dict = {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            
            # Check if this is a new HTF bar
            if pair in self.last_htf_trigger:
                if bar.timestamp > self.last_htf_trigger[pair]:
                    self.append_bar_to_history(pair, bar_dict, "htf")
                    logger.info(f"[HTF BAR FROM STREAM] {pair} @ {bar.timestamp}")
                    
                    htf_df = self.htf_history.get(pair, pd.DataFrame())
                    ltf_df = self.ltf_history.get(pair, pd.DataFrame())
                    await self.on_htf_bar(pair, bar_dict, htf_df, ltf_df)
            
            # Check if this is a new LTF bar
            if pair in self.last_ltf_trigger:
                if bar.timestamp > self.last_ltf_trigger[pair]:
                    self.append_bar_to_history(pair, bar_dict, "ltf")
                    logger.info(f"[LTF BAR FROM STREAM] {pair} @ {bar.timestamp}")
                    
                    htf_df = self.htf_history.get(pair, pd.DataFrame())
                    ltf_df = self.ltf_history.get(pair, pd.DataFrame())
                    await self.on_ltf_bar(pair, bar_dict, htf_df, ltf_df)
                    
        except Exception as e:
            logger.error(f"Error in bar handler: {e}")
    
    def start(self):
        """Seed historical data then start the live stream."""
        # First, seed historical data
        self.seed_historical_data()
        
        try:
            # Initialize the crypto stream
            self._stream = CryptoDataStream(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                # For paper trading, use the sandbox URL
                # url_override="wss://stream.data.sandbox.alpaca.markets/v1beta3/crypto/us"
            )
            
            # Subscribe to trades for all crypto pairs
            for pair in CRYPTO_PAIRS:
                self._stream.subscribe_trades(self._trade_handler, pair)
                logger.info(f"Subscribed to trades for {pair}")
            
            # Optional: Also subscribe to bars if you want pre-built bars
            # for pair in CRYPTO_PAIRS:
            #     self._stream.subscribe_bars(self._bar_handler, pair)
            
            logger.info(f"✅ Starting Alpaca crypto stream for {CRYPTO_PAIRS}")
            
            # Run the stream (this is blocking)
            # Use run() for production, run_in_background() for async contexts
            self._stream.run()
            
        except Exception as e:
            logger.error(f"Failed to start crypto stream: {e}")
            raise
    
    def stop(self):
        """Stop the stream."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                logger.info("Stream stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")


# -------------------------------------------------------
# Quick test — run directly to verify connection
# -------------------------------------------------------

if __name__ == "__main__":
    
    async def dummy_htf_handler(pair, bar, htf_df, ltf_df):
        print(f"\n[HTF TEST] {pair} bar closed at {bar['timestamp']}")
        print(f"  Open: {bar['open']:.2f} High: {bar['high']:.2f}")
        print(f"  Low: {bar['low']:.2f} Close: {bar['close']:.2f}")
        print(f"  Volume: {bar['volume']:.0f}")
        print(f"  HTF History: {len(htf_df)} candles")
        print(f"  LTF History: {len(ltf_df)} candles")
    
    async def dummy_ltf_handler(pair, bar, htf_df, ltf_df):
        print(f"[LTF TEST] {pair} bar closed at {bar['timestamp']} | Close: {bar['close']:.2f}")
    
    stream = AlpacaCryptoStream(
        on_htf_bar=dummy_htf_handler,
        on_ltf_bar=dummy_ltf_handler,
    )
    
    print("Starting Alpaca Crypto Stream with Official SDK...")
    print("Press Ctrl+C to stop")
    
    try:
        stream.start()
    except KeyboardInterrupt:
        print("\nStopping stream...")
        stream.stop()