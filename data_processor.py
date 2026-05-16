# ============================================================
# data_processor.py — Indicator Engine & Breakout Detection
# ============================================================

import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

from config import (
    EMA_FAST, EMA_SLOW, EMA_LTF_FAST, EMA_LTF_SLOW,
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MIN_CONSOLIDATION_CANDLES, CONSOLIDATION_ATR_MULTIPLIER,
    BREAKOUT_ATR_MULTIPLIER,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Enums & Data Classes
# -------------------------------------------------------

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class BreakoutDirection(Enum):
    BULLISH = "bullish"   # Break above resistance
    BEARISH = "bearish"   # Break below support
    NONE = "none"


@dataclass
class ConsolidationZone:
    """Represents a detected consolidation range on the HTF."""
    high: float
    low: float
    candle_count: int
    atr_at_detection: float
    range_size: float = field(init=False)

    def __post_init__(self):
        self.range_size = self.high - self.low

    def __str__(self):
        return (
            f"ConsolidationZone(high={self.high:.5f}, low={self.low:.5f}, "
            f"range={self.range_size:.5f}, candles={self.candle_count})"
        )


@dataclass
class BreakoutSignal:
    """A confirmed breakout from a consolidation zone."""
    direction: BreakoutDirection
    breakout_level: float       # The level that was broken
    breakout_candle_close: float
    atr: float
    candle_timestamp: pd.Timestamp

    def __str__(self):
        return (
            f"BreakoutSignal({self.direction.value.upper()} | "
            f"level={self.breakout_level:.5f} | "
            f"close={self.breakout_candle_close:.5f})"
        )


@dataclass
class HTFAnalysis:
    """Full HTF (1hr) analysis result passed to Claude."""
    pair: str
    trend: TrendDirection
    ema_fast: float
    ema_slow: float
    atr: float
    bb_upper: float
    bb_lower: float
    bb_width: float
    consolidation: Optional[ConsolidationZone]
    breakout: Optional[BreakoutSignal]
    recent_highs: List[float]
    recent_lows: List[float]
    last_close: float
    last_open: float
    last_high: float
    last_low: float


@dataclass
class LTFAnalysis:
    """Full LTF (5min) analysis result passed to Claude for entry timing."""
    pair: str
    trend: TrendDirection
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    last_close: float
    last_open: float
    last_high: float
    last_low: float
    is_bullish_candle: bool
    is_bearish_candle: bool
    is_engulfing: bool
    is_pin_bar: bool
    retest_detected: bool
    retest_level: Optional[float]


# -------------------------------------------------------
# HTF Processor — 1 Hour Chart
# -------------------------------------------------------

class HTFProcessor:
    """
    Processes 1hr OHLCV data.
    Computes trend, volatility, consolidation zones,
    and breakout signals.
    """

    def analyze(self, df: pd.DataFrame, pair: str) -> Optional[HTFAnalysis]:
        """
        Run full HTF analysis on a DataFrame of 1hr candles.
        Returns HTFAnalysis or None if insufficient data.
        """
        if df is None or len(df) < EMA_SLOW + 5:
            logger.warning(f"[HTF] Insufficient data for {pair}: {len(df) if df is not None else 0} candles")
            return None

        df = df.copy()

        # --- Indicators ---
        df[f"ema_{EMA_FAST}"] = ta.ema(df["close"], length=EMA_FAST)
        df[f"ema_{EMA_SLOW}"] = ta.ema(df["close"], length=EMA_SLOW)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)

        # Calculate Bollinger Bands - FIXED: Use iloc positioning to avoid column name issues
        bb = ta.bbands(df["close"], length=BOLLINGER_PERIOD, std=BOLLINGER_STD)
        # Access columns by position (0=upper band, 1=middle band, 2=lower band)
        df["bb_upper"] = bb.iloc[:, 0]  # First column = Upper Bollinger Band
        df["bb_lower"] = bb.iloc[:, 2]  # Third column = Lower Bollinger Band

        df.dropna(inplace=True)
        if len(df) < MIN_CONSOLIDATION_CANDLES + 2:
            return None

        last = df.iloc[-1]
        ema_fast = last[f"ema_{EMA_FAST}"]
        ema_slow = last[f"ema_{EMA_SLOW}"]
        atr = last["atr"]

        # --- Trend ---
        trend = self._detect_trend(last, ema_fast, ema_slow)

        # --- Consolidation ---
        consolidation = self._detect_consolidation(df, atr)

        # --- Breakout ---
        breakout = self._detect_breakout(df, consolidation, atr)

        # --- Key levels ---
        recent_highs = df["high"].tail(20).nlargest(3).tolist()
        recent_lows = df["low"].tail(20).nsmallest(3).tolist()

        bb_width = last["bb_upper"] - last["bb_lower"]

        return HTFAnalysis(
            pair=pair,
            trend=trend,
            ema_fast=round(ema_fast, 5),
            ema_slow=round(ema_slow, 5),
            atr=round(atr, 5),
            bb_upper=round(last["bb_upper"], 5),
            bb_lower=round(last["bb_lower"], 5),
            bb_width=round(bb_width, 5),
            consolidation=consolidation,
            breakout=breakout,
            recent_highs=[round(h, 5) for h in recent_highs],
            recent_lows=[round(l, 5) for l in recent_lows],
            last_close=round(last["close"], 5),
            last_open=round(last["open"], 5),
            last_high=round(last["high"], 5),
            last_low=round(last["low"], 5),
        )

    def _detect_trend(self, last_row, ema_fast: float, ema_slow: float) -> TrendDirection:
        """
        Trend based on EMA alignment and price position.
        Bullish: price > EMA20 > EMA50
        Bearish: price < EMA20 < EMA50
        """
        price = last_row["close"]
        if price > ema_fast > ema_slow:
            return TrendDirection.BULLISH
        elif price < ema_fast < ema_slow:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def _detect_consolidation(
        self,
        df: pd.DataFrame,
        atr: float,
    ) -> Optional[ConsolidationZone]:
        """
        Scan the last N candles for a tight consolidation range.
        A valid consolidation:
        - Has at least MIN_CONSOLIDATION_CANDLES candles
        - Range (high - low) is less than ATR * CONSOLIDATION_ATR_MULTIPLIER
        """
        lookback = df.tail(30)

        best_zone = None
        best_count = 0

        for start_idx in range(len(lookback) - MIN_CONSOLIDATION_CANDLES):
            window = lookback.iloc[start_idx:]

            for end_idx in range(MIN_CONSOLIDATION_CANDLES, len(window) + 1):
                segment = window.iloc[:end_idx]
                zone_high = segment["high"].max()
                zone_low = segment["low"].min()
                zone_range = zone_high - zone_low

                if zone_range <= atr * CONSOLIDATION_ATR_MULTIPLIER:
                    if end_idx > best_count:
                        best_count = end_idx
                        best_zone = ConsolidationZone(
                            high=round(zone_high, 5),
                            low=round(zone_low, 5),
                            candle_count=end_idx,
                            atr_at_detection=round(atr, 5),
                        )
                else:
                    break  # Range expanded, stop extending this window

        if best_zone:
            logger.info(f"Consolidation detected: {best_zone}")

        return best_zone

    def _detect_breakout(
        self,
        df: pd.DataFrame,
        consolidation: Optional[ConsolidationZone],
        atr: float,
    ) -> Optional[BreakoutSignal]:
        """
        Check if the latest candle has broken out of the consolidation zone.
        Breakout is valid if:
        - Candle CLOSES above/below the consolidation high/low
        - Breakout candle body is >= ATR * BREAKOUT_ATR_MULTIPLIER (strong candle)
        """
        if consolidation is None:
            return None

        last = df.iloc[-1]
        close = last["close"]
        candle_body = abs(last["close"] - last["open"])
        min_body = atr * BREAKOUT_ATR_MULTIPLIER

        # Bullish breakout — close above consolidation high
        if close > consolidation.high and candle_body >= min_body:
            logger.info(f"BULLISH BREAKOUT detected above {consolidation.high}")
            return BreakoutSignal(
                direction=BreakoutDirection.BULLISH,
                breakout_level=consolidation.high,
                breakout_candle_close=round(close, 5),
                atr=round(atr, 5),
                candle_timestamp=df.index[-1],
            )

        # Bearish breakout — close below consolidation low
        if close < consolidation.low and candle_body >= min_body:
            logger.info(f"BEARISH BREAKOUT detected below {consolidation.low}")
            return BreakoutSignal(
                direction=BreakoutDirection.BEARISH,
                breakout_level=consolidation.low,
                breakout_candle_close=round(close, 5),
                atr=round(atr, 5),
                candle_timestamp=df.index[-1],
            )

        return None


# -------------------------------------------------------
# LTF Processor — 5 Minute Chart
# -------------------------------------------------------

class LTFProcessor:
    """
    Processes 5min OHLCV data.
    Detects entry triggers: retest of broken level,
    rejection candles, and momentum confirmation.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        pair: str,
        htf_analysis: Optional[HTFAnalysis] = None,
    ) -> Optional[LTFAnalysis]:
        """
        Run LTF analysis. Optionally uses HTF context
        (breakout level) to detect retest setups.
        """
        if df is None or len(df) < RSI_PERIOD + 5:
            logger.warning(f"[LTF] Insufficient data for {pair}")
            return None

        df = df.copy()

        # --- Indicators ---
        df[f"ema_{EMA_LTF_FAST}"] = ta.ema(df["close"], length=EMA_LTF_FAST)
        df[f"ema_{EMA_LTF_SLOW}"] = ta.ema(df["close"], length=EMA_LTF_SLOW)
        df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)

        df.dropna(inplace=True)
        if len(df) < 3:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        ema_fast = last[f"ema_{EMA_LTF_FAST}"]
        ema_slow = last[f"ema_{EMA_LTF_SLOW}"]
        rsi = last["rsi"]
        atr = last["atr"]

        # --- Trend ---
        trend = self._detect_trend(last, ema_fast, ema_slow)

        # --- Candle patterns ---
        is_bullish = last["close"] > last["open"]
        is_bearish = last["close"] < last["open"]
        is_engulfing = self._is_engulfing(last, prev)
        is_pin_bar = self._is_pin_bar(last, atr)

        # --- Retest detection ---
        retest_detected, retest_level = self._detect_retest(
            df, htf_analysis
        )

        return LTFAnalysis(
            pair=pair,
            trend=trend,
            ema_fast=round(ema_fast, 5),
            ema_slow=round(ema_slow, 5),
            rsi=round(rsi, 2),
            atr=round(atr, 5),
            last_close=round(last["close"], 5),
            last_open=round(last["open"], 5),
            last_high=round(last["high"], 5),
            last_low=round(last["low"], 5),
            is_bullish_candle=is_bullish,
            is_bearish_candle=is_bearish,
            is_engulfing=is_engulfing,
            is_pin_bar=is_pin_bar,
            retest_detected=retest_detected,
            retest_level=retest_level,
        )

    def _detect_trend(self, last_row, ema_fast: float, ema_slow: float) -> TrendDirection:
        price = last_row["close"]
        if price > ema_fast > ema_slow:
            return TrendDirection.BULLISH
        elif price < ema_fast < ema_slow:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def _is_engulfing(self, last: pd.Series, prev: pd.Series) -> bool:
        """
        Bullish engulfing: current bullish candle body fully engulfs previous bearish body.
        Bearish engulfing: current bearish candle body fully engulfs previous bullish body.
        """
        curr_bull = last["close"] > last["open"]
        prev_bear = prev["close"] < prev["open"]
        bullish_engulf = (
            curr_bull and prev_bear
            and last["close"] > prev["open"]
            and last["open"] < prev["close"]
        )

        curr_bear = last["close"] < last["open"]
        prev_bull = prev["close"] > prev["open"]
        bearish_engulf = (
            curr_bear and prev_bull
            and last["open"] > prev["close"]
            and last["close"] < prev["open"]
        )

        return bullish_engulf or bearish_engulf

    def _is_pin_bar(self, candle: pd.Series, atr: float) -> bool:
        """
        Pin bar: wick is at least 2x the body size,
        and the body is less than 30% of total candle range.
        """
        body = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]

        if total_range == 0:
            return False

        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        longest_wick = max(upper_wick, lower_wick)

        return (
            longest_wick >= 2 * body
            and body <= 0.3 * total_range
            and total_range >= 0.3 * atr   # Not a doji on tiny range
        )

    def _detect_retest(
        self,
        df: pd.DataFrame,
        htf_analysis: Optional[HTFAnalysis],
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if price is retesting a broken HTF level.
        A retest is detected when:
        - HTF breakout exists
        - LTF price has pulled back to within 0.5 ATR of breakout level
        - Price is showing rejection (not closing through the level)
        """
        if htf_analysis is None or htf_analysis.breakout is None:
            return False, None

        level = htf_analysis.breakout.breakout_level
        atr = df["atr"].iloc[-1]
        tolerance = atr * 0.5

        last_close = df["close"].iloc[-1]
        last_low = df["low"].iloc[-1]
        last_high = df["high"].iloc[-1]

        if htf_analysis.breakout.direction == BreakoutDirection.BULLISH:
            # Price should come back down near breakout level but not close below it
            touched_level = last_low <= level + tolerance
            held_above = last_close > level - tolerance
            if touched_level and held_above:
                logger.info(f"Retest of bullish breakout level {level:.5f} detected")
                return True, round(level, 5)

        elif htf_analysis.breakout.direction == BreakoutDirection.BEARISH:
            # Price should come back up near breakout level but not close above it
            touched_level = last_high >= level - tolerance
            held_below = last_close < level + tolerance
            if touched_level and held_below:
                logger.info(f"Retest of bearish breakout level {level:.5f} detected")
                return True, round(level, 5)

        return False, None


# -------------------------------------------------------
# Main DataProcessor — orchestrates both timeframes
# -------------------------------------------------------

class DataProcessor:
    """
    Top-level processor. Takes raw HTF and LTF DataFrames,
    returns structured analysis objects ready for Claude.
    """

    def __init__(self):
        self.htf_processor = HTFProcessor()
        self.ltf_processor = LTFProcessor()

    def process(
        self,
        pair: str,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
    ) -> Tuple[Optional[HTFAnalysis], Optional[LTFAnalysis]]:
        """
        Process both timeframes for a given pair.
        Returns (HTFAnalysis, LTFAnalysis).
        """
        htf = self.htf_processor.analyze(htf_df, pair)
        ltf = self.ltf_processor.analyze(ltf_df, pair, htf_analysis=htf)

        return htf, ltf

    def to_claude_context(
        self,
        htf: HTFAnalysis,
        ltf: LTFAnalysis,
    ) -> str:
        """
        Serialize analysis objects into a structured string
        that Claude can reason over clearly.
        """
        breakout_info = "None detected"
        if htf.breakout:
            breakout_info = (
                f"{htf.breakout.direction.value.upper()} breakout | "
                f"Level: {htf.breakout.breakout_level} | "
                f"Candle close: {htf.breakout.breakout_candle_close}"
            )

        consolidation_info = "None detected"
        if htf.consolidation:
            consolidation_info = (
                f"High: {htf.consolidation.high} | "
                f"Low: {htf.consolidation.low} | "
                f"Range: {htf.consolidation.range_size:.5f} | "
                f"Candles: {htf.consolidation.candle_count}"
            )

        retest_info = f"Yes — level: {ltf.retest_level}" if ltf.retest_detected else "No"

        return f"""
=== MARKET ANALYSIS: {htf.pair} ===

--- 1HR CHART (HTF) ---
Trend:              {htf.trend.value.upper()}
Last Close:         {htf.last_close}
EMA20:              {htf.ema_fast}
EMA50:              {htf.ema_slow}
ATR(14):            {htf.atr}
BB Upper:           {htf.bb_upper}
BB Lower:           {htf.bb_lower}
BB Width:           {htf.bb_width}
Recent Highs:       {htf.recent_highs}
Recent Lows:        {htf.recent_lows}
Consolidation Zone: {consolidation_info}
Breakout Signal:    {breakout_info}

--- 5MIN CHART (LTF) ---
Trend:              {ltf.trend.value.upper()}
Last Close:         {ltf.last_close}
EMA9:               {ltf.ema_fast}
EMA21:              {ltf.ema_slow}
RSI(14):            {ltf.rsi}
ATR(14):            {ltf.atr}
Bullish Candle:     {ltf.is_bullish_candle}
Bearish Candle:     {ltf.is_bearish_candle}
Engulfing Pattern:  {ltf.is_engulfing}
Pin Bar:            {ltf.is_pin_bar}
Retest Detected:    {retest_info}
""".strip()