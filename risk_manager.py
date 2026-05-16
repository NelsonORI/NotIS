# ============================================================
# risk_manager.py — Position Sizing & Drawdown Control
# ============================================================

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config import (
    ACCOUNT_RISK_PER_TRADE,
    MAX_OPEN_TRADES,
    MAX_DAILY_DRAWDOWN,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Data Classes
# -------------------------------------------------------

@dataclass
class PositionSize:
    """Result of a position sizing calculation."""
    pair: str
    units: float
    risk_amount: float
    account_balance: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    pip_value: float
    pips_at_risk: float

    def __str__(self):
        return (
            f"PositionSize | {self.pair} | Units: {self.units:,.0f} | "
            f"Risk: ${self.risk_amount:.2f} ({self.risk_pct:.1%}) | "
            f"Pips at risk: {self.pips_at_risk:.1f}"
        )


@dataclass
class DailyStats:
    """Tracks daily P&L and trade count for drawdown control."""
    date: str
    starting_balance: float
    current_balance: float
    trades_taken: int
    open_trades: int
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    trading_halted: bool = False

    def update(self, current_balance: float):
        self.current_balance = current_balance
        self.daily_pnl = current_balance - self.starting_balance
        self.daily_pnl_pct = self.daily_pnl / self.starting_balance
        if self.daily_pnl_pct <= -MAX_DAILY_DRAWDOWN:
            self.trading_halted = True
            logger.warning(
                f"[RISK] Daily drawdown limit hit: {self.daily_pnl_pct:.2%}. "
                f"Trading halted for the day."
            )


# -------------------------------------------------------
# Risk Manager
# -------------------------------------------------------

class RiskManager:
    """
    Controls position sizing and enforces all risk rules.

    Rules enforced:
    - Max 1% account risk per trade
    - Max 3 concurrent open positions
    - Max 3% daily drawdown — halt trading if hit
    - No trade if already in the same pair
    """

    PIP_SIZE = {
        "EUR/USD": 0.0001,
        "GBP/USD": 0.0001,
        "USD/JPY": 0.01,
        "GBP/JPY": 0.01,
        "EUR/JPY": 0.01,
        "AUD/USD": 0.0001,
        "USD/CAD": 0.0001,
        "USD/CHF": 0.0001,
        "NZD/USD": 0.0001,
    }

    def __init__(self):
        self.daily_stats: Optional[DailyStats] = None
        self.open_trade_pairs: set = set()

    # ---------------------------------------------------
    # Daily Stats Management
    # ---------------------------------------------------

    def start_day(self, account_balance: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.daily_stats = DailyStats(
            date=today,
            starting_balance=account_balance,
            current_balance=account_balance,
            trades_taken=0,
            open_trades=len(self.open_trade_pairs),
        )
        logger.info(
            f"[RISK] New trading day: {today} | "
            f"Starting balance: ${account_balance:,.2f}"
        )

    def update_balance(self, current_balance: float):
        if self.daily_stats:
            self.daily_stats.update(current_balance)

    def register_open_trade(self, pair: str):
        self.open_trade_pairs.add(pair)
        if self.daily_stats:
            self.daily_stats.open_trades = len(self.open_trade_pairs)
            self.daily_stats.trades_taken += 1

    def register_closed_trade(self, pair: str):
        self.open_trade_pairs.discard(pair)
        if self.daily_stats:
            self.daily_stats.open_trades = len(self.open_trade_pairs)

    # ---------------------------------------------------
    # Pre-Trade Checks
    # ---------------------------------------------------

    def can_trade(self, pair: str) -> tuple:
        if self.daily_stats is None:
            return False, "Daily stats not initialized. Call start_day() first."

        if self.daily_stats.trading_halted:
            return False, (
                f"Trading halted — daily drawdown limit of "
                f"{MAX_DAILY_DRAWDOWN:.0%} reached."
            )

        if len(self.open_trade_pairs) >= MAX_OPEN_TRADES:
            return False, (
                f"Max open trades reached ({MAX_OPEN_TRADES}). "
                f"Wait for a position to close."
            )

        if pair in self.open_trade_pairs:
            return False, f"Already have an open position on {pair}."

        return True, "OK"

    # ---------------------------------------------------
    # Position Sizing
    # ---------------------------------------------------

    def calculate_position_size(
        self,
        pair: str,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[PositionSize]:
        """
        Fixed fractional risk model.
        Risk 1% of account per trade, size units accordingly.
        """
        pip_size = self.PIP_SIZE.get(pair)
        if pip_size is None:
            logger.error(f"[RISK] Unknown pair: {pair}")
            return None

        if entry_price <= 0 or stop_loss <= 0:
            logger.error(f"[RISK] Invalid prices: entry={entry_price} sl={stop_loss}")
            return None

        risk_amount  = account_balance * ACCOUNT_RISK_PER_TRADE
        distance     = abs(entry_price - stop_loss)
        pips_at_risk = distance / pip_size

        if pips_at_risk == 0:
            logger.error("[RISK] Stop loss equals entry price.")
            return None

        # Pip value per unit
        if "JPY" in pair:
            pip_value_per_unit = pip_size / entry_price
        else:
            pip_value_per_unit = pip_size

        units = risk_amount / (pips_at_risk * pip_value_per_unit)
        units = max(1000, int(units / 1000) * 1000)  # Round to micro lots

        actual_risk     = pips_at_risk * pip_value_per_unit * units
        actual_risk_pct = actual_risk / account_balance

        pos = PositionSize(
            pair=pair,
            units=units,
            risk_amount=round(actual_risk, 2),
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=round(actual_risk_pct, 4),
            pip_value=round(pip_value_per_unit * units, 4),
            pips_at_risk=round(pips_at_risk, 1),
        )

        logger.info(f"[RISK] {pos}")
        return pos

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------

    def get_daily_summary(self) -> dict:
        if not self.daily_stats:
            return {}
        return {
            "date": self.daily_stats.date,
            "starting_balance": self.daily_stats.starting_balance,
            "current_balance": self.daily_stats.current_balance,
            "daily_pnl": round(self.daily_stats.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_stats.daily_pnl_pct * 100, 2),
            "trades_taken": self.daily_stats.trades_taken,
            "open_trades": self.daily_stats.open_trades,
            "trading_halted": self.daily_stats.trading_halted,
            "open_pairs": list(self.open_trade_pairs),
        }