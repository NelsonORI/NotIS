# ============================================================
# order_manager.py — Alpaca Paper Trade Execution
# ============================================================

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

import requests

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
)
from claude_analyst import TradeSignal, TradeDecision
from risk_manager import RiskManager, PositionSize

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Enums & Data Classes
# -------------------------------------------------------

class OrderStatus(Enum):
    PENDING   = "pending"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    FAILED    = "failed"


@dataclass
class Order:
    """Represents a submitted paper trade order."""
    order_id: str
    pair: str
    side: str
    units: float
    entry_price: float
    stop_loss: float
    take_profit: float
    status: OrderStatus
    submitted_at: str
    signal: TradeSignal
    position_size: PositionSize
    fill_price: Optional[float] = None
    closed_at: Optional[str] = None
    pnl: Optional[float] = None

    def __str__(self):
        return (
            f"Order | {self.pair} {self.side.upper()} | "
            f"Units: {self.units:,.0f} | Entry: {self.entry_price} | "
            f"SL: {self.stop_loss} | TP: {self.take_profit} | "
            f"Status: {self.status.value}"
        )


# -------------------------------------------------------
# Order Manager
# -------------------------------------------------------

class OrderManager:
    """
    Handles all order submission and management via Alpaca REST API.

    Flow:
    1. Receive TradeSignal from Claude
    2. Run pre-trade checks via RiskManager
    3. Calculate position size
    4. Submit bracket order (entry + SL + TP) to Alpaca paper API
    5. Register trade with RiskManager
    6. Return Order object for logging
    """

    ORDERS_ENDPOINT   = f"{ALPACA_BASE_URL}/v2/orders"
    ACCOUNT_ENDPOINT  = f"{ALPACA_BASE_URL}/v2/account"
    POSITIONS_ENDPOINT = f"{ALPACA_BASE_URL}/v2/positions"

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type": "application/json",
        }
        self.open_orders: dict = {}

    # ---------------------------------------------------
    # Account
    # ---------------------------------------------------

    def get_account_balance(self) -> float:
        try:
            resp = requests.get(self.ACCOUNT_ENDPOINT, headers=self.headers, timeout=10)
            resp.raise_for_status()
            balance = float(resp.json().get("cash", 0))
            logger.info(f"[ORDER] Account balance: ${balance:,.2f}")
            return balance
        except Exception as e:
            logger.error(f"[ORDER] Failed to fetch account balance: {e}")
            return 0.0

    def get_open_positions(self) -> list:
        try:
            resp = requests.get(self.POSITIONS_ENDPOINT, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[ORDER] Failed to fetch positions: {e}")
            return []

    # ---------------------------------------------------
    # Order Submission
    # ---------------------------------------------------

    def submit_order(self, signal: TradeSignal) -> Optional[Order]:
        """
        Main entry point. Takes a TradeSignal, runs checks,
        sizes the position, submits a bracket order to Alpaca.
        """
        if signal.decision == TradeDecision.WAIT:
            logger.info(f"[ORDER] WAIT signal for {signal.pair}. No order submitted.")
            return None

        # Pre-trade risk check
        allowed, reason = self.risk_manager.can_trade(signal.pair)
        if not allowed:
            logger.warning(f"[ORDER] Trade blocked for {signal.pair}: {reason}")
            return None

        # Fetch live account balance
        balance = self.get_account_balance()
        if balance <= 0:
            logger.error("[ORDER] Could not fetch account balance.")
            return None

        self.risk_manager.update_balance(balance)

        # Calculate position size
        pos = self.risk_manager.calculate_position_size(
            pair=signal.pair,
            account_balance=balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        if pos is None:
            logger.error(f"[ORDER] Position sizing failed for {signal.pair}.")
            return None

        side = "buy" if signal.decision == TradeDecision.LONG else "sell"

        order_payload = {
            "symbol": signal.pair,
            "qty": str(int(pos.units)),
            "side": side,
            "type": "limit",
            "time_in_force": "gtc",
            "limit_price": str(round(signal.entry_price, 5)),
            "order_class": "bracket",
            "stop_loss": {
                "stop_price": str(round(signal.stop_loss, 5)),
            },
            "take_profit": {
                "limit_price": str(round(signal.take_profit, 5)),
            },
        }

        logger.info(
            f"[ORDER] Submitting {side.upper()} | {signal.pair} | "
            f"Units: {int(pos.units):,} | Entry: {signal.entry_price} | "
            f"SL: {signal.stop_loss} | TP: {signal.take_profit}"
        )

        try:
            resp = requests.post(
                self.ORDERS_ENDPOINT,
                headers=self.headers,
                json=order_payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            order = Order(
                order_id=data["id"],
                pair=signal.pair,
                side=side,
                units=pos.units,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status=OrderStatus.PENDING,
                submitted_at=datetime.now(timezone.utc).isoformat(),
                signal=signal,
                position_size=pos,
            )

            self.open_orders[order.order_id] = order
            self.risk_manager.register_open_trade(signal.pair)

            logger.info(f"[ORDER] Order submitted successfully: {order}")
            return order

        except requests.HTTPError as e:
            logger.error(
                f"[ORDER] HTTP error: {e} | "
                f"Response: {e.response.text if e.response else 'N/A'}"
            )
            return None
        except Exception as e:
            logger.error(f"[ORDER] Unexpected error: {e}")
            return None

    # ---------------------------------------------------
    # Order Management
    # ---------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        try:
            resp = requests.delete(
                f"{self.ORDERS_ENDPOINT}/{order_id}",
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            if order_id in self.open_orders:
                self.open_orders[order_id].status = OrderStatus.CANCELLED
                pair = self.open_orders[order_id].pair
                self.risk_manager.register_closed_trade(pair)
            logger.info(f"[ORDER] Order {order_id} cancelled.")
            return True
        except Exception as e:
            logger.error(f"[ORDER] Failed to cancel order {order_id}: {e}")
            return False

    def sync_order_statuses(self):
        """Poll Alpaca for latest status of all open orders."""
        for order_id, order in list(self.open_orders.items()):
            try:
                resp = requests.get(
                    f"{self.ORDERS_ENDPOINT}/{order_id}",
                    headers=self.headers,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")

                if status == "filled":
                    order.status = OrderStatus.FILLED
                    order.fill_price = float(data.get("filled_avg_price", order.entry_price))
                    logger.info(f"[ORDER] {order.pair} FILLED @ {order.fill_price}")

                elif status in ("cancelled", "expired", "rejected"):
                    order.status = OrderStatus.CANCELLED
                    order.closed_at = datetime.now(timezone.utc).isoformat()
                    self.risk_manager.register_closed_trade(order.pair)
                    logger.info(f"[ORDER] {order.pair} order {status.upper()}.")
                    del self.open_orders[order_id]

            except Exception as e:
                logger.error(f"[ORDER] Failed to sync order {order_id}: {e}")

    def get_orders_summary(self) -> list:
        return [
            {
                "order_id": o.order_id,
                "pair": o.pair,
                "side": o.side,
                "units": o.units,
                "entry_price": o.entry_price,
                "stop_loss": o.stop_loss,
                "take_profit": o.take_profit,
                "status": o.status.value,
                "submitted_at": o.submitted_at,
                "fill_price": o.fill_price,
                "pnl": o.pnl,
                "confidence": o.signal.confidence,
                "risk_reward": o.signal.risk_reward,
                "reasoning": o.signal.reasoning,
                "confluences": o.signal.confluences,
                "warnings": o.signal.warnings,
            }
            for o in self.open_orders.values()
        ]