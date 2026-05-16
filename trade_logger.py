# ============================================================
# trade_logger.py — SQLite Trade Journal
# ============================================================

import sqlite3
import logging
import json
from datetime import datetime, timezone
from typing import List
from contextlib import contextmanager

from config import DB_PATH
from order_manager import Order
from claude_analyst import TradeSignal

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Persists every signal, order, and daily snapshot to SQLite.
    Every record includes Claude's full reasoning — complete audit trail.

    Tables:
    - signals    : every Claude analysis including WAITs
    - orders     : every submitted order + outcome
    - daily_stats: end-of-day snapshots
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[DB] Transaction error: {e}")
            raise
        finally:
            conn.close()

    # ---------------------------------------------------
    # Schema
    # ---------------------------------------------------

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT NOT NULL,
                    pair         TEXT NOT NULL,
                    decision     TEXT NOT NULL,
                    confidence   REAL,
                    entry_price  REAL,
                    stop_loss    REAL,
                    take_profit  REAL,
                    risk_reward  REAL,
                    reasoning    TEXT,
                    confluences  TEXT,
                    warnings     TEXT
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id     TEXT UNIQUE NOT NULL,
                    pair         TEXT NOT NULL,
                    side         TEXT NOT NULL,
                    units        REAL NOT NULL,
                    entry_price  REAL NOT NULL,
                    stop_loss    REAL NOT NULL,
                    take_profit  REAL NOT NULL,
                    fill_price   REAL,
                    status       TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    closed_at    TEXT,
                    pnl          REAL,
                    risk_amount  REAL,
                    risk_pct     REAL,
                    pips_at_risk REAL,
                    risk_reward  REAL,
                    confidence   REAL,
                    reasoning    TEXT,
                    confluences  TEXT,
                    warnings     TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    date             TEXT UNIQUE NOT NULL,
                    starting_balance REAL,
                    closing_balance  REAL,
                    daily_pnl        REAL,
                    daily_pnl_pct    REAL,
                    trades_taken     INTEGER,
                    wins             INTEGER,
                    losses           INTEGER,
                    win_rate         REAL,
                    trading_halted   INTEGER
                );
            """)
        logger.info(f"[DB] Initialized at {self.db_path}")

    # ---------------------------------------------------
    # Signal Logging
    # ---------------------------------------------------

    def log_signal(self, signal: TradeSignal):
        """Log every Claude signal — WAITs included."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO signals (
                    timestamp, pair, decision, confidence,
                    entry_price, stop_loss, take_profit, risk_reward,
                    reasoning, confluences, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                signal.pair,
                signal.decision.value,
                signal.confidence,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                signal.risk_reward,
                signal.reasoning,
                json.dumps(signal.confluences),
                json.dumps(signal.warnings),
            ))

    # ---------------------------------------------------
    # Order Logging
    # ---------------------------------------------------

    def log_order(self, order: Order):
        """Log a new order on submission."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, pair, side, units,
                    entry_price, stop_loss, take_profit,
                    fill_price, status, submitted_at, closed_at, pnl,
                    risk_amount, risk_pct, pips_at_risk,
                    risk_reward, confidence, reasoning, confluences, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id, order.pair, order.side, order.units,
                order.entry_price, order.stop_loss, order.take_profit,
                order.fill_price, order.status.value,
                order.submitted_at, order.closed_at, order.pnl,
                order.position_size.risk_amount,
                order.position_size.risk_pct,
                order.position_size.pips_at_risk,
                order.signal.risk_reward,
                order.signal.confidence,
                order.signal.reasoning,
                json.dumps(order.signal.confluences),
                json.dumps(order.signal.warnings),
            ))
        logger.info(f"[DB] Order logged: {order.order_id} | {order.pair} {order.side}")

    def update_order(self, order: Order):
        """Update fill price, status, close time, and PnL."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE orders SET
                    fill_price = ?,
                    status     = ?,
                    closed_at  = ?,
                    pnl        = ?
                WHERE order_id = ?
            """, (
                order.fill_price,
                order.status.value,
                order.closed_at,
                order.pnl,
                order.order_id,
            ))
        logger.info(f"[DB] Order updated: {order.order_id} | PnL: {order.pnl}")

    # ---------------------------------------------------
    # Daily Stats
    # ---------------------------------------------------

    def log_daily_stats(self, stats: dict):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO daily_stats (
                    date, starting_balance, closing_balance,
                    daily_pnl, daily_pnl_pct, trades_taken,
                    wins, losses, win_rate, trading_halted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    closing_balance = excluded.closing_balance,
                    daily_pnl       = excluded.daily_pnl,
                    daily_pnl_pct   = excluded.daily_pnl_pct,
                    trades_taken    = excluded.trades_taken,
                    wins            = excluded.wins,
                    losses          = excluded.losses,
                    win_rate        = excluded.win_rate,
                    trading_halted  = excluded.trading_halted
            """, (
                stats.get("date"),
                stats.get("starting_balance"),
                stats.get("closing_balance"),
                stats.get("daily_pnl"),
                stats.get("daily_pnl_pct"),
                stats.get("trades_taken"),
                stats.get("wins", 0),
                stats.get("losses", 0),
                stats.get("win_rate", 0.0),
                int(stats.get("trading_halted", False)),
            ))

    # ---------------------------------------------------
    # Queries for Flask dashboard
    # ---------------------------------------------------

    def get_all_orders(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY submitted_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_signals(self, limit: int = 100) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_stats(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_stats ORDER BY date DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_orders(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM orders
                WHERE status IN ('pending', 'filled') AND closed_at IS NULL
                ORDER BY submitted_at DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_performance_summary(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                   AS total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
                    SUM(pnl)                                   AS total_pnl,
                    AVG(risk_reward)                           AS avg_rr,
                    AVG(confidence)                            AS avg_confidence
                FROM orders
                WHERE status = 'filled' AND pnl IS NOT NULL
            """).fetchone()

        total = row["total_trades"] or 0
        wins  = row["wins"] or 0

        return {
            "total_trades":   total,
            "wins":           wins,
            "losses":         row["losses"] or 0,
            "win_rate":       round(wins / total * 100, 1) if total > 0 else 0,
            "total_pnl":      round(row["total_pnl"] or 0, 2),
            "avg_rr":         round(row["avg_rr"] or 0, 2),
            "avg_confidence": round((row["avg_confidence"] or 0) * 100, 1),
        }