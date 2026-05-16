# ============================================================
# groq_analyst.py — AI Reasoning Layer with Groq (The Brain)
# ============================================================

import json
import logging
from openai import OpenAI
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from config import GROQ_API_KEY, GROQ_MODEL, MIN_RISK_REWARD
from data_processor import HTFAnalysis, LTFAnalysis, DataProcessor

logger = logging.getLogger(__name__)

# Initialize Groq client (OpenAI-compatible)
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# -------------------------------------------------------
# Enums & Data Classes (unchanged)
# -------------------------------------------------------

class TradeDecision(Enum):
    LONG  = "long"
    SHORT = "short"
    WAIT  = "wait"


@dataclass
class TradeSignal:
    """
    A fully reasoned trade signal produced by Groq LLM.
    Contains everything needed by the order manager.
    """
    pair: str
    decision: TradeDecision
    confidence: float           # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasoning: str              # LLM's full explanation
    confluences: list           # List of confirming factors
    warnings: list              # List of risk factors flagged

    def __str__(self):
        return (
            f"TradeSignal | {self.pair} | {self.decision.value.upper()} | "
            f"Confidence: {self.confidence:.0%} | R:R {self.risk_reward:.1f} | "
            f"Entry: {self.entry_price} SL: {self.stop_loss} TP: {self.take_profit}"
        )


# -------------------------------------------------------
# Prompt Builder (Enhanced for Groq)
# -------------------------------------------------------

SYSTEM_PROMPT = """You are an expert institutional crypto trader and analyst specializing in breakout strategies for crypto markets.

Your role is to analyze market data and make precise, disciplined trade decisions.

STRATEGY RULES YOU MUST FOLLOW:
1. Only trade CONFIRMED breakouts — price must have closed beyond the consolidation zone on the 1HR chart
2. Enter on LTF (5min) retest of the broken level — not on the initial breakout candle
3. HTF trend and LTF entry must be ALIGNED — no counter-trend trades
4. Minimum R:R ratio of 2.0 — if the setup doesn't offer this, output WAIT
5. Only trade during London or New York sessions — avoid Asian session low liquidity
6. If confluence factors are weak (fewer than 2), output WAIT regardless

DECISION OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{
  "decision": "long" | "short" | "wait",
  "confidence": 0.0 to 1.0,
  "entry_price": float or null,
  "stop_loss": float or null,
  "take_profit": float or null,
  "risk_reward": float or null,
  "reasoning": "Your concise but complete reasoning here",
  "confluences": ["list", "of", "confirming", "factors"],
  "warnings": ["list", "of", "risk", "factors"]
}

ENTRY RULES:
- LONG: Enter just above the retest candle high, SL just below the retest candle low, TP at next major resistance
- SHORT: Enter just below the retest candle low, SL just above the retest candle high, TP at next major support
- If no retest detected yet, output WAIT with reasoning

BE DISCIPLINED. A WAIT is a valid and often correct decision. Protect capital above all."""


def build_user_prompt(context: str, session_active: bool) -> str:
    return f"""Analyze this crypto setup and provide your trade decision.

{context}

Current Session Active: {session_active}

Based on the above data, provide your JSON trade decision. Remember:
- Only enter if there is a confirmed HTF breakout AND a valid LTF retest
- Minimum 2 confluence factors required
- Minimum 2.0 R:R ratio required
- Output WAIT if any critical condition is not met"""


# -------------------------------------------------------
# Groq Analyst (Formerly ClaudeAnalyst)
# -------------------------------------------------------

class GroqAnalyst:
    """
    Sends structured market analysis to Groq LLM and
    parses the response into a TradeSignal.
    """

    def __init__(self):
        self.processor = DataProcessor()

    def analyze(
        self,
        pair: str,
        htf: HTFAnalysis,
        ltf: LTFAnalysis,
        session_active: bool = True,
    ) -> TradeSignal:
        """
        Core method. Takes processed analysis objects,
        calls Groq, returns a parsed TradeSignal.
        """
        # Build the structured context string
        context = self.processor.to_claude_context(htf, ltf)  # Note: method name still works

        logger.info(f"[GROQ] Sending analysis request for {pair}")
        logger.debug(f"Context:\n{context}")

        try:
            # Groq API call (OpenAI-compatible format)
            response = client.chat.completions.create(
                model=GROQ_MODEL,  # From config: "llama-3.3-70b-versatile"
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(context, session_active)
                    }
                ],
                temperature=0.3,  # Lower temp = more consistent trading decisions
                max_tokens=1000,
                response_format={"type": "json_object"}  # Forces JSON output (if model supports)
            )

            raw = response.choices[0].message.content.strip()
            logger.info(f"[GROQ] Raw response: {raw}")

            signal = self._parse_response(pair, raw)
            logger.info(f"[GROQ] Signal: {signal}")
            return signal

        except Exception as e:
            logger.error(f"[GROQ] API error: {e}")
            return self._wait_signal(pair, f"API error: {str(e)}")

    def _parse_response(self, pair: str, raw: str) -> TradeSignal:
        """Parse Groq's JSON response into a TradeSignal."""
        try:
            # Strip any accidental markdown fences
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)

            decision_str = data.get("decision", "wait").lower()
            decision = TradeDecision(decision_str)

            entry   = data.get("entry_price")
            sl      = data.get("stop_loss")
            tp      = data.get("take_profit")
            rr      = data.get("risk_reward") or self._calc_rr(entry, sl, tp)

            # Enforce minimum R:R — downgrade to WAIT if not met
            if decision != TradeDecision.WAIT and rr and rr < MIN_RISK_REWARD:
                logger.warning(
                    f"[GROQ] R:R {rr:.2f} below minimum {MIN_RISK_REWARD}. "
                    f"Overriding to WAIT."
                )
                decision = TradeDecision.WAIT
                data["warnings"] = data.get("warnings", []) + [
                    f"R:R {rr:.2f} below minimum threshold of {MIN_RISK_REWARD}"
                ]

            return TradeSignal(
                pair=pair,
                decision=decision,
                confidence=float(data.get("confidence", 0.0)),
                entry_price=float(entry) if entry else 0.0,
                stop_loss=float(sl) if sl else 0.0,
                take_profit=float(tp) if tp else 0.0,
                risk_reward=float(rr) if rr else 0.0,
                reasoning=data.get("reasoning", "No reasoning provided."),
                confluences=data.get("confluences", []),
                warnings=data.get("warnings", []),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"[GROQ] Failed to parse response: {e}\nRaw: {raw}")
            return self._wait_signal(pair, f"Parse error: {str(e)}")

    def _calc_rr(
        self,
        entry: Optional[float],
        sl: Optional[float],
        tp: Optional[float],
    ) -> Optional[float]:
        """Calculate R:R ratio from entry, SL, TP."""
        try:
            risk   = abs(entry - sl)
            reward = abs(tp - entry)
            if risk == 0:
                return None
            return round(reward / risk, 2)
        except Exception:
            return None

    def _wait_signal(self, pair: str, reason: str) -> TradeSignal:
        """Return a WAIT signal with a reason."""
        return TradeSignal(
            pair=pair,
            decision=TradeDecision.WAIT,
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            risk_reward=0.0,
            reasoning=reason,
            confluences=[],
            warnings=[reason],
        )