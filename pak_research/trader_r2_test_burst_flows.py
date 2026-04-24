"""
IMC Prosperity 4 — Trader Template
====================================
Design principles drawn from:
  - FrankfurtHedgehogs : top-level config block, ProductTrader base, auto-clamping bid/ask,
                          wall-mid computation, check_for_informed pattern
  - round5__1_         : Logger class, StatefulStrategy save/load contract,
                          Signal(IntEnum), MarketMakingStrategy with get_true_value hook,
                          DeanonymizedTradesStrategy pattern
  - round_5_all        : internal order-book simulation on buy/sell, InteractionBlocks
                          separation of strategy primitives

Datamodel reference   : Appendix B of the Prosperity 4 wiki
Allowed imports       : json, math, statistics, typing, jsonpickle, numpy, pandas
Return signature      : (result: dict, conversions: int, traderData: str)

NOTE  Round 1 — INTARIAN_PEPPER_ROOT and ASH_COATED_OSMIUM now live.
      All future-round classes (EtfStrategy, OptionStrategy, etc.) are stubs —
      the utility functions they need (Black-Scholes, ETF synthetic value) are
      already implemented so the team can drop in logic round by round.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import math
from abc import abstractmethod
from copy import deepcopy
from enum import IntEnum
from math import exp, log, sqrt
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
#  One place for all product names, position limits, and strategy parameters.
#  Update this section at the start of every round.
# ══════════════════════════════════════════════════════════════════════════════

# ── Product symbols ────────────────────────────────────────────────────────────
# Round 0 (Tutorial)
EMERALDS = "EMERALDS"
TOMATOES = "TOMATOES"

# Round 1
PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"   # trending +1000/day; wall_mid = fair value
OSMIUM      = "ASH_COATED_OSMIUM"      # stationary ~10000; EmeraldsStrategy archetype

# Round 2+ ETF products
# CROISSANTS        = "CROISSANTS"
# JAMS              = "JAMS"
# DJEMBES           = "DJEMBES"
# PICNIC_BASKET1    = "PICNIC_BASKET1"
# PICNIC_BASKET2    = "PICNIC_BASKET2"

# Round 3+ Options
# VOLCANIC_ROCK     = "VOLCANIC_ROCK"
# OPTION_STRIKES    = [9500, 9750, 10000, 10250, 10500]
# OPTION_SYMBOLS    = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in OPTION_STRIKES]

# Round 4+ Commodity
# MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

# ── Position limits ────────────────────────────────────────────────────────────
POSITION_LIMITS: Dict[str, int] = {
    EMERALDS:    50,
    TOMATOES:    50,
    PEPPER_ROOT: 80,
    OSMIUM:      80,
    # CROISSANTS: 250,
    # JAMS: 350,
    # DJEMBES: 60,
    # PICNIC_BASKET1: 60,
    # PICNIC_BASKET2: 100,
    # VOLCANIC_ROCK: 400,
    # **{s: 200 for s in OPTION_SYMBOLS},
    # MAGNIFICENT_MACARONS: 75,
}
DEFAULT_LIMIT = 80

# ── Informed trader ID (revealed in Round 5, but tracked from Round 1) ─────────
INFORMED_TRADER = "Olivia"

# ── Signal sentinel ────────────────────────────────────────────────────────────
class Signal(IntEnum):
    """Direction signal used by informed-trader and mean-reversion strategies."""
    SHORT   = -1
    NEUTRAL =  0
    LONG    =  1

# ── ETF basket compositions (fill when Round 2 starts) ────────────────────────
# ETF_COMPOSITIONS: Dict[str, Dict[str, int]] = {
#     PICNIC_BASKET1: {CROISSANTS: 6, JAMS: 3, DJEMBES: 1},
#     PICNIC_BASKET2: {CROISSANTS: 4, JAMS: 2},
# }
# ETF_INITIAL_PREMIUMS: Dict[str, float] = {PICNIC_BASKET1: 5.0, PICNIC_BASKET2: 53.0}

# ── Options config (fill when Round 3 starts) ──────────────────────────────────
# OPTION_DAY            = 3       # days until expiry this round
# OPTION_DAYS_PER_YEAR  = 365
# IV_SCALPING_THR       = 0.7
# IV_SCALPING_WINDOW    = 100


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LOGGER
#  Identical to the standard Prosperity logger used by both round5 files.
#  Compresses state + orders into the fixed log format the visualiser expects.
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """
    Structured logger compatible with the Prosperity log visualiser.
    Call logger.print() anywhere in strategy code.
    Call logger.flush() once at the end of Trader.run().
    """

    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self._to_json(
                [self._compress_state(state, ""), self._compress_orders(orders),
                 conversions, "", ""]
            )
        )
        max_item = (self.max_log_length - base_length) // 3
        print(
            self._to_json(
                [
                    self._compress_state(state, self._truncate(state.traderData, max_item)),
                    self._compress_orders(orders),
                    conversions,
                    self._truncate(trader_data, max_item),
                    self._truncate(self.logs, max_item),
                ]
            )
        )
        self.logs = ""

    # ── internal helpers ────────────────────────────────────────────────────────
    def _compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp, trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            self._compress_trades(state.own_trades),
            self._compress_trades(state.market_trades),
            state.position,
            self._compress_observations(state.observations),
        ]

    def _compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> list:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values() for t in arr
        ]

    def _compress_observations(self, obs: Observation) -> list:
        conv = {}
        for p, o in obs.conversionObservations.items():
            conv[p] = [o.bidPrice, o.askPrice, o.transportFees,
                       o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [obs.plainValueObservations, conv]

    def _compress_orders(self, orders: Dict[Symbol, List[Order]]) -> list:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def _to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def _truncate(self, value: str, max_length: int) -> str:
        """Binary-search truncation that respects JSON encoding length."""
        lo, hi, out = 0, min(len(value), max_length), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid] + ("..." if mid < len(value) else "")
            if len(json.dumps(candidate)) <= max_length:
                out, lo = candidate, mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MATH UTILITIES
#  Self-contained numerical helpers that strategies can call.
#  No external state — pure functions only.
# ══════════════════════════════════════════════════════════════════════════════

_N = NormalDist()   # reused across calls to avoid repeated instantiation


class MathUtils:
    """
    Pure-function math helpers.
    All methods are static — call as MathUtils.bs_call(...) etc.
    """

    # ── Black-Scholes ──────────────────────────────────────────────────────────
    @staticmethod
    def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        """European call price via Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * _N.cdf(d1) - K * exp(-r * T) * _N.cdf(d2)

    @staticmethod
    def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        """Delta of a European call."""
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return _N.cdf(d1)

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> Optional[float]:
        """
        Bisection implied-volatility solver.
        Returns None if the market price is below intrinsic value.
        """
        intrinsic = max(S - K * exp(-r * T), 0.0)
        if market_price <= intrinsic:
            return None
        lo, hi = 1e-6, 10.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            diff = MathUtils.bs_call(S, K, T, mid, r) - market_price
            if abs(diff) < tol:
                return mid
            lo, hi = (mid, hi) if diff < 0 else (lo, mid)
        return (lo + hi) / 2.0

    # ── ETF synthetic value ────────────────────────────────────────────────────
    @staticmethod
    def etf_synthetic(
        component_mids: Dict[str, float],
        composition: Dict[str, int],
    ) -> Optional[float]:
        """
        Compute the synthetic (fair) value of an ETF basket.

        Args:
            component_mids : {symbol: mid_price} for each constituent
            composition    : {symbol: quantity} basket recipe

        Returns:
            Synthetic basket value, or None if any constituent price is missing.

        Example (Round 2):
            MathUtils.etf_synthetic(
                {"CROISSANTS": 4500.0, "JAMS": 6000.0, "DJEMBES": 13000.0},
                {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
            )
        """
        total = 0.0
        for symbol, qty in composition.items():
            price = component_mids.get(symbol)
            if price is None:
                return None
            total += price * qty
        return total

    @staticmethod
    def etf_spread(
        basket_mid: float,
        component_mids: Dict[str, float],
        composition: Dict[str, int],
        running_premium: float = 0.0,
    ) -> Optional[float]:
        """
        basket_mid − synthetic_value − running_premium.
        Positive → basket is expensive → sell basket.
        Negative → basket is cheap    → buy basket.
        """
        synth = MathUtils.etf_synthetic(component_mids, composition)
        if synth is None:
            return None
        return basket_mid - synth - running_premium

    # ── Rolling statistics ──────────────────────────────────────────────────────
    @staticmethod
    def rolling_mean(values: List[float], window: int) -> float:
        w = values[-window:]
        return sum(w) / len(w) if w else 0.0

    @staticmethod
    def rolling_std(values: List[float], window: int) -> float:
        w = values[-window:]
        n = len(w)
        if n < 2:
            return 0.0
        m = sum(w) / n
        return math.sqrt(sum((x - m) ** 2 for x in w) / n)

    @staticmethod
    def z_score(values: List[float], window: int) -> Optional[float]:
        """Z-score of the last value relative to a rolling window."""
        if len(values) < window:
            return None
        std = MathUtils.rolling_std(values, window)
        if std == 0:
            return None
        mean = MathUtils.rolling_mean(values, window)
        return (values[-1] - mean) / std

    # ── Incremental running mean (Welford) ──────────────────────────────────────
    @staticmethod
    def update_running_mean(
        old_mean: float, old_n: int, new_value: float
    ) -> Tuple[float, int]:
        """
        Online update of a running mean.
        Returns (new_mean, new_n).
        Useful for tracking the ETF basket premium across 10,000 timesteps
        without storing the full history.
        """
        n = old_n + 1
        return old_mean + (new_value - old_mean) / n, n


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PRODUCT CONTEXT
#  Thin wrapper around one product's slice of TradingState.
#  Computes wall mid, best bid/ask, and remaining capacity in __init__
#  so every strategy gets consistent, pre-computed values for free.
# ══════════════════════════════════════════════════════════════════════════════

class ProductContext:
    """
    Snapshot of market data for a single product at one timestep.

    Attributes (read-only after init):
        buy_orders  : {price: volume}  sorted descending
        sell_orders : {price: volume}  sorted ascending  (volumes positive)
        best_bid / best_ask            : best price on each side, or None
        wall_bid / wall_ask            : highest-volume price level each side
        wall_mid                       : (wall_bid + wall_ask) / 2, or None
        position                       : current position (signed int)
        max_buy  / max_sell            : remaining capacity before limit hit
    """

    def __init__(self, symbol: str, state: TradingState) -> None:
        self.symbol   = symbol
        self.state    = state
        self.position = state.position.get(symbol, 0)
        self.limit    = POSITION_LIMITS.get(symbol, DEFAULT_LIMIT)

        od = state.order_depths.get(symbol, OrderDepth())

        # Sort and normalise volumes (sell_orders have negative volumes in datamodel)
        self.buy_orders: Dict[int, int] = dict(
            sorted(od.buy_orders.items(), reverse=True)
        )
        self.sell_orders: Dict[int, int] = {
            p: abs(v)
            for p, v in sorted(od.sell_orders.items())
        }

        self.best_bid: Optional[int] = max(self.buy_orders)  if self.buy_orders  else None
        self.best_ask: Optional[int] = min(self.sell_orders) if self.sell_orders else None

        # Wall = highest-volume level on each side
        self.wall_bid: Optional[int] = (
            max(self.buy_orders,  key=self.buy_orders.__getitem__)
            if self.buy_orders  else None
        )
        self.wall_ask: Optional[int] = (
            min(self.sell_orders, key=self.sell_orders.__getitem__)
            if self.sell_orders else None
        )
        self.wall_mid: Optional[float] = (
            (self.wall_bid + self.wall_ask) / 2
            if self.wall_bid is not None and self.wall_ask is not None
            else None
        )

        # Remaining order capacity (respects current position + limit)
        self.max_buy:  int = self.limit - self.position
        self.max_sell: int = self.limit + self.position

    def mid_price(self) -> Optional[float]:
        """Simple (best_bid + best_ask) / 2."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BASE STRATEGY
#  All strategies inherit from Strategy.
#  Stateful strategies additionally implement save() / load().
#  Order placement methods clamp volumes and simulate the internal book.
# ══════════════════════════════════════════════════════════════════════════════

class Strategy:
    """
    Base class for all product strategies.

    Subclasses must implement act(ctx, state).
    Optional: override save() / load() for stateful strategies.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._orders: List[Order]  = []
        self._conversions: int     = 0
        self._buy_spent: int       = 0   # total buy qty placed this timestep
        self._sell_spent: int      = 0   # total sell qty placed this timestep
        # Internal copy of order book, updated as orders are placed
        # (from round_5_all: avoids double-quoting at consumed levels)
        self._book: Optional[OrderDepth] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        """Called by Trader.run(). Resets state, builds context, calls act()."""
        self._orders      = []
        self._conversions = 0
        self._buy_spent   = 0
        self._sell_spent  = 0
        self._book        = deepcopy(state.order_depths.get(self.symbol, OrderDepth()))
        # Normalise sell volumes to positive in the internal book
        self._book.sell_orders = {p: abs(v) for p, v in self._book.sell_orders.items()}

        ctx = ProductContext(self.symbol, state)

        # Guard: only act if both sides have liquidity
        if ctx.best_bid is not None and ctx.best_ask is not None:
            self.act(ctx, state)

        return self._orders, self._conversions

    @abstractmethod
    def act(self, ctx: ProductContext, state: TradingState) -> None:
        """Override with trading logic. Use self.buy() / self.sell()."""
        raise NotImplementedError

    # ── State persistence (override in stateful subclasses) ────────────────────
    def save(self) -> Any:
        """Serialise persistent state to a JSON-safe object."""
        return None

    def load(self, data: Any) -> None:
        """Restore persistent state from the object returned by save()."""
        pass

    # ── Order helpers ──────────────────────────────────────────────────────────
    def buy(self, price: int, quantity: int) -> None:
        """
        Place a buy order.
        - Clamps quantity to remaining capacity.
        - Simulates consuming the internal ask book so subsequent logic
          sees the correct remaining liquidity (from round_5_all design).
        """
        ctx_limit = POSITION_LIMITS.get(self.symbol, DEFAULT_LIMIT)
        remaining_capacity = ctx_limit - self._buy_spent
        qty = max(0, min(int(quantity), remaining_capacity))
        if qty <= 0:
            return
        self._orders.append(Order(self.symbol, int(price), qty))
        self._buy_spent += qty
        # Simulate consuming the ask side of the internal book
        remaining = qty
        while remaining > 0 and self._book.sell_orders:
            best_ask = min(self._book.sell_orders)
            if best_ask > price:
                break
            available = self._book.sell_orders[best_ask]
            consumed  = min(available, remaining)
            if consumed >= available:
                del self._book.sell_orders[best_ask]
            else:
                self._book.sell_orders[best_ask] -= consumed
            remaining -= consumed

    def sell(self, price: int, quantity: int) -> None:
        """
        Place a sell order.
        - Clamps quantity to remaining capacity.
        - Simulates consuming the internal bid book.
        """
        ctx_limit = POSITION_LIMITS.get(self.symbol, DEFAULT_LIMIT)
        remaining_capacity = ctx_limit - self._sell_spent
        qty = max(0, min(int(quantity), remaining_capacity))
        if qty <= 0:
            return
        self._orders.append(Order(self.symbol, int(price), -qty))
        self._sell_spent += qty
        # Simulate consuming the bid side of the internal book
        remaining = qty
        while remaining > 0 and self._book.buy_orders:
            best_bid = max(self._book.buy_orders)
            if best_bid < price:
                break
            available = self._book.buy_orders[best_bid]
            consumed  = min(available, remaining)
            if consumed >= available:
                del self._book.buy_orders[best_bid]
            else:
                self._book.buy_orders[best_bid] -= consumed
            remaining -= consumed

    def convert(self, amount: int) -> None:
        self._conversions += amount


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — STRATEGY PRIMITIVES
#  Reusable building blocks that individual strategies compose.
#  From round_5_all's InteractionBlocks concept, but typed and self-contained.
# ══════════════════════════════════════════════════════════════════════════════

class Primitives:
    """
    Stateless strategy building blocks.
    Each method modifies the calling strategy via strategy.buy() / strategy.sell().
    """

    @staticmethod
    def take_best_orders(
        strategy: Strategy,
        ctx: ProductContext,
        fair_value: float,
        take_edge: float = 1.0,
    ) -> None:
        """
        Sweep every ask level that is at least take_edge below fair value (buy),
        and every bid level that is at least take_edge above fair value (sell).
        Uses the internal simulated book so each level is only taken once even
        if multiple calls occur within the same timestep.
        """
        # Buy every ask strictly profitable vs fair value
        for price, volume in sorted(ctx.sell_orders.items()):
            if price > fair_value - take_edge:
                break
            strategy.buy(price, volume)

        # Sell every bid strictly profitable vs fair value
        for price, volume in sorted(ctx.buy_orders.items(), reverse=True):
            if price < fair_value + take_edge:
                break
            strategy.sell(price, volume)

    @staticmethod
    def zero_ev_flush(
        strategy: Strategy,
        ctx: ProductContext,
        fair_value: float,
    ) -> None:
        """
        FIX 1 — Zero-EV inventory flush (from round_5_all).

        After the take pass we may still carry inventory from prior timesteps.
        If a bot is bidding or offering exactly at fair value, trade with them
        at zero profit to drain excess inventory and restore position capacity
        for future positive-EV fills.

        This is the primary mechanism that keeps the position lean so the
        taking pass never misses an edge because the limit is already full.

        Only fires when:
          - We are long AND a bot is bidding at fair value → sell to flatten.
          - We are short AND a bot is offering at fair value → buy to flatten.
        """
        fv_int = int(fair_value)
        pos_after = ctx.position + strategy._buy_spent - strategy._sell_spent

        # Long inventory: sell at fair value if a bot bids there
        if pos_after > 0 and fv_int in strategy._book.buy_orders:
            available = strategy._book.buy_orders[fv_int]
            qty = min(pos_after, available)
            if qty > 0:
                strategy.sell(fv_int, qty)

        # Short inventory: buy at fair value if a bot offers there
        elif pos_after < 0 and fv_int in strategy._book.sell_orders:
            available = strategy._book.sell_orders[fv_int]
            qty = min(-pos_after, available)
            if qty > 0:
                strategy.buy(fv_int, qty)

    @staticmethod
    def post_passive_quotes(
        strategy: Strategy,
        ctx: ProductContext,
        fair_value: float,
        make_edge: float = 1.0,
        bid_edge_override: Optional[float] = None,
        ask_edge_override: Optional[float] = None,
    ) -> None:
        """
        FIX 2 — Corrected passive quote placement (from round_5_all logic).

        Target: post at the best available price that is still strictly inside
        fair value — i.e. the highest bid below fair_value, stepped up by 1 tick.

        The old code derived quote prices by stepping in front of the deepest
        existing bot quote (e.g. 9992 → 9993 for Resin). This is too conservative.
        round_5_all correctly targets pos_ev_bid = int(fair_value) - 1 as the
        ceiling and steps up from existing bids below that ceiling. The result
        for a fixed-true-price product like Resin is always 9999, not 9993.

        bid_edge_override / ask_edge_override: allow per-side edge widening
        (used by TomatoesStrategy for informed-signal bias).
        """
        bid_edge = bid_edge_override if bid_edge_override is not None else make_edge
        ask_edge = ask_edge_override if ask_edge_override is not None else make_edge

        # Ceiling / floor: never cross fair value
        max_bid = int(fair_value) - int(bid_edge)   # e.g. 9999 for Resin
        min_ask = int(fair_value) + int(ask_edge)   # e.g. 10001 for Resin

        # Step up from the best existing bid that is still below max_bid
        # This ensures we are at the front of the queue at the best legal price.
        bid_price = max_bid
        for price in sorted(strategy._book.buy_orders.keys(), reverse=True):
            if price < max_bid:
                # Step in front of it (volume check: only if not a 1-lot scalp)
                candidate = price + 1 if strategy._book.buy_orders[price] > 1 else price
                bid_price = min(candidate, max_bid)
                break

        # Step down from the best existing ask that is still above min_ask
        ask_price = min_ask
        for price in sorted(strategy._book.sell_orders.keys()):
            if price > min_ask:
                candidate = price - 1 if strategy._book.sell_orders[price] > 1 else price
                ask_price = max(candidate, min_ask)
                break

        remaining_buy  = ctx.max_buy  - strategy._buy_spent
        remaining_sell = ctx.max_sell - strategy._sell_spent

        if remaining_buy  > 0: strategy.buy (bid_price, remaining_buy)
        if remaining_sell > 0: strategy.sell(ask_price, remaining_sell)

    @staticmethod
    def check_informed_signal(symbol: str, state: TradingState) -> Signal:
        """
        Scan market_trades and own_trades for the informed trader.
        Returns the most recent directional signal.
        From Frankfurt's check_for_informed(), cleaned up.
        """
        trades = (
            state.market_trades.get(symbol, []) +
            state.own_trades.get(symbol, [])
        )
        bought_ts: Optional[int] = None
        sold_ts:   Optional[int] = None

        for t in trades:
            if t.buyer  == INFORMED_TRADER: bought_ts = t.timestamp
            if t.seller == INFORMED_TRADER: sold_ts   = t.timestamp

        if bought_ts is None and sold_ts is None:
            return Signal.NEUTRAL
        if bought_ts is not None and sold_ts is None:
            return Signal.LONG
        if bought_ts is None and sold_ts is not None:
            return Signal.SHORT
        # Both seen: most recent wins
        return Signal.LONG if bought_ts > sold_ts else Signal.SHORT


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MARKET MAKING STRATEGY BASE
#  Subclasses only need to implement get_fair_value().
#  All take / make / flatten logic is inherited.
#  From round5__1_'s MarketMakingStrategy hook pattern.
# ══════════════════════════════════════════════════════════════════════════════

class MarketMakingStrategy(Strategy):
    """
    Base for all market-making strategies.
    Subclasses implement get_fair_value(ctx, state) → float.

    Every tick executes three steps in order:
      1. take_best_orders  — sweep all levels with positive edge vs fair value
      2. zero_ev_flush     — drain excess inventory at fair value (FIX 1)
      3. post_passive_quotes — quote at the optimal price inside fair value (FIX 2)

    Deflection guard (FIX 3) is implemented in subclasses that need it
    (drifting-price products like Tomatoes/Kelp) since fixed-price products
    don't need it — the fair value never moves.
    """

    def __init__(
        self,
        symbol: str,
        take_edge: float = 1.0,
        make_edge: float = 1.0,
    ) -> None:
        super().__init__(symbol)
        self.take_edge = take_edge
        self.make_edge = make_edge

    @abstractmethod
    def get_fair_value(self, ctx: ProductContext, state: TradingState) -> Optional[float]:
        raise NotImplementedError

    def act(self, ctx: ProductContext, state: TradingState) -> None:
        fv = self.get_fair_value(ctx, state)
        if fv is None:
            return

        # 1. Take favourable orders (sweep all levels with positive edge)
        Primitives.take_best_orders(self, ctx, fv, self.take_edge)

        # 2. Zero-EV flush: drain excess inventory at fair value (FIX 1)
        #    This frees position capacity so we never miss a good take next tick.
        Primitives.zero_ev_flush(self, ctx, fv)

        # 3. Post passive quotes at the correct price (FIX 2)
        Primitives.post_passive_quotes(self, ctx, fv, self.make_edge)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SIGNAL STRATEGY BASE
#  For directional strategies (informed trader, ETF arb, options).
#  Subclasses implement get_signal() → Signal.
#  From round5__1_'s SignalStrategy pattern.
# ══════════════════════════════════════════════════════════════════════════════

class SignalStrategy(Strategy):
    """
    Base for directional strategies.
    Maintains a persistent signal across timesteps.
    Subclasses implement get_signal(ctx, state) → Optional[Signal].
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.signal = Signal.NEUTRAL

    @abstractmethod
    def get_signal(self, ctx: ProductContext, state: TradingState) -> Optional[Signal]:
        raise NotImplementedError

    def act(self, ctx: ProductContext, state: TradingState) -> None:
        new_signal = self.get_signal(ctx, state)
        if new_signal is not None:
            self.signal = new_signal

        pos = ctx.position

        if self.signal == Signal.NEUTRAL:
            # Unwind to flat
            if pos > 0 and ctx.best_bid is not None:
                self.sell(ctx.best_bid, pos)
            elif pos < 0 and ctx.best_ask is not None:
                self.buy(ctx.best_ask, -pos)

        elif self.signal == Signal.LONG:
            remaining = ctx.max_buy - self._buy_spent
            if remaining > 0 and ctx.best_ask is not None:
                self.buy(ctx.best_ask, remaining)

        elif self.signal == Signal.SHORT:
            remaining = ctx.max_sell - self._sell_spent
            if remaining > 0 and ctx.best_bid is not None:
                self.sell(ctx.best_bid, remaining)

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self) -> int:
        return int(self.signal)

    def load(self, data: int) -> None:
        self.signal = Signal(data)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — CONCRETE STRATEGIES (Tutorial Round)
# ══════════════════════════════════════════════════════════════════════════════

# ── EMERALDS — Fixed true price market maker ───────────────────────────────────
class EmeraldsStrategy(MarketMakingStrategy):
    """
    EMERALDS: fixed true price at 10,000, verified via unrealised PnL experiment.

    Fair price verification (Frankfurt Hedgehogs method):
      Buying 1 lot at wall_mid → instantaneous unrealised PnL = 0.000 ± 0.000 (both days)
      Buying 1 lot at mid_price → PnL = +0.002 ± 0.717  (noisy — mid_price ≠ true price)
      Buying 1 lot at best_ask  → PnL = -7.87  ± 1.02   (taker pays ~8 pts to cross)
      Buying 1 lot at bid_wall+1 → PnL = +9.00 ± 0.000  (passive maker earns ~9 pts/lot)

    Strategy (3 steps per tick):
      1. Take every ask < 10000 and every bid > 10000.
      2. Zero-EV flush: if a bot bids/offers at exactly 10000, trade to drain inventory.
      3. Post passive quote at 9999 bid / 10001 ask (steps in front of 9992/10008 wall).
    """

    TRUE_PRICE  = 10_000
    _TAKE_EDGE  = 1      # take anything ≥1 tick from 10000
    _MAKE_EDGE  = 1      # quote at 9999/10001 → earns ~9 pts/lot gross

    def __init__(self) -> None:
        super().__init__(EMERALDS, take_edge=self._TAKE_EDGE, make_edge=self._MAKE_EDGE)

    def get_fair_value(self, ctx: ProductContext, state: TradingState) -> float:
        """
        Use TRUE_PRICE when wall_mid confirms it (std=0.0 verified across both days).
        Fallback to wall_mid if the product ever changes in a future round.
        """
        wm = ctx.wall_mid
        if wm is not None and abs(wm - self.TRUE_PRICE) < 10:
            return float(self.TRUE_PRICE)
        return wm if wm is not None else float(self.TRUE_PRICE)


# ── TOMATOES — Drifting price market maker ────────────────────────────────────
class TomatoesStrategy(MarketMakingStrategy):
    """
    TOMATOES: drifting price — wall_mid is the fair value proxy.

    TUNABLE PARAMETERS — change these based on the OFFICIAL backtester only.
    Do NOT tune from CSV simulator alone: take_best_orders fires 0 times from
    CSV data (spread too wide), so CSV PnL differences from take_edge changes
    are not meaningful. The official sim models bot reactions to our quotes.

    _TAKE_EDGE:   Set to the value that scored best on the official backtester.
                  Validated: 3 gave best official PnL despite CSV sim showing
                  it fires 0 times. Effect is through position capacity dynamics
                  in the official matching engine.

    _MAKE_EDGE:   Dead parameter for TOMATOES (bot bids at fv-6.5, our quote
                  resolves to bot_bid+1 regardless of make_edge 1–5). Leave at 1.

    _DEFLECT_THR: 0.5 is the natural binary threshold — TOMATOES moves in 0.5-tick
                  steps, so any non-zero move is exactly 0.5+. Leave at 0.5.
                  Do NOT raise above 0.5 (would miss deflections).
                  Do NOT make dynamic (risks raising above 0.5 in low-vol regime).
    """

    # ── Tune these on the official backtester ─────────────────────────────────
    _TAKE_EDGE   = 3      # ← best on official backtester; fires 0x from CSV data
    _MAKE_EDGE   = 1      # dead parameter for TOMATOES; leave at 1
    _DEFLECT_THR = 0.5    # binary threshold matching TOMATOES tick size

    def __init__(self) -> None:
        super().__init__(TOMATOES, take_edge=self._TAKE_EDGE, make_edge=self._MAKE_EDGE)
        self.signal:    Signal          = Signal.NEUTRAL
        self.prev_fair: Optional[float] = None

    def get_fair_value(self, ctx: ProductContext, state: TradingState) -> Optional[float]:
        return ctx.wall_mid

    def act(self, ctx: ProductContext, state: TradingState) -> None:
        # Informed trader signal (persists via save/load)
        new_sig = Primitives.check_informed_signal(TOMATOES, state)
        if new_sig != Signal.NEUTRAL:
            self.signal = new_sig

        fv = self.get_fair_value(ctx, state)
        if fv is None:
            return

        # ── Deflection guard: 1-tick suppression on the moved-against side ────
        # Fires on every non-zero move (threshold=0.5 = TOMATOES minimum tick).
        # Prevents posting a stale quote immediately after a directional move.
        deflect_bid = deflect_ask = False
        if self.prev_fair is not None:
            delta = fv - self.prev_fair
            if delta > self._DEFLECT_THR:
                deflect_bid = True   # price rose → don't bid into the spike
            elif delta < -self._DEFLECT_THR:
                deflect_ask = True   # price fell → don't ask into the dip
        self.prev_fair = fv

        # ── 1. Take favourable orders ──────────────────────────────────────────
        Primitives.take_best_orders(self, ctx, fv, self._TAKE_EDGE)

        # ── 2. Zero-EV flush ───────────────────────────────────────────────────
        Primitives.zero_ev_flush(self, ctx, fv)

        # ── 3. Passive quotes with deflection + informed signal bias ──────────
        bid_edge: float = self._MAKE_EDGE + (1 if self.signal == Signal.SHORT else 0)
        ask_edge: float = self._MAKE_EDGE + (1 if self.signal == Signal.LONG  else 0)

        if deflect_bid: bid_edge = 100.0
        if deflect_ask: ask_edge = 100.0

        remaining_buy  = ctx.max_buy  - self._buy_spent
        remaining_sell = ctx.max_sell - self._sell_spent

        if remaining_buy > 0 or remaining_sell > 0:
            Primitives.post_passive_quotes(
                self, ctx, fv,
                make_edge=self._MAKE_EDGE,
                bid_edge_override=bid_edge,
                ask_edge_override=ask_edge,
            )

    def save(self) -> Dict[str, Any]:
        return {"signal": int(self.signal), "prev_fair": self.prev_fair}

    def load(self, data: Any) -> None:
        if isinstance(data, dict):
            self.signal    = Signal(data.get("signal", 0))
            self.prev_fair = data.get("prev_fair")
        elif isinstance(data, int):
            self.signal = Signal(data)



# ── PEPPER_ROOT — Trend-aware buy-biased market maker (Round 2) ──────────────
class PepperRootStrategy(MarketMakingStrategy):
    """
    INTARIAN_PEPPER_ROOT: normally trends +1000/day but Round 2 hints at
    possible regime changes ("bot unusual behaviour", dynamic may change).

    === ROUND 2 RISK MANAGEMENT ===

    Round 1 baseline (R²≈1.0 across 3 days):
      fv(t) = day_start + timestamp × 0.001
      detrended std ≈ 0.83, never drifts >8 ticks from formula line.

    Risk detection — we track an exponential moving average (EMA) of the
    detrended residual (wall_mid − formula_fv). In R1 this EMA stays near 0.
    If R2 has a weaker or reversed trend, the EMA drifts negative.

    Three regimes based on EMA of detrended residual:
      NORMAL  (ema > TREND_WEAK_THR):  trend intact → take aggressively, stay long
      CAUTIOUS (ema ≤ TREND_WEAK_THR): trend weakening → only take very cheap asks,
                                        post passive bid but don't add beyond current pos
      FLATTEN (ema ≤ TREND_STOP_THR):  trend reversed → sell ALL long inventory,
                                        stop buying entirely

    Thresholds calibrated from R2 EMA distribution:
      EMA std on R2 actual data = 0.083, EMA min = -0.34 (never approaches -0.5)

      TREND_WEAK_THR = -0.5  (6σ of EMA — safe on all R1/R2 data, fires within 4s
                               if trend halves; within 2s if trend reverses)
      TREND_STOP_THR = -1.0  (12σ of EMA — fires within 5s if trend halves;
                               within 3s if trend reverses)

    Both thresholds are confirmed to NEVER fire on normal R1 or R2 data.
    Both fire quickly on any genuine drift reduction ≥ 50%.

    === ENTRY PRICE CAP ===
    _MAX_TAKE_ABOVE_FV = 7: only take asks at fv+7 or cheaper.
    R1 ask was fv+6.0 to fv+7.0 → cap=6 caught ~80% of ticks.
    R2 ask is fv+6.5 to fv+7.6 → cap=6 catches only 2% on days 0/1.
    cap=7 covers 65-80% of R2 ticks consistently. Confirmed ~79.4k/day.
    """

    _TAKE_EDGE          = 3      # legacy; actual cap is _MAX_TAKE_ABOVE_FV
    _MAKE_EDGE          = 1      # passive bid at fv-1
    _DRIFT_PER_TICK     = 0.001  # expected drift; +1000 per day confirmed R²≈1.0
    _DRIFT_PER_DAY      = 1000.0 # used to project next-day start from prior day
    _MAX_TAKE_ABOVE_FV  = 7      # R2: ask at fv+6.5–7.6 → cap=7 is correct
    _EMA_ALPHA          = 0.02   # EMA smoothing (half-life ≈ 35 ticks / 3.5s)
    _TREND_WEAK_THR     = -0.5   # 6σ of EMA on R2 data → cautious mode
    _TREND_STOP_THR     = -1.0   # 12σ of EMA on R2 data → flatten and stop

    def __init__(self) -> None:
        super().__init__(PEPPER_ROOT, take_edge=self._TAKE_EDGE, make_edge=self._MAKE_EDGE)
        self.signal:          Signal          = Signal.NEUTRAL
        self.prev_fair:       Optional[float] = None
        self._day_start:      Optional[float] = None   # locked on first tick of each day
        self._prev_day_start: Optional[float] = None   # carried from end of prior day
        self._det_ema:        float           = 0.0    # EMA of (wall_mid - formula_fv)

    def _get_day_start(self, ctx: ProductContext, state: TradingState) -> Optional[float]:
        """
        Determine the day's starting price and lock it in once.

        Two-stage approach, in priority order:

        Stage 1 — carry from previous day (crash-immune):
          _prev_day_start is saved into traderData at end of each day.
          On the first tick of a new day (timestamp resets to near 0), we
          project: day_start = prev_day_start + _DRIFT_PER_DAY.
          This is immune to market crashes — even if wall_mid opens at 11000
          when the real day_start should be 12000, we still use 12000.
          The EMA then correctly sees det = 11000 - 12000 = -1000 → FLATTEN.

        Stage 2 — estimate from wall_mid (first-day fallback):
          On day -1 (no prior day), prev_day_start is None.
          We estimate from wall_mid as before, retry if wall_mid is NaN.
          Detrended noise is ±0.83 ticks; rounding to nearest 1000 is exact.

        Day-boundary detection: Prosperity resets timestamp to 0 each day.
        If the current timestamp < 1000 (first second) and we already have
        a _day_start from a prior run (prev_day_start), it is a new day.
        """
        if self._day_start is not None:
            return self._day_start

        # Stage 1: project from prior day if available (new-day detection)
        if self._prev_day_start is not None:
            # New day: timestamp has reset; project forward by one day
            self._day_start = self._prev_day_start + self._DRIFT_PER_DAY
            return self._day_start

        # Stage 2: estimate from wall_mid (first day only)
        wm = ctx.wall_mid
        if wm is None:
            return None          # retry next tick — never lock in a wrong value
        ts = state.timestamp
        raw_start = wm - ts * self._DRIFT_PER_TICK
        self._day_start = round(raw_start / 1000) * 1000
        return self._day_start

    def get_fair_value(self, ctx: ProductContext, state: TradingState) -> Optional[float]:
        day_start = self._get_day_start(ctx, state)
        if day_start is None:
            return None          # wall_mid not yet available; act() will skip this tick
        return day_start + state.timestamp * self._DRIFT_PER_TICK

    def act(self, ctx: ProductContext, state: TradingState) -> None:
        # ── Day-boundary detection ────────────────────────────────────────────
        # Prosperity resets timestamp to 0 at the start of each new day.
        # When we see a very early timestamp AND _day_start is already set from
        # a prior day, it is a new trading day: save the old day_start as prev
        # so _get_day_start can project forward by _DRIFT_PER_DAY, then clear
        # _day_start so it re-locks on the first valid tick of the new day.
        # Also reset _det_ema so the new day starts with a clean EMA baseline.
        if state.timestamp < 1000 and self._day_start is not None:
            self._prev_day_start = self._day_start
            self._day_start      = None
            self._det_ema        = 0.0

        fv = self.get_fair_value(ctx, state)
        if fv is None:
            return

        # ── Update trend EMA ──────────────────────────────────────────────────
        wm = ctx.wall_mid
        if wm is not None:
            detrend = wm - fv
            self._det_ema = (self._EMA_ALPHA * detrend
                             + (1.0 - self._EMA_ALPHA) * self._det_ema)

        self.prev_fair = fv

        limit = POSITION_LIMITS.get(PEPPER_ROOT, 80)
        pos_now = ctx.position + self._buy_spent - self._sell_spent

        # ── Regime: FLATTEN — trend clearly reversed, close all longs ─────────
        if self._det_ema <= self._TREND_STOP_THR:
            if pos_now > 0:
                # Sell everything at best available bid
                for price, volume in sorted(ctx.buy_orders.items(), reverse=True):
                    remaining_sell = pos_now - self._sell_spent
                    if remaining_sell <= 0:
                        break
                    qty = min(volume, remaining_sell)
                    if qty > 0:
                        self.sell(price, qty)
            return  # no new buys

        # ── Regime: CAUTIOUS — only take very cheap asks, don't add aggressively
        if self._det_ema <= self._TREND_WEAK_THR:
            # Only take if ask is BELOW fv (genuine bargain below fair value).
            # With thresholds -0.5/-1.0, this window is ~8 ticks and ask is
            # typically still above fv, so no buys execute in practice.
            if pos_now < limit:
                for price, volume in sorted(ctx.sell_orders.items()):
                    if price >= fv:
                        break   # only genuine bargains below fair value
                    remaining = limit - pos_now
                    if remaining <= 0:
                        break
                    qty = min(volume, remaining)
                    if qty > 0:
                        self.buy(price, qty)
            # No passive bid in CAUTIOUS: trend may be failing, don't add
            # inventory via passive fills. Hold existing position and wait.
            return

        # ── Regime: NORMAL — trend intact, original aggressive strategy ────────
        # 1. Price-capped take: buy asks at fv+6 or cheaper
        if pos_now < limit:
            for price, volume in sorted(ctx.sell_orders.items()):
                if price > fv + self._MAX_TAKE_ABOVE_FV:
                    break
                remaining = limit - pos_now
                if remaining <= 0:
                    break
                qty = min(volume, remaining)
                if qty > 0:
                    self.buy(price, qty)

        # 2. Zero-EV flush
        fv_int = int(fv)
        pos_after = ctx.position + self._buy_spent - self._sell_spent
        if pos_after < 0 and fv_int in self._book.sell_orders:
            qty = min(-pos_after, self._book.sell_orders[fv_int])
            if qty > 0:
                self.buy(fv_int, qty)

        # 3. Passive bid
        remaining_buy = ctx.max_buy - self._buy_spent
        if remaining_buy > 0:
            max_bid = int(fv) - int(self._MAKE_EDGE)
            bid_price = max_bid
            for price in sorted(self._book.buy_orders.keys(), reverse=True):
                if price < max_bid:
                    candidate = price + 1 if self._book.buy_orders[price] > 1 else price
                    bid_price = min(candidate, max_bid)
                    break
            self.buy(bid_price, remaining_buy)

    def save(self) -> Dict[str, Any]:
        return {
            "signal":         int(self.signal),
            "prev_fair":      self.prev_fair,
            "day_start":      self._day_start,
            "prev_day_start": self._prev_day_start,
            "det_ema":        self._det_ema,
        }

    def load(self, data: Any) -> None:
        if isinstance(data, dict):
            self.signal             = Signal(data.get("signal", 0))
            self.prev_fair          = data.get("prev_fair")
            self._day_start         = data.get("day_start")
            self._prev_day_start    = data.get("prev_day_start")
            self._det_ema           = data.get("det_ema", 0.0)
        elif isinstance(data, int):
            self.signal = Signal(data)


# ── OSMIUM — Stationary MM with deterministic burst-window awareness ─────────
class OsmiumStrategy(MarketMakingStrategy):
    """
    ASH_COATED_OSMIUM: stationary around true price ~10000.

    === BURST-WINDOW ALPHA ===
    Analysis of R1 + R2 data reveals that bot takers follow a DETERMINISTIC
    schedule: they place the exact same lot count at the exact same timestamps
    every day. During these windows, price dips (SELL-dominant) or rises
    (BUY-dominant) by 2–8 ticks before reverting.

    The graph from the top R1 team shows a PnL jump at t=69K–76K (10ms units
    = 690–760s), matching the cluster of four windows at 655s/725s/735s/755s.

    Schedule (confirmed 3/3 days in R2, also present in R1):
      ts=155000ms   BUY  31 lots   → price rises,  suppress ask
      ts=235000ms  SELL  25 lots   → price drops,  suppress bid
      ts=265000ms  SELL  24 lots   → price drops,  suppress bid
      ts=335000ms  SELL  36 lots   → price drops,  suppress bid
      ts=440000ms   BUY  28 lots   → price rises,  suppress ask
      ts=510000ms  SELL  22 lots   → price drops,  suppress bid
      ts=655000ms  SELL  26 lots   → price drops,  suppress bid
      ts=725000ms  SELL  24 lots   → price drops,  suppress bid
      ts=735000ms   BUY  21 lots   → price rises,  suppress ask
      ts=755000ms  SELL  35 lots   → price drops,  suppress bid
      ts=845000ms   BUY  29 lots   → price rises,  suppress ask
      ts=905000ms  SELL  28 lots   → price drops,  suppress bid
      ts=935000ms   BUY  24 lots   → price rises,  suppress ask

    EXPLOITATION MECHANISM — asymmetric quoting (zero spread cost):
    We do NOT cross the spread (which would cost ~8 ticks each way).
    Instead, we suppress ONE side of our passive quote for a short window
    around the burst, preventing adverse fills from the taker flow:

      SELL window → suppress our passive BID for WINDOW_TICKS before/during.
        During the burst, sell-takers hit the bot's bid and drive price down.
        If our bid were live, we'd buy into falling price (bad fill).
        Suppressing it means we skip those adverse fills entirely.

      BUY window → suppress our passive ASK for WINDOW_TICKS before/during.
        During the burst, buy-takers lift the ask and drive price up.
        If our ask were live, we'd sell into rising price (bad fill).

    This earns edge by AVOIDING bad fills, not by taking directionally.
    The cost is zero — we simply don't post on one side for ~30 ticks.

    GUARD CONDITIONS (anti-overfitting):
    1. The burst must be "observed" — a minimum volume of trades must actually
       occur within the window. If the official sim uses different randomisation
       (e.g., 80% of flows as stated in the wiki), the window might not fire.
       If observed_volume < MIN_BURST_VOLUME, the suppression is not applied
       and normal MM runs. We never take directional risk.
    2. Suppression only applies while pos is within safe range — never suppress
       if already at the limit on the "wrong" side.
    3. The schedule is treated as a soft hint, not a hard rule.

    QUOTING APPROACH — post_passive_quotes with make_edge=1:
      Steps to bot_bid+1 / bot_ask-1 for queue priority, capped at 9999/10001.
    take_edge=3: OSMIUM drifts ±17 ticks from 10000, takes do fire.
    """

    TRUE_PRICE   = 10_000
    _MAKE_EDGE   = 1   # ceiling only — actual posting adapts via bot_bid+1 step
    _TAKE_MARGIN = 3   # take when ask < wall_mid - MARGIN or bid > wall_mid + MARGIN

    # Burst window schedule: (timestamp_ms, direction, expected_lots)
    # direction: +1 = BUY-dominant (suppress ask), -1 = SELL-dominant (suppress bid)
    _BURST_SCHEDULE: List[Tuple[int, int, int]] = [
        (155000,  +1, 31),
        (235000,  -1, 25),
        (265000,  -1, 24),
        (335000,  -1, 36),
        (440000,  +1, 28),
        (510000,  -1, 22),
        (655000,  -1, 26),
        (725000,  -1, 24),
        (735000,  +1, 21),
        (755000,  -1, 35),
        (845000,  +1, 29),
        (905000,  -1, 28),
        (935000,  +1, 24),
    ]

    # How long before the burst to begin suppression (ticks = 100ms each)
    _SUPPRESS_LEAD  = 20   # 2 seconds before window
    _SUPPRESS_LAG   = 30   # 3 seconds into/after window
    # Minimum observed volume to confirm the burst is happening
    # (guard: if official sim randomises away this window, don't suppress)
    _MIN_BURST_VOL  = 8    # lots; half the smallest expected burst

    def __init__(self) -> None:
        super().__init__(OSMIUM, take_edge=self._TAKE_MARGIN, make_edge=self._MAKE_EDGE)
        self._burst_obs_vol: int   = 0
        self._burst_window_ts: int = -1

    def get_fair_value(self, ctx: ProductContext, state: TradingState) -> float:
        wm = ctx.wall_mid
        if wm is not None and abs(wm - self.TRUE_PRICE) < 10:
            return float(self.TRUE_PRICE)
        return wm if wm is not None else float(self.TRUE_PRICE)

    def _burst_signal(self, state: TradingState) -> Tuple[bool, bool]:
        """
        Returns (suppress_bid, suppress_ask) for this tick.

        Checks whether we are within SUPPRESS_LEAD ticks before any scheduled
        burst window. If the expected burst volume has NOT been observed yet
        (guard condition), suppression is skipped — normal MM runs instead.

        suppress_bid = True  → don't post passive bid this tick (SELL window)
        suppress_ask = True  → don't post passive ask this tick (BUY window)
        """
        ts = state.timestamp
        for win_ts, direction, expected_lots in self._BURST_SCHEDULE:
            lead_start = win_ts - self._SUPPRESS_LEAD * 100
            lag_end    = win_ts + self._SUPPRESS_LAG  * 100

            if lead_start <= ts < lag_end:
                # We are inside this window's suppression zone.
                # Track observed volume to confirm the burst is real.
                if win_ts != self._burst_window_ts:
                    self._burst_window_ts = win_ts
                    self._burst_obs_vol   = 0

                # Count volume from own_trades (fills we received this tick)
                # as a proxy for total market activity
                for trade in state.market_trades.get(OSMIUM, []):
                    if win_ts <= trade.timestamp < lag_end:
                        self._burst_obs_vol += trade.quantity

                # Guard: suppress only if burst volume is confirming
                confirmed = self._burst_obs_vol >= self._MIN_BURST_VOL

                if confirmed or ts < win_ts:
                    # Suppress the side that would give us adverse fills
                    if direction == -1:   # SELL burst → suppress bid
                        return True, False
                    else:                  # BUY burst  → suppress ask
                        return False, True

        return False, False   # no active window → normal MM

    def act(self, ctx: ProductContext, state: TradingState) -> None:
        fv = self.get_fair_value(ctx, state)

        # ── 1. Dynamic takes anchored to wall_mid ─────────────────────────────
        # Use live wall_mid as the take anchor instead of hardcoded fv=10000.
        # This fires correctly when wm drifts from 10000 (18-32% of R2 ticks).
        # e.g. wm=10015, ask=10009: fv-based (10009>9997) = no take,
        #                           wm-based (10009<10012) = take correctly.
        wm = ctx.wall_mid
        take_anchor = wm if wm is not None else fv
        pos_now = ctx.position + self._buy_spent - self._sell_spent
        limit   = POSITION_LIMITS.get(OSMIUM, 80)

        if pos_now < limit:
            for price, volume in sorted(ctx.sell_orders.items()):
                if price >= take_anchor - self._TAKE_MARGIN:
                    break
                remaining = limit - pos_now
                if remaining <= 0:
                    break
                qty = min(volume, remaining)
                if qty > 0:
                    self.buy(price, qty)

        if pos_now > -limit:
            for price, volume in sorted(ctx.buy_orders.items(), reverse=True):
                if price <= take_anchor + self._TAKE_MARGIN:
                    break
                remaining = pos_now + limit
                if remaining <= 0:
                    break
                qty = min(volume, remaining)
                if qty > 0:
                    self.sell(price, qty)

        # ── 2. Zero-EV flush ──────────────────────────────────────────────────
        Primitives.zero_ev_flush(self, ctx, fv)

        # ── 3. Burst-aware passive quoting ────────────────────────────────────
        # make_edge=1 is the ceiling (never post bid above fv-1 or ask below fv+1).
        # The actual price is bot_bid+1 / bot_ask-1 within that ceiling,
        # so quoting already adapts to wherever the bot is sitting each tick.
        suppress_bid, suppress_ask = self._burst_signal(state)

        remaining_buy  = ctx.max_buy  - self._buy_spent
        remaining_sell = ctx.max_sell - self._sell_spent

        max_bid = int(fv) - self._MAKE_EDGE
        min_ask = int(fv) + self._MAKE_EDGE

        bid_price = max_bid
        for price in sorted(self._book.buy_orders.keys(), reverse=True):
            if price < max_bid:
                candidate = price + 1 if self._book.buy_orders[price] > 1 else price
                bid_price = min(candidate, max_bid)
                break

        ask_price = min_ask
        for price in sorted(self._book.sell_orders.keys()):
            if price > min_ask:
                candidate = price - 1 if self._book.sell_orders[price] > 1 else price
                ask_price = max(candidate, min_ask)
                break

        if not suppress_bid and remaining_buy  > 0:
            self.buy (bid_price, remaining_buy)
        if not suppress_ask and remaining_sell > 0:
            self.sell(ask_price, remaining_sell)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — FUTURE ROUND STUBS
#  Blank strategy shells ready to be filled when new products appear.
#  The utility functions they need (MathUtils.bs_call, etf_synthetic, etc.)
#  are already implemented in Section 3.
# ══════════════════════════════════════════════════════════════════════════════

# class KelpStrategy(MarketMakingStrategy):
#     """Round 1: random-walk price, wall_mid as fair value. Same as Tomatoes."""
#     def __init__(self): super().__init__(KELP, take_edge=1.0, make_edge=1.0)
#     def get_fair_value(self, ctx, state): return ctx.wall_mid


# class SquidInkStrategy(SignalStrategy):
#     """Round 1: follow Olivia's daily extrema signal."""
#     def __init__(self): super().__init__(SQUID_INK)
#     def get_signal(self, ctx, state):
#         return Primitives.check_informed_signal(SQUID_INK, state)


# class EtfStrategy(Strategy):
#     """
#     Round 2: Basket ETF mean-reversion.
#     When spread > threshold → sell basket.
#     When spread < -threshold → buy basket.
#     Uses MathUtils.etf_spread() and an online running premium.
#     """
#     def __init__(self, symbol, composition, threshold=80, initial_premium=0.0):
#         super().__init__(symbol)
#         self.composition = composition
#         self.threshold   = threshold
#         self.premium_mean, self.premium_n = initial_premium, 60_000
#     def act(self, ctx, state):
#         comp_mids = {
#             s: ProductContext(s, state).wall_mid
#             for s in self.composition
#         }
#         if any(v is None for v in comp_mids.values()): return
#         raw_spread = MathUtils.etf_spread(ctx.wall_mid, comp_mids, self.composition, 0.0)
#         self.premium_mean, self.premium_n = MathUtils.update_running_mean(
#             self.premium_mean, self.premium_n, raw_spread)
#         spread = raw_spread - self.premium_mean
#         if spread > self.threshold  and ctx.max_sell > 0: self.sell(ctx.best_bid, ctx.max_sell)
#         elif spread < -self.threshold and ctx.max_buy  > 0: self.buy(ctx.best_ask, ctx.max_buy)
#     def save(self): return [self.premium_mean, self.premium_n]
#     def load(self, data): self.premium_mean, self.premium_n = data


# class OptionStrategy(Strategy):
#     """
#     Round 3: IV smile scalping via Black-Scholes.
#     Uses MathUtils.bs_call(), .implied_vol() per timestep.
#     """
#     def __init__(self, symbol, strike, days_to_expiry, iv_threshold=0.7):
#         super().__init__(symbol)
#         self.K = strike
#         self.T = days_to_expiry / MathUtils  # fill MathUtils.DAYS_PER_YEAR
#         self.iv_threshold = iv_threshold
#         self.iv_history: List[float] = []
#     def act(self, ctx, state):
#         underlying_mid = ProductContext(VOLCANIC_ROCK, state).wall_mid
#         if underlying_mid is None or ctx.wall_mid is None: return
#         iv = MathUtils.implied_vol(ctx.wall_mid, underlying_mid, self.K, self.T)
#         if iv is None: return
#         self.iv_history.append(iv)
#         # ... IV deviation vs fitted smile parabola → trade signal
#     def save(self): return self.iv_history[-500:]   # keep last 500
#     def load(self, data): self.iv_history = data


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — TRADER (entry point)
#  Required class name and signature per the wiki.
#  Instantiates strategies once, loops per timestep, handles state persistence.
# ══════════════════════════════════════════════════════════════════════════════

class Trader:
    """
    Required entry point. Must not be renamed.

    run() is called every timestep with a fresh TradingState.
    Returns: (orders: dict, conversions: int, traderData: str)

    traderData is a JSON string persisted across calls via state.traderData.
    The Lambda environment is stateless — never rely on instance variables
    persisting between calls. Use save() / load() on strategies instead.
    """

    # Also required for Round 2 (Manual Bidding challenge)
    def bid(self) -> int:
        # Market Access Fee bid for Round 2.
        # Top 50% of bids get 25% more order-book volume.
        # Expected gain from extra flow: ~5-10k SeaShells over 3 days.
        # Bid 2,000: safely in the upper half without overpaying.
        # If accepted (fee deducted), net gain is still positive.
        return 2000

    def __init__(self) -> None:
        # ── Register strategies here. One entry per product. ──────────────────
        # Key = product symbol, value = Strategy instance.
        # Add new strategies each round without touching run().
        self._strategies: Dict[str, Strategy] = {
            EMERALDS:    EmeraldsStrategy(),
            TOMATOES:    TomatoesStrategy(),
            PEPPER_ROOT: PepperRootStrategy(),
            OSMIUM:      OsmiumStrategy(),
            # PICNIC_BASKET1: EtfStrategy(PICNIC_BASKET1, ETF_COMPOSITIONS[PICNIC_BASKET1], 80),
            # PICNIC_BASKET2: EtfStrategy(PICNIC_BASKET2, ETF_COMPOSITIONS[PICNIC_BASKET2], 50),
        }

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:

        # ── 1. Restore persisted state ─────────────────────────────────────────
        saved: Dict[str, Any] = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                pass

        for symbol, strategy in self._strategies.items():
            if symbol in saved:
                try:
                    strategy.load(saved[symbol])
                except Exception:
                    pass

        # ── 2. Run each active strategy ────────────────────────────────────────
        orders:      Dict[str, List[Order]] = {}
        conversions: int = 0

        for symbol, strategy in self._strategies.items():
            if symbol not in state.order_depths:
                orders[symbol] = []
                continue
            try:
                strat_orders, strat_conv = strategy.run(state)
                orders[symbol] = strat_orders
                conversions   += strat_conv
            except Exception as e:
                logger.print(f"ERROR [{symbol}]: {e}")
                orders[symbol] = []

        # ── 3. Persist state for next call ─────────────────────────────────────
        new_saved: Dict[str, Any] = {}
        for symbol, strategy in self._strategies.items():
            try:
                new_saved[symbol] = strategy.save()
            except Exception:
                pass

        trader_data = json.dumps(new_saved, separators=(",", ":"))

        # ── 4. Flush logger and return ─────────────────────────────────────────
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
