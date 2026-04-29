"""
IMC Prosperity 5 — Trader (initial drop)
==========================================

Round 5 ships 50 products in 10 groups of 5, position limit 10/product, single
currency XIRECS, fully anonymized trade tape (no informed-trader attribution).

Empirical findings driving v1 (see ying_research/Round_5/round5_group_analysis.ipynb):
  - PEBBLES is the only group with a HARD basket invariant: sum of all 5 mids
    is glued to 50000.0 (full-sample std=2.80; 88.94% of ticks are exactly
    50000; per-day std identical across days 2/3/4 → no regime drift).
  - Engle-Granger ADF on the basket residual: stat=-122.7, p≈0  (overwhelming
    rejection of unit-root → clean stationarity).
  - For each pebble, the "synthetic fair value" = 50000 - sum(other 4 mids)
    has 1σ ≈ 2.80, p99 ≈ 17.5, max-abs ≈ 18.5 over 30,000 ticks.

Strategy v1 — PebblesBasketStrategy (5-leg basket arbitrage):
  - For each pebble, compute synth_fair_i = 50000 - sum(other 4 mids).
  - Run a market-maker around synth_fair_i on each leg (take_edge=1, make_edge=1).
  - Regime gate: if |basket_residual| > 30 for ≥ 5 ticks → degrade (only quote,
    don't take).  If |basket_residual| > 60 for ≥ 10 ticks → freeze (cancel
    quoting, unwind to flat).  Both thresholds are far above any historical
    excursion (max=18.5).

The other 9 groups are placed as TODO stubs at the bottom of Section 9 and
are NOT registered in Trader.__init__.  v2 will fill them based on the deeper
group dives in the notebook.

Code structure (mirrors ying_research/Round_4/trader_r4_init.py):
  Section 1  CONFIGURATION    — Round 5 product symbols, position limits
  Section 2  LOGGER           — Prosperity log-visualiser-compatible logger
  Section 3  MATH UTILITIES   — rolling stats helpers (BS / option utils dropped)
  Section 4  PRODUCT CONTEXT  — per-symbol order-book snapshot
  Section 5  BASE STRATEGY    — single-symbol Strategy with internal book sim
  Section 6  PRIMITIVES       — take/make/flush building blocks (no informed-trader hook)
  Section 7  MM BASE          — MarketMakingStrategy(get_fair_value)
  Section 8  MULTI-SYMBOL BASE — for cross-product strategies (PEBBLES basket)
  Section 9  CONCRETE         — PebblesBasketStrategy + 9 TODO stubs
  Section 10 TRADER           — entry point, registers active strategies

Datamodel reference   : Appendix B of the Prosperity 5 wiki
Allowed imports       : json, math, statistics, typing, jsonpickle, numpy, pandas
Return signature      : (result: dict, conversions: int, traderData: str)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import math
from abc import abstractmethod
from copy import deepcopy
from enum import IntEnum
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
#  All Round-5 product names, group composition, and position limits.
#  Update this section first whenever a new round drops.
# ══════════════════════════════════════════════════════════════════════════════

# ── Group composition ─────────────────────────────────────────────────────────
GROUPS: Dict[str, List[str]] = {
    "GALAXY_SOUNDS": [
        "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "SLEEP_POD": [
        "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    ],
    "MICROCHIP": [
        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    ],
    "PEBBLES": ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "ROBOT": [
        "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
        "ROBOT_LAUNDRY", "ROBOT_IRONING",
    ],
    "UV_VISOR": [
        "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
        "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    ],
    "TRANSLATOR": [
        "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "PANEL": ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
    "OXYGEN_SHAKE": [
        "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    ],
    "SNACKPACK": [
        "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
    ],
}
ALL_PRODUCTS: List[str] = [p for ps in GROUPS.values() for p in ps]

# ── Position limits — every Round-5 product is capped at 10 ───────────────────
DEFAULT_LIMIT = 10
POSITION_LIMITS: Dict[str, int] = {p: DEFAULT_LIMIT for p in ALL_PRODUCTS}

# ── PEBBLES basket invariant constants ───────────────────────────────────────
# Verified from ying_research/Round_5/round5_group_analysis.ipynb §2 / §3.5:
#   sum(5 PEBBLES mids) = 50000 (full-sample std=2.80, ADF p ≈ 0).
PEBBLES_TARGET: int = 50_000
PEBBLES_SYMBOLS: List[str] = GROUPS["PEBBLES"]


# ── Signal sentinel (kept for future stateful strategies) ─────────────────────
class Signal(IntEnum):
    """Direction signal used by mean-reversion / cointegration strategies."""
    SHORT   = -1
    NEUTRAL =  0
    LONG    =  1


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LOGGER
#  Identical to the standard Prosperity logger (see trader_r4_init.py).
#  Required so the official log visualiser can parse our run output.
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Structured logger compatible with the Prosperity log visualiser."""

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
                       o.exportTariff, o.importTariff,
                       getattr(o, "sugarPrice", None), getattr(o, "sunlightIndex", None)]
        return [obs.plainValueObservations, conv]

    def _compress_orders(self, orders: Dict[Symbol, List[Order]]) -> list:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def _to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def _truncate(self, value: str, max_length: int) -> str:
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
#  Round-5 strategies need rolling-window stats and a Welford running mean.
#  Black-Scholes and option helpers from R3/R4 are dropped (R5 has no options).
# ══════════════════════════════════════════════════════════════════════════════

class MathUtils:
    """Pure-function math helpers — no internal state."""

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
        """Z-score of the last value vs a rolling window."""
        if len(values) < window:
            return None
        std = MathUtils.rolling_std(values, window)
        if std == 0:
            return None
        mean = MathUtils.rolling_mean(values, window)
        return (values[-1] - mean) / std

    @staticmethod
    def update_running_mean(
        old_mean: float, old_n: int, new_value: float
    ) -> Tuple[float, int]:
        """Online Welford update; returns (new_mean, new_n)."""
        n = old_n + 1
        return old_mean + (new_value - old_mean) / n, n


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PRODUCT CONTEXT
#  Thin wrapper around one product's slice of TradingState.  Same as R4.
# ══════════════════════════════════════════════════════════════════════════════

class ProductContext:
    """Snapshot of market data for a single product at one timestep."""

    def __init__(self, symbol: str, state: TradingState) -> None:
        self.symbol   = symbol
        self.state    = state
        self.position = state.position.get(symbol, 0)
        self.limit    = POSITION_LIMITS.get(symbol, DEFAULT_LIMIT)

        od = state.order_depths.get(symbol, OrderDepth())

        self.buy_orders: Dict[int, int] = dict(
            sorted(od.buy_orders.items(), reverse=True)
        )
        self.sell_orders: Dict[int, int] = {
            p: abs(v) for p, v in sorted(od.sell_orders.items())
        }

        self.best_bid: Optional[int] = max(self.buy_orders)  if self.buy_orders  else None
        self.best_ask: Optional[int] = min(self.sell_orders) if self.sell_orders else None

        self.wall_bid: Optional[int] = (
            max(self.buy_orders, key=self.buy_orders.__getitem__)
            if self.buy_orders else None
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

        self.max_buy:  int = self.limit - self.position
        self.max_sell: int = self.limit + self.position

    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BASE STRATEGY
#  Single-symbol strategy with internal book simulation (same as R4).
#  Multi-symbol strategies inherit from MultiSymbolStrategy (Section 8).
# ══════════════════════════════════════════════════════════════════════════════

class Strategy:
    """Single-symbol base strategy."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._orders: List[Order]  = []
        self._conversions: int     = 0
        self._buy_spent: int       = 0
        self._sell_spent: int      = 0
        self._book: Optional[OrderDepth] = None

    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        """Per-tick lifecycle. Override act() in subclasses."""
        self._orders      = []
        self._conversions = 0
        self._buy_spent   = 0
        self._sell_spent  = 0
        self._book        = deepcopy(state.order_depths.get(self.symbol, OrderDepth()))
        self._book.sell_orders = {p: abs(v) for p, v in self._book.sell_orders.items()}

        ctx = ProductContext(self.symbol, state)
        if ctx.best_bid is not None and ctx.best_ask is not None:
            self.act(ctx, state)

        return self._orders, self._conversions

    @abstractmethod
    def act(self, ctx: ProductContext, state: TradingState) -> None:
        raise NotImplementedError

    def save(self) -> Any:
        return None

    def load(self, data: Any) -> None:
        pass

    # ── Order helpers ──────────────────────────────────────────────────────────
    def buy(self, price: int, quantity: int) -> None:
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
        ctx_limit = POSITION_LIMITS.get(self.symbol, DEFAULT_LIMIT)
        remaining_capacity = ctx_limit - self._sell_spent
        qty = max(0, min(int(quantity), remaining_capacity))
        if qty <= 0:
            return
        self._orders.append(Order(self.symbol, int(price), -qty))
        self._sell_spent += qty
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — STRATEGY PRIMITIVES
#  Reusable building blocks: take favourable, zero-EV flush, post passive.
#  R5 drops the informed-trader hooks (no named traders in the trade tape).
# ══════════════════════════════════════════════════════════════════════════════

class Primitives:
    """Stateless building blocks; each method mutates the calling Strategy."""

    @staticmethod
    def take_best_orders(
        strategy: Strategy,
        ctx: ProductContext,
        fair_value: float,
        take_edge: float = 1.0,
    ) -> None:
        """Sweep every level with positive edge vs fair value."""
        for price, volume in sorted(ctx.sell_orders.items()):
            if price > fair_value - take_edge:
                break
            strategy.buy(price, volume)
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
        """Drain leftover inventory at fair value to free up position capacity."""
        fv_int = int(fair_value)
        pos_after = ctx.position + strategy._buy_spent - strategy._sell_spent
        if pos_after > 0 and fv_int in strategy._book.buy_orders:
            available = strategy._book.buy_orders[fv_int]
            qty = min(pos_after, available)
            if qty > 0:
                strategy.sell(fv_int, qty)
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
        """Post passive bid/ask just inside fair value, in front of bot quotes."""
        bid_edge = bid_edge_override if bid_edge_override is not None else make_edge
        ask_edge = ask_edge_override if ask_edge_override is not None else make_edge

        max_bid = int(fair_value) - int(bid_edge)
        min_ask = int(fair_value) + int(ask_edge)

        bid_price = max_bid
        for price in sorted(strategy._book.buy_orders.keys(), reverse=True):
            if price < max_bid:
                candidate = price + 1 if strategy._book.buy_orders[price] > 1 else price
                bid_price = min(candidate, max_bid)
                break

        ask_price = min_ask
        for price in sorted(strategy._book.sell_orders.keys()):
            if price > min_ask:
                candidate = price - 1 if strategy._book.sell_orders[price] > 1 else price
                ask_price = max(candidate, min_ask)
                break

        remaining_buy  = ctx.max_buy  - strategy._buy_spent
        remaining_sell = ctx.max_sell - strategy._sell_spent

        if remaining_buy  > 0:
            strategy.buy(bid_price, remaining_buy)
        if remaining_sell > 0:
            strategy.sell(ask_price, remaining_sell)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MARKET MAKING STRATEGY BASE
#  Subclasses only need to implement get_fair_value().
# ══════════════════════════════════════════════════════════════════════════════

class MarketMakingStrategy(Strategy):
    """take → flush → make pattern around get_fair_value()."""

    def __init__(self, symbol: str, take_edge: float = 1.0, make_edge: float = 1.0) -> None:
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
        Primitives.take_best_orders(self, ctx, fv, self.take_edge)
        Primitives.zero_ev_flush(self, ctx, fv)
        Primitives.post_passive_quotes(self, ctx, fv, self.make_edge)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — MULTI-SYMBOL STRATEGY BASE
#  For cross-product strategies (PEBBLES basket arb, future ETF/pair trades).
#  Returns Dict[Symbol, List[Order]] instead of List[Order] so one strategy
#  can fire orders into multiple symbols in a single tick.
# ══════════════════════════════════════════════════════════════════════════════

class MultiSymbolStrategy:
    """Coordinator strategy that spans multiple product symbols."""

    def __init__(self, symbols: List[str]) -> None:
        self.symbols = list(symbols)
        # Per-symbol staged orders for THIS tick (dict, not List, so we can clear in run())
        self._staged: Dict[str, List[Order]] = {s: [] for s in self.symbols}
        # Per-symbol "spent" counters for buy/sell (used for capacity clamping)
        self._buy_spent:  Dict[str, int] = {s: 0 for s in self.symbols}
        self._sell_spent: Dict[str, int] = {s: 0 for s in self.symbols}

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """Reset per-tick state, call act(), return per-symbol orders."""
        self._staged     = {s: [] for s in self.symbols}
        self._buy_spent  = {s: 0 for s in self.symbols}
        self._sell_spent = {s: 0 for s in self.symbols}
        self.act(state)
        return self._staged

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError

    def save(self) -> Any:
        return None

    def load(self, data: Any) -> None:
        pass

    # ── Per-symbol order helpers (capacity-clamped, no internal book sim) ─────
    def buy(self, symbol: str, price: int, quantity: int, current_position: int) -> None:
        limit = POSITION_LIMITS.get(symbol, DEFAULT_LIMIT)
        remaining = limit - current_position - self._buy_spent[symbol]
        qty = max(0, min(int(quantity), remaining))
        if qty <= 0:
            return
        self._staged[symbol].append(Order(symbol, int(price), qty))
        self._buy_spent[symbol] += qty

    def sell(self, symbol: str, price: int, quantity: int, current_position: int) -> None:
        limit = POSITION_LIMITS.get(symbol, DEFAULT_LIMIT)
        remaining = limit + current_position - self._sell_spent[symbol]
        qty = max(0, min(int(quantity), remaining))
        if qty <= 0:
            return
        self._staged[symbol].append(Order(symbol, int(price), -qty))
        self._sell_spent[symbol] += qty


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — CONCRETE STRATEGIES
#  v1: only PEBBLES is implemented.  The other 9 groups are TODO stubs that
#  describe the expected shape based on the notebook findings.
# ══════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# PEBBLES — 5-leg basket arbitrage (the only fully-implemented strategy in v1)
# ────────────────────────────────────────────────────────────────────────────

class PebblesBasketStrategy(MultiSymbolStrategy):
    """
    Coordinated 5-leg basket arbitrage on PEBBLES_{XS,S,M,L,XL}.

    Rationale:
      sum(5 PEBBLES mids) is a hard invariant at 50_000.
        full-sample std    = 2.80
        max-abs deviation  = 18.50  (over 30,000 historical ticks)
        ADF stat / p-value = -122.7 / ≈ 0
        per-day std        = 2.82 / 2.76 / 2.82  (no regime drift)

    For any single pebble, its synthetic fair value is
        synth_fair_i = PEBBLES_TARGET - sum(other 4 mids)
    with 1σ ≈ 2.80 and p99 ≈ 17.5.  The trader wraps each leg in a
    market-maker around synth_fair_i with take_edge = make_edge = 1.

    Regime gates (defensive — none of these have ever fired in historical data):
      DEGRADE: |basket_residual| > 30 sustained for ≥ 5 ticks
                → skip the take pass, only post passive quotes (avoid getting
                  picked off if the invariant has shifted).
      FREEZE : |basket_residual| > 60 sustained for ≥ 10 ticks
                → cancel all quoting, place market orders to flatten any open
                  position back to 0 on each leg.

    Persistent state (save/load):
      _breach_count_30 : how many consecutive ticks we have seen |residual|>30
      _breach_count_60 : how many consecutive ticks we have seen |residual|>60
      _frozen          : True once FREEZE is latched (never auto-recovers within run)
    """

    _TAKE_EDGE       = 0
    _MAKE_EDGE       = 0.5
    _DEGRADE_RESID   = 30
    _DEGRADE_TICKS   = 5
    _FREEZE_RESID    = 60
    _FREEZE_TICKS    = 10

    def __init__(self) -> None:
        super().__init__(PEBBLES_SYMBOLS)
        self._breach_count_30: int  = 0
        self._breach_count_60: int  = 0
        self._frozen:          bool = False

    # ── Helper accessors ────────────────────────────────────────────────────
    @staticmethod
    def _mid(ctx: ProductContext) -> Optional[float]:
        wm = ctx.wall_mid
        return wm if wm is not None else ctx.mid_price()

    def _basket_residual(self, mids: Dict[str, float]) -> float:
        return sum(mids.values()) - PEBBLES_TARGET

    # ── Core decision logic ─────────────────────────────────────────────────
    def act(self, state: TradingState) -> None:
        # 1. Build per-leg context and gather mids
        ctxs:   Dict[str, ProductContext] = {}
        mids:   Dict[str, float]          = {}
        for sym in self.symbols:
            c = ProductContext(sym, state)
            if c.best_bid is None or c.best_ask is None:
                # If any leg is illiquid this tick, abandon entirely — basket arb
                # requires hedging across all 5 legs.
                logger.print("PEBBLES skip: illiquid leg", sym)
                return
            ctxs[sym] = c
            mid = self._mid(c)
            if mid is None:
                logger.print("PEBBLES skip: no mid", sym)
                return
            mids[sym] = mid

        # 2. Update regime gates
        residual = self._basket_residual(mids)
        abs_resid = abs(residual)

        if abs_resid > self._FREEZE_RESID:
            self._breach_count_60 += 1
            if self._breach_count_60 >= self._FREEZE_TICKS:
                self._frozen = True
        else:
            self._breach_count_60 = 0

        if abs_resid > self._DEGRADE_RESID:
            self._breach_count_30 += 1
        else:
            self._breach_count_30 = 0
        degraded = self._breach_count_30 >= self._DEGRADE_TICKS

        logger.print(
            "PEBBLES",
            "resid", round(residual, 2),
            "deg_cnt", self._breach_count_30,
            "frz_cnt", self._breach_count_60,
            "frozen", int(self._frozen),
        )

        # 3. FREEZE: liquidate any open position on each leg, do not quote
        if self._frozen:
            for sym, c in ctxs.items():
                if c.position > 0 and c.best_bid is not None:
                    self.sell(sym, c.best_bid, c.position, c.position)
                elif c.position < 0 and c.best_ask is not None:
                    self.buy(sym, c.best_ask, -c.position, c.position)
            return

        # 4. Per-leg market-make around synth_fair
        for sym, c in ctxs.items():
            others_sum = sum(m for s, m in mids.items() if s != sym)
            synth_fair = PEBBLES_TARGET - others_sum

            # 4a. Take favourable orders (skipped in DEGRADE mode)
            if not degraded:
                # Buy every ask strictly profitable vs synth_fair
                for price, volume in sorted(c.sell_orders.items()):
                    if price > synth_fair - self._TAKE_EDGE:
                        break
                    self.buy(sym, price, volume, c.position)
                # Sell every bid strictly profitable vs synth_fair
                for price, volume in sorted(c.buy_orders.items(), reverse=True):
                    if price < synth_fair + self._TAKE_EDGE:
                        break
                    self.sell(sym, price, volume, c.position)

            # 4b. Post passive quotes inside synth_fair
            max_bid = math.floor(synth_fair - self._MAKE_EDGE)
            min_ask = math.ceil(synth_fair + self._MAKE_EDGE)

            bid_price = max_bid
            for p in sorted(c.buy_orders.keys(), reverse=True):
                if p < max_bid:
                    candidate = p + 1 if c.buy_orders[p] > 1 else p
                    bid_price = min(candidate, max_bid)
                    break

            ask_price = min_ask
            for p in sorted(c.sell_orders.keys()):
                if p > min_ask:
                    candidate = p - 1 if c.sell_orders[p] > 1 else p
                    ask_price = max(candidate, min_ask)
                    break

            # Remaining capacity (after the take pass already moved _buy_spent / _sell_spent)
            limit = POSITION_LIMITS.get(sym, DEFAULT_LIMIT)
            rem_buy  = limit - c.position - self._buy_spent[sym]
            rem_sell = limit + c.position - self._sell_spent[sym]
            if rem_buy > 0 and bid_price > 0:
                self.buy(sym, bid_price, rem_buy, c.position)
            if rem_sell > 0 and ask_price > 0:
                self.sell(sym, ask_price, rem_sell, c.position)

    def save(self) -> Dict[str, Any]:
        return {
            "deg_cnt": self._breach_count_30,
            "frz_cnt": self._breach_count_60,
            "frozen":  self._frozen,
        }

    def load(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        self._breach_count_30 = int(data.get("deg_cnt", 0))
        self._breach_count_60 = int(data.get("frz_cnt", 0))
        self._frozen          = bool(data.get("frozen", False))


# ────────────────────────────────────────────────────────────────────────────
# Stubs for the remaining 9 groups (NOT registered in Trader.__init__).
# Each entry summarises the v2 plan from the notebook so we can drop in
# code by group without re-deriving the strategy spec.
# ────────────────────────────────────────────────────────────────────────────

# TODO — SNACKPACK — soft basket + 2 internal pairs
#   - sum_std = 189.6, ADF p ≈ 0.05  (borderline-stationary basket)
#   - CHOCOLATE+VANILLA      : return-corr -0.92, sum-std 76, ADF p=0.40 (drift across days)
#   - STRAWBERRY+RASPBERRY   : return-corr -0.93, sum-std 332, ADF p=0.16 (drift across days)
#   - Strategy: rolling-mean z-score on each pair (window ≈ 1000 ticks).
#                Enter |z|>=1.5, exit |z|<=0.3, ±10/leg, freeze if rolling std > 3× FS std.

# TODO — MICROCHIP — best EG pair: SQUARE / RECTANGLE, β=-2.15, p=0.01, hl=819
#   - Z-score residual against rolling 2000-tick mean+std (≈ 2× half-life).
#   - Enter |z|>=2.0, exit |z|<=0.5; β re-estimated on rolling 5000-tick window.

# TODO — ROBOT — best EG pair: VACUUMING / LAUNDRY, β=+0.69, p=0.02, hl=1082

# TODO — SLEEP_POD — best EG pair: LAMB_WOOL / NYLON, β=+0.40, p=0.02, hl=1197

# TODO — UV_VISOR — best EG pair: AMBER / MAGENTA, β=-1.41, p=0.02, hl=1058

# TODO — OXYGEN_SHAKE — best EG pair: CHOCOLATE / GARLIC, β=+0.38, p=0.03, hl=1418

# TODO — GALAXY_SOUNDS — best EG pair: DARK_MATTER / PLANETARY_RINGS, β=+0.19, p=0.04, hl=1149

# TODO — TRANSLATOR — best EG pair: ECLIPSE_CHARCOAL / VOID_BLUE, β=+0.29, p=0.04, hl=1240

# TODO — PANEL — no tradable EG pair (best p=0.13).  Default to per-symbol MM
#                around wall_mid with take_edge=1, make_edge=1.


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — TRADER (entry point)
#  Required class name and signature per the wiki.
#  Routes per-symbol strategies through the single-symbol loop and
#  multi-symbol strategies through their own coordinator pass.
# ══════════════════════════════════════════════════════════════════════════════

class Trader:
    """Round 5 entry point. v1 ships PEBBLES only."""

    # Round-2 manual bidding hook (kept for forward compatibility; tune per round).
    def bid(self) -> int:
        return 15

    def __init__(self) -> None:
        # Per-symbol strategies (single-symbol, registered by symbol).
        self._strategies: Dict[str, Strategy] = {}

        # Multi-symbol strategies (registered by a stable name).
        self._multi: Dict[str, MultiSymbolStrategy] = {
            "PEBBLES_BASKET": PebblesBasketStrategy(),
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
        for key, strat in self._strategies.items():
            if key in saved:
                try:
                    strat.load(saved[key])
                except Exception:
                    pass
        for key, strat in self._multi.items():
            if key in saved:
                try:
                    strat.load(saved[key])
                except Exception:
                    pass

        # ── 2. Run strategies ──────────────────────────────────────────────────
        orders:      Dict[str, List[Order]] = {}
        conversions: int = 0

        # 2a. Per-symbol strategies
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

        # 2b. Multi-symbol strategies (PEBBLES basket etc.)
        for key, multi in self._multi.items():
            try:
                multi_orders = multi.run(state)
                for sym, sym_orders in multi_orders.items():
                    orders.setdefault(sym, []).extend(sym_orders)
            except Exception as e:
                logger.print(f"ERROR [multi:{key}]: {e}")

        # ── 3. Persist state for next call ─────────────────────────────────────
        new_saved: Dict[str, Any] = {}
        for key, strat in self._strategies.items():
            try:
                new_saved[key] = strat.save()
            except Exception:
                pass
        for key, multi in self._multi.items():
            try:
                new_saved[key] = multi.save()
            except Exception:
                pass
        trader_data = json.dumps(new_saved, separators=(",", ":"))

        # ── 4. Flush logger and return ─────────────────────────────────────────
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
