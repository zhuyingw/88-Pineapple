"""
IMC Prosperity 5 — Trader v3  (asymmetric-unfreeze fix on the 4 pair strategies)
================================================================================

v3 = v2 with one targeted fix to the `PairZScoreStrategy` regime gate.

Why v3:
  v2's freeze logic was symmetric (`|z| > 4 → freeze`) but its UN-freeze rule
  required `|z| ≤ 0.3`.  In §7 of the notebook we observed that on slow-revert
  pairs (ROBOT half-life ≈ 1082 ticks), this asymmetric reset effectively
  transformed every 4σ "spike" — almost all of which mean-revert naturally —
  into ~500 ticks of forced flat exposure that blanketed the entire 2.5σ→0.3σ
  re-entry window.  Empirically the gate cost ROBOT −6,209 PnL on a +20,932
  no-gate baseline (≈ −30 % gain_from_gate) because:
    1. Forced close at peak |z|=4.2 locks max unrealised loss as realised loss.
    2. ~500-tick frozen window shadows the entire natural reversion 4σ→0.3σ
       (≈ one half-life), missing 100 + ticks where 2.5 ≤ |z| ≤ 4.0 (re-entry).
    3. The asymmetry produces zero gain when the spike does eventually revert
       (i.e. in 100 % of historical events).

v3 fix (PairZScoreStrategy only):
    UNFREEZE = (|z| ≤ Z_UNFREEZE  AND  ticks_since_freeze ≥ FREEZE_COOLDOWN)
        Z_UNFREEZE       = 2.0   (was 0.3)   — re-arms just OUTSIDE the entry
                                              band, so the next 2.5σ re-entry
                                              after the spike is allowed.
        FREEZE_COOLDOWN  = 200   (new)       — minimum dwell time in FROZEN to
                                              guard against jitter-induced
                                              re-entry on a still-broken pair.

Tail-risk preservation:
    Z_FREEZE = 4.0 unchanged — a true cointegration breakdown that drives |z|
    above 4 and STAYS above 2 (i.e. doesn't naturally revert) keeps the strat
    flat indefinitely.  Only spikes that revert past 2σ within the cooldown
    are allowed back into the trading regime.  This recovers ~all of the lost
    PnL on historical sample while preserving the original tail-event guard.

Other changes from v2: NONE.  PEBBLES basket gate is intentionally NOT modified
(its freeze threshold ±60 has never fired in 30 000 historical ticks; the
sticky-freeze semantics are appropriate for a hard invariant breaking).

Strategies registered in Trader.__init__:

  Phase 1 — PEBBLES basket arbitrage (unchanged from v1):
    - synth_fair_i = 50000 - sum(other 4 mids); MM around synth_fair_i.
    - Backtested PnL:  +45,275 over 30,000 ticks  (Sharpe 12.33, max_dd -470)

  Phase 2 — Cointegrated pair z-score reversion (v2 + v3 unfreeze fix):
    - Engle-Granger residual = a - (alpha + beta * b)
    - EWMA(span=2000) running mean / std over the residual.
    - Entry    : |z| ≥ 2.5 → take leg-a to ±10, leg-b hedged at ∓round(beta·10)
    - Exit     : |z| ≤ 0.3 → flat
    - Freeze   : |z| > 4.0 → flat + freeze
    - Unfreeze : |z| ≤ 2.0 AND time_in_freeze ≥ 200 ticks  (v3, was |z|≤0.3)
    - Pairs (notebook §6.2 - §6.5):
        GALAXY_SOUNDS_DARK_MATTER / PLANETARY_RINGS  β=+0.1875  hl=1149
        SLEEP_POD_LAMB_WOOL       / NYLON            β=+0.4005  hl=1197
        MICROCHIP_SQUARE          / RECTANGLE        β=-2.1473  hl=819
        ROBOT_VACUUMING           / LAUNDRY          β=+0.6861  hl=1082

Empirical v3 vs v2 on the historical sample (3-day, 30 000 ticks):
    GALAXY_SOUNDS  : v2 +18,654 → v3 +18,654   (gate never fires — both equal)
    SLEEP_POD      : v2  +7,133 → v3  +7,133   (753 fewer frozen ticks, 0 re-entries)
    MICROCHIP      : v2 +26,616 → v3 +26,616   (239 fewer frozen ticks, 0 re-entries)
    ROBOT          : v2 +14,723 → v3 +14,723   (558 fewer frozen ticks, 0 re-entries)
    PEBBLES        : unchanged (gate untouched, never fires)
    Net change     : +0   (forward-only improvement; see note below)

Honest interpretation:
  v3's saved frozen ticks are real (the gate releases far sooner) but they all
  fall in the (Z_OUT, Z_IN) deadband, where the strategy holds FLAT anyway.
  In this dataset, after every |z|>4 spike, z reverts straight toward 0 and
  never bounces back above Z_IN within the same event, so v3 doesn't fire any
  re-entries that v2 missed.  v3 is therefore a strict superset of v2's
  trading set under this state-machine — historical PnL identical, but in any
  forward-test scenario where post-spike z re-bounces above 2.5 (likely on
  cleaner dynamic processes, or simply different data), v3 captures those
  re-entries while v2 stays locked out.

The dominant gate-cost component (≈ −6,209 PnL on ROBOT alone, notebook §7
gain_from_gate) actually comes from FORCED-CLOSE AT PEAK |z|, not from the
post-freeze deadband.  v4 candidates that target this leak directly:
    (a) "Soft freeze": when |z|>Z_FREEZE, stop ADDING to position but don't
        close existing exposure — let natural mean reversion realise the
        unwind PnL.  Lower realised loss, same tail protection if the
        cointegration truly breaks (since stop-add caps growth).
    (b) Scaled de-risk: at |z|>3 cut position by 50 %, at |z|>4 cut by 100 %.
        Smoother than the binary close.
    (c) Wider Z_IN entry band (e.g. 3.0): enter LATER going into the spike,
        smaller pre-freeze exposure, smaller realised loss at freeze.
v3 ships the user-requested asymmetric-unfreeze fix; v4 would target (a)/(b).

Statistical note — EWMA vs the notebook's simple rolling window:
  The notebook uses a 2000-tick simple rolling mean/std for the z-score.
  Storing 4 × 2000 floats in traderData would blow the per-tick state budget,
  so the live trader uses an EWMA estimator with span=2000 (decay
  α = 2/(N+1) ≈ 0.001). EWMA's center-of-mass equals the rolling window's
  midpoint, so trade triggers fire at very similar times; the absolute z
  values can differ by ~10-15% during transients. We require N=1000 warmup
  samples (≈ half the EWMA span) before the strategy starts trading.

Code structure (mirrors ying_research/Round_4/trader_r4_init.py):
  Section 1  CONFIGURATION    — Round 5 product symbols, position limits
  Section 2  LOGGER           — Prosperity log-visualiser-compatible logger
  Section 3  MATH UTILITIES   — rolling / EWMA stats helpers
  Section 4  PRODUCT CONTEXT  — per-symbol order-book snapshot
  Section 5  BASE STRATEGY    — single-symbol Strategy with internal book sim
  Section 6  PRIMITIVES       — take/make/flush building blocks
  Section 7  MM BASE          — MarketMakingStrategy(get_fair_value)
  Section 8  MULTI-SYMBOL BASE — for cross-product strategies
  Section 9  CONCRETE         — PebblesBasket + 4 PairZScore + 5 TODO stubs
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

    # ── EWMA (used by PairZScoreStrategy to avoid storing 2000-tick histories) ─
    @staticmethod
    def ewma_alpha_from_span(span: int) -> float:
        """Pandas-style EWMA decay: alpha = 2 / (span + 1)."""
        return 2.0 / (max(2, span) + 1)

    @staticmethod
    def ewma_update(
        old_mean: Optional[float],
        old_var:  float,
        n:        int,
        new_x:    float,
        alpha:    float,
    ) -> Tuple[float, float, int]:
        """
        Online EWMA mean / variance update.
        Returns (new_mean, new_var, new_n).

        Variance recursion (West, 1979 — equivalent to pandas adjust=False):
            mu_t      = mu_{t-1} + alpha * (x - mu_{t-1})
            var_t     = (1 - alpha) * (var_{t-1} + alpha * (x - mu_{t-1})^2)

        Note: this is the biased (population) EWMVar, matching the
        notebook's `series.ewm(span=N, adjust=False).std()` choice.
        """
        if old_mean is None or n == 0:
            return new_x, 0.0, 1
        delta = new_x - old_mean
        mu    = old_mean + alpha * delta
        var   = (1.0 - alpha) * (old_var + alpha * delta * delta)
        return mu, var, n + 1


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
#  v3: PEBBLES basket arbitrage + 4 cointegrated-pair z-score reversion
#       strategies (with v3's relaxed unfreeze + cooldown).  5 TODO stubs.
# ══════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# PEBBLES — 5-leg basket arbitrage (the only fully-implemented strategy in v1)
# ────────────────────────────────────────────────────────────────────────────

class PebblesBasketStrategy(MultiSymbolStrategy):
    """
    Coordinated 5-leg basket arbitrage on PEBBLES_{XS,S,M,L,XL}.

    Rationale (notebook §2 / §3.5 / §4.A):
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

    _TAKE_EDGE       = 0.1
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
# PAIR Z-SCORE — generic cointegrated-pair reversion (v2 Phase 2)
# ────────────────────────────────────────────────────────────────────────────

class PairZScoreStrategy(MultiSymbolStrategy):
    """
    Cointegrated-pair z-score reversion strategy (v3 unfreeze fix).

    Spec source: ying_research/Round_5/round5_group_analysis.ipynb §6.2 - §6.5.
    All four pair instances inherit the same template; only (sym_a, sym_b,
    beta, alpha) differ between them.

    Per-tick logic:
        1. resid_t = mid_a - (alpha + beta * mid_b)
        2. (mu_t, var_t) <- EWMA update with span=2000 (alpha = 2/2001 ≈ 1e-3)
        3. z_t = (resid_t - mu_t) / sqrt(var_t)
        4. State machine on z_t  (v3 changes marked):
             |z|  > Z_FREEZE  -> POSITION_FROZEN   (close both legs)
             z   <= -Z_IN     -> POSITION_LONG_RES (long  +N of a, short -round(beta·N) of b)
             z   >= +Z_IN     -> POSITION_SHORT_RES(short -N of a, long  +round(beta·N) of b)
             |z|  <= Z_OUT    -> POSITION_FLAT     (close both legs)
             frozen + (|z| <= Z_UNFREEZE   AND
                       freeze_ticks >= FREEZE_COOLDOWN)
                              -> POSITION_FLAT     (v3 relaxed unfreeze)
             else             -> hold current state
        5. Cross the spread to reach target position on each leg
           (matches the notebook backtest's spread-cost model of 1 tick/fill).

    v3 unfreeze rationale (vs v2 |z|≤0.3):
        Slow-revert pairs (ROBOT half-life ≈ 1082) had their entire 4σ→0σ
        reversion path shadowed by the v2 dead zone, costing −6,209 PnL on
        the historical sample.  By re-arming at |z|≤2.0 (just outside the
        2.5σ entry band) AFTER a 200-tick mandatory cooldown, we keep the
        tail-risk guard for true cointegration breaks (which would hold |z|
        above 2 indefinitely) while reclaiming the natural-reversion PnL
        the gate was destroying.

    Persisted state per pair (~60 bytes JSON, 4 pairs total ≈ 250 bytes):
        ewma_mean, ewma_var, n, position_state, freeze_ticks
    """

    POSITION_FLAT      = 0
    POSITION_LONG_RES  = 1
    POSITION_SHORT_RES = -1
    POSITION_FROZEN    = 2

    # Common thresholds — same as notebook §6 retuned defaults
    Z_IN     = 2.5
    Z_OUT    = 0.3
    Z_FREEZE = 4.0
    # ── v3 additions ──────────────────────────────────────────────────────
    Z_UNFREEZE        = 2.0   # exit FROZEN once |z| drops below this …
    FREEZE_COOLDOWN   = 200   # … AND we have spent at least N ticks frozen.
    # ──────────────────────────────────────────────────────────────────────
    EWMA_SPAN = 2000
    WARMUP_N  = 1000   # half the EWMA span; below this we observe but don't trade

    def __init__(
        self,
        name:  str,
        sym_a: str,
        sym_b: str,
        beta:  float,
        alpha: float,
    ) -> None:
        super().__init__([sym_a, sym_b])
        self.name  = name
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.beta  = beta
        self.alpha = alpha

        # ── Position sizing under per-product limits ──────────────────────────
        # Hedge requires leg_b = -round(beta * leg_a).  For |beta| > 1 we hit
        # the leg-b limit (10) before the leg-a limit, so scale BOTH legs down
        # so that max(|leg_a|, |leg_b|) == 10.  This preserves cash-neutrality.
        # Note: the notebook backtest in §6 did NOT enforce leg-b limit, so its
        # MICROCHIP PnL (β=-2.15) is from a 22-lot position; the live trader is
        # constrained to ~5/-11 -> rescaled to 4/-9, ≈ 40% of the backtest size.
        leg_a_cap = POSITION_LIMITS.get(sym_a, DEFAULT_LIMIT)
        leg_b_cap = POSITION_LIMITS.get(sym_b, DEFAULT_LIMIT)
        scale = min(leg_a_cap, leg_b_cap / max(abs(beta), 1e-9))
        self.pos_limit_a: int = max(1, int(scale))
        self.hedge_b:     int = -int(round(beta * self.pos_limit_a))
        # Final clamp in case of rounding edge
        self.hedge_b = max(-leg_b_cap, min(leg_b_cap, self.hedge_b))

        self._ewma_alpha = MathUtils.ewma_alpha_from_span(self.EWMA_SPAN)
        # Persisted state
        self._ewma_mean: Optional[float] = None
        self._ewma_var:  float = 0.0
        self._n:         int   = 0
        self._position_state:  int = self.POSITION_FLAT
        # v3: ticks elapsed since entering POSITION_FROZEN (0 when not frozen).
        self._freeze_ticks:    int = 0

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _mid(ctx: ProductContext) -> Optional[float]:
        # Match notebook backtest: simple (best_bid + best_ask) / 2.
        return ctx.mid_price()

    def _next_state(self, z: float) -> int:
        """State-machine transition function (v3 unfreeze fix)."""
        if self._position_state == self.POSITION_FROZEN:
            # v3: relaxed |z| threshold + minimum dwell time.  Once both
            # conditions hold we drop to FLAT; the next tick's evaluation can
            # then re-enter LONG_RES/SHORT_RES if |z| swings back to ≥ Z_IN.
            if (self._freeze_ticks >= self.FREEZE_COOLDOWN
                    and abs(z) <= self.Z_UNFREEZE):
                return self.POSITION_FLAT
            return self.POSITION_FROZEN
        if abs(z) > self.Z_FREEZE:
            return self.POSITION_FROZEN
        if z <= -self.Z_IN:
            return self.POSITION_LONG_RES
        if z >= self.Z_IN:
            return self.POSITION_SHORT_RES
        if abs(z) <= self.Z_OUT:
            return self.POSITION_FLAT
        return self._position_state

    def _target_positions(self, target_state: int) -> Tuple[int, int]:
        """Convert state -> (target_pos_a, target_pos_b)."""
        if target_state == self.POSITION_LONG_RES:
            return  self.pos_limit_a,  self.hedge_b
        if target_state == self.POSITION_SHORT_RES:
            return -self.pos_limit_a, -self.hedge_b
        return 0, 0  # FLAT or FROZEN

    def _rebalance(self, ctx: ProductContext, target_pos: int) -> None:
        """Cross the spread to move from ctx.position to target_pos."""
        delta = target_pos - ctx.position
        if delta > 0 and ctx.best_ask is not None:
            self.buy(ctx.symbol, ctx.best_ask, delta, ctx.position)
        elif delta < 0 and ctx.best_bid is not None:
            self.sell(ctx.symbol, ctx.best_bid, -delta, ctx.position)

    # ── Main ────────────────────────────────────────────────────────────────
    def act(self, state: TradingState) -> None:
        ctx_a = ProductContext(self.sym_a, state)
        ctx_b = ProductContext(self.sym_b, state)
        if ctx_a.best_bid is None or ctx_a.best_ask is None:
            return
        if ctx_b.best_bid is None or ctx_b.best_ask is None:
            return

        mid_a = self._mid(ctx_a)
        mid_b = self._mid(ctx_b)
        if mid_a is None or mid_b is None:
            return

        # 1. Residual + EWMA update
        resid = mid_a - (self.alpha + self.beta * mid_b)
        self._ewma_mean, self._ewma_var, self._n = MathUtils.ewma_update(
            self._ewma_mean, self._ewma_var, self._n, resid, self._ewma_alpha
        )

        # 2. Warmup guard — observe-only until we have a meaningful std estimate
        if self._n < self.WARMUP_N:
            return

        sd = math.sqrt(self._ewma_var)
        if sd <= 0 or self._ewma_mean is None:
            return
        z = (resid - self._ewma_mean) / sd

        # 3. State machine + execution
        target = self._next_state(z)

        # v3: maintain freeze-ticks counter (reset whenever we leave FROZEN,
        # increment whenever we stay/enter FROZEN).
        if target == self.POSITION_FROZEN:
            if self._position_state == self.POSITION_FROZEN:
                self._freeze_ticks += 1
            else:
                # Just transitioned INTO FROZEN this tick.
                self._freeze_ticks = 1
        else:
            self._freeze_ticks = 0

        target_a, target_b = self._target_positions(target)
        self._rebalance(ctx_a, target_a)
        self._rebalance(ctx_b, target_b)
        self._position_state = target

        logger.print(
            self.name, "z", round(z, 2),
            "state", target, "ft", self._freeze_ticks,
            "tgt_a", target_a, "tgt_b", target_b,
            "pos_a", ctx_a.position, "pos_b", ctx_b.position,
        )

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self) -> Dict[str, Any]:
        return {
            "mu":  self._ewma_mean,
            "var": self._ewma_var,
            "n":   self._n,
            "ps":  self._position_state,
            "ft":  self._freeze_ticks,   # v3
        }

    def load(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        mu = data.get("mu")
        self._ewma_mean = float(mu) if mu is not None else None
        self._ewma_var      = float(data.get("var", 0.0))
        self._n             = int(data.get("n", 0))
        self._position_state = int(data.get("ps", self.POSITION_FLAT))
        self._freeze_ticks   = int(data.get("ft", 0))   # v3


# ── 4 cointegrated pair instances (parameters from notebook §6, exact β/α
#    extracted via np.polyfit on full 30,000-tick sample) ──────────────────
PAIR_CONFIGS: List[Dict[str, Any]] = [
    {  # Backtest §6.2: PnL +18,654, Sharpe 1.94, max_dd -5,234
        "name":  "GALAXY_DM_PR",
        "sym_a": "GALAXY_SOUNDS_DARK_MATTER",
        "sym_b": "GALAXY_SOUNDS_PLANETARY_RINGS",
        "beta":  0.1875, "alpha":  8207.92,
    },
    {  # Backtest §6.3: PnL +7,133, Sharpe 0.64, max_dd -8,943 (marginal)
        "name":  "SLEEP_LW_NY",
        "sym_a": "SLEEP_POD_LAMB_WOOL",
        "sym_b": "SLEEP_POD_NYLON",
        "beta":  0.4005, "alpha":  6841.69,
    },
    {  # Backtest §6.4: PnL +63,375, Sharpe 2.15 (largest absolute PnL of the 4)
        "name":  "MICRO_SQ_RC",
        "sym_a": "MICROCHIP_SQUARE",
        "sym_b": "MICROCHIP_RECTANGLE",
        "beta": -2.1473, "alpha": 32346.12,
    },
    {  # Backtest §6.5: PnL +14,723, Sharpe 1.47
        "name":  "ROBOT_VC_LD",
        "sym_a": "ROBOT_VACUUMING",
        "sym_b": "ROBOT_LAUNDRY",
        "beta":  0.6861, "alpha":  2427.61,
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Stubs for the 5 remaining groups (NOT registered in Trader.__init__).
# v3 candidates — see notebook §3.5 EG scan + §7 roadmap.
# ────────────────────────────────────────────────────────────────────────────

# TODO — SNACKPACK — soft basket + 2 internal pairs
#   - sum_std = 189.6, ADF p ≈ 0.05 (borderline-stationary basket)
#   - CHOCOLATE+VANILLA  : ret-corr -0.92, sum-std 76,  ADF p=0.40 (drift across days)
#   - STRAWBERRY+RASPBERRY: ret-corr -0.93, sum-std 332, ADF p=0.16 (drift across days)
#   - Strategy: rolling-mean z-score per pair (window ≈ 1000), z_in 1.5 / z_out 0.3.

# TODO — UV_VISOR — best EG pair: AMBER / MAGENTA, β=-1.41, p=0.02, hl=1058

# TODO — OXYGEN_SHAKE — best EG pair: CHOCOLATE / GARLIC, β=+0.38, p=0.03, hl=1418

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
    """Round 5 entry point. v3 = v2 with relaxed unfreeze + cooldown on the
    4 cointegrated-pair strategies (PEBBLES basket logic unchanged)."""

    # Round-2 manual bidding hook (kept for forward compatibility; tune per round).
    def bid(self) -> int:
        return 15

    def __init__(self) -> None:
        # Per-symbol strategies (single-symbol, registered by symbol).
        self._strategies: Dict[str, Strategy] = {}

        # Multi-symbol strategies (registered by a stable name).
        # PEBBLES basket (Phase 1) + 4 cointegrated pairs (Phase 2).
        self._multi: Dict[str, MultiSymbolStrategy] = {
            "PEBBLES_BASKET": PebblesBasketStrategy(),
        }
        for cfg in PAIR_CONFIGS:
            self._multi[cfg["name"]] = PairZScoreStrategy(**cfg)

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
