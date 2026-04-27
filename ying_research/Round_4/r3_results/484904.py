import math
from abc import abstractmethod, ABC
from collections import deque
from copy import deepcopy
from math import exp, log, sqrt
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, TypeAlias
import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


HYDROGEL_PACK = "HYDROGEL_PACK"
VELVETFRUIT_EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_ALL_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_REGRESSION_STRIKES = [5100, 5200, 5300]
VEV_ACTIVE_STRIKES = VEV_REGRESSION_STRIKES
VEV_ITM_STRIKES: List[int] = []
VEV_IV_ARB_STRIKES: List[int] = []
VEV_STRUCTURAL_LONG_STRIKES: List[int] = []
VEV_STRUCTURAL_SHORT_STRIKES: List[int] = []
VEV_ALL_SYMBOLS = [f"VEV_{strike}" for strike in VEV_ALL_STRIKES]
VEV_TTE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_SURFACE_FIT_STRIKES = [5100, 5200, 5300, 5400, 5500]

ROUND3_DAY_TICKS = 1_000_000
ROUND3_DAYS_LEFT_CANDIDATES = [5.0, 6.0, 7.0, 8.0]
VEV_MAX_FIT_SPREAD = 8
VEV_HEDGE_RATIO = 0.10
VELVET_HEDGE_MAX_TAKE = 30   # kept for compatibility; unused by the MM variant
VEV_STRIKE_IV_OFFSETS: Dict[int, float] = {}

HYDROGEL_MM_FLOW_ALPHA = 0.16
HYDROGEL_MM_FAIR_ALPHA = 0.06
HYDROGEL_MM_MICRO_COEFF = 0.28
HYDROGEL_MM_IMBALANCE_COEFF = 0.24
HYDROGEL_MM_MR_COEFF = 0.90
HYDROGEL_MM_FLOW_COEFF = 0.10
HYDROGEL_MM_INV_COEFF = 12.0
HYDROGEL_MM_BASE_HALF = 2.2
HYDROGEL_MM_SPREAD_COEFF = 0.24
HYDROGEL_MM_INV_HALF_COEFF = 2.2
HYDROGEL_MM_FLOW_HALF_COEFF = 0.15
HYDROGEL_MM_TAKE_EDGE = 2.0
HYDROGEL_MM_MAX_TAKE = 72
HYDROGEL_MM_MAX_MAKE = 84

VELVET_MM_FLOW_ALPHA = 0.12
VELVET_MM_FAIR_ALPHA = 0.04
VELVET_MM_MICRO_COEFF = 0.18
VELVET_MM_IMBALANCE_COEFF = 0.12
VELVET_MM_MR_COEFF = 0.45
VELVET_MM_FLOW_COEFF = 0.10
VELVET_MM_INV_COEFF = 6.0
VELVET_MM_BASE_HALF = 2.0
VELVET_MM_SPREAD_COEFF = 0.20
VELVET_MM_SPREAD_INV = 1.4
VELVET_MM_SPREAD_FLOW = 0.15
VELVET_MM_TAKE_EDGE = 3.0
VELVET_MM_MAX_TAKE = 10
VELVET_MM_MAX_MAKE = 20


VEV_FLOW_ALPHA = 0.20
VEV_POSITION_SKEW = 1.5
VEV_FLOW_SKEW = 0.75

# ITM / IV-arb / Structural constants referenced by Round3VoucherStrategy
# and stub classes. Strike lists are all empty so never exercised at runtime.
VEV_ITM_REF_ALPHA = 0.08
VEV_ITM_DEV_ALPHA = 0.05
VEV_ITM_OPEN_EDGE = 0.75
VEV_ITM_CLOSE_EDGE = 0.20
VEV_ITM_MAX_TAKE = 28
VEV_ITM_MAX_MAKE = 8
VEV_IV_REF_ALPHA = 0.08
VEV_IV_DEV_ALPHA = 0.05
VEV_IV_OPEN_EDGE = 0.85
VEV_IV_CLOSE_EDGE = 0.25
VEV_IV_MAX_TAKE = 28
VEV_IV_MAX_MAKE = 8
VEV_STRUCTURAL_OPEN_EDGE: Dict[int, float] = {}
VEV_STRUCTURAL_CLOSE_EDGE: Dict[int, float] = {}
VEV_STRUCTURAL_BASE_SIZE: Dict[int, int] = {}
VEV_STRUCTURAL_MAX_TAKE: Dict[int, int] = {}
VEV_STRUCTURAL_SOFT_LIMIT: Dict[int, int] = {}
VEV_STRUCTURAL_MAX_MAKE = 12
VEV_STRUCTURAL_FLOW_SKEW = 0.2
VEV_STRUCTURAL_INV_SKEW = 1.5




VEV_REG_COEFS = {
    5100: (-3950.957951, 0.784321),
    5200: (-2871.358698, 0.565115),
    5300: (-1704.686347, 0.333603),
}
VEV_REG_OPEN_EDGE = {
    5100: 0.80,
    5200: 0.70,
    5300: 0.75,
}
VEV_REG_CLOSE_EDGE = {
    5100: 0.20,
    5200: 0.18,
    5300: 0.20,
}
VEV_REG_BASE_SIZE = {
    5100: 18,
    5200: 24,
    5300: 20,
}
VEV_REG_MAX_TAKE = {
    5100: 54,
    5200: 72,
    5300: 60,
}
VEV_REG_SOFT_LIMIT = {
    5100: 180,
    5200: 220,
    5300: 180,
}
VEV_REG_MAX_MAKE = {
    5100: 8,
    5200: 10,
    5300: 8,
}
VEV_REG_SPOT_MICRO_COEFF = 0.0
VEV_REG_FLOW_SKEW = 0.0
VEV_REG_INV_SKEW = 0.50

# Fits from Round 3 historical data on core strikes using the same moneyness
# convention as the old Round 5 option infrastructure: log(K / S) / sqrt(T).
VEV_ASK_PARAMS = {"a": 0.15206328578304454, "b": -0.027837821487215083, "c": 0.23363990997996087}
VEV_BID_PARAMS = {"a": -0.1298748026776373, "b": 0.040522408302738226, "c": 0.22824194023723748}
VEV_MID_PARAMS = {"a": 0.027912493133482198, "b": 0.0024014737715807987, "c": 0.23074743825161328}


class Round3StateCache:
    timestamp: Optional[int] = None
    tte: Optional[float] = None
    days_left: Optional[float] = None
    underlying_mid: Optional[float] = None
    pending_underlying_orders: int = 0
    option_delta_target: float = 0.0
    surface_slope: Optional[float] = None
    surface_intercept: Optional[float] = None
    fair_ivs: Dict[int, float] = {}
    market_ivs: Dict[int, float] = {}

    @classmethod
    def reset_for_run(cls) -> None:
        cls.pending_underlying_orders = 0
        cls.option_delta_target = 0.0
        cls.timestamp = None
        cls.tte = None
        cls.days_left = None
        cls.underlying_mid = None
        cls.surface_slope = None
        cls.surface_intercept = None
        cls.fair_ivs = {}
        cls.market_ivs = {}


class MarketUtils:
    """Holds utility functions used across strategies."""

    def safe_best_bid(self, order_depth) -> Optional[int]:
        if not order_depth or not order_depth.buy_orders:
            return None
        return max(order_depth.buy_orders.keys())

    def safe_best_ask(self, order_depth) -> Optional[int]:
        if not order_depth or not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys())

    def has_liquidity(self, order_depth) -> bool:
        return bool(order_depth and order_depth.buy_orders and order_depth.sell_orders)

    def rolling_mean_std(self, values: List[float], period: int) -> Tuple[float, float]:
        window = values[-period:]
        n = len(window)
        if n == 0:
            return 0.0, 0.0
        m = sum(window) / n
        var = sum((x - m) ** 2 for x in window) / n
        return m, math.sqrt(var)

    def book_mid(self, order_depth: Optional[OrderDepth]) -> Optional[float]:
        if not order_depth:
            return None
        best_bid = self.safe_best_bid(order_depth)
        best_ask = self.safe_best_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return best_bid + 0.5
        if best_ask is not None:
            return best_ask - 0.5
        return None

    def microprice(self, order_depth: Optional[OrderDepth]) -> Optional[float]:
        if not order_depth:
            return None

        best_bid = self.safe_best_bid(order_depth)
        best_ask = self.safe_best_ask(order_depth)
        if best_bid is None or best_ask is None:
            return self.book_mid(order_depth)

        bid_size = order_depth.buy_orders.get(best_bid, 0)
        ask_size = -order_depth.sell_orders.get(best_ask, 0)
        total_size = bid_size + ask_size
        if total_size <= 0:
            return self.book_mid(order_depth)

        return (best_bid * ask_size + best_ask * bid_size) / total_size

    def top_level_imbalance(self, order_depth: Optional[OrderDepth]) -> float:
        if not order_depth:
            return 0.0
        best_bid = self.safe_best_bid(order_depth)
        best_ask = self.safe_best_ask(order_depth)
        if best_bid is None or best_ask is None:
            return 0.0
        bid_size = max(0, order_depth.buy_orders.get(best_bid, 0))
        ask_size = max(0, -order_depth.sell_orders.get(best_ask, 0))
        total_size = bid_size + ask_size
        if total_size <= 0:
            return 0.0
        return (bid_size - ask_size) / total_size

    def top_level_spread(self, order_depth: Optional[OrderDepth]) -> Optional[int]:
        if not order_depth:
            return None
        best_bid = self.safe_best_bid(order_depth)
        best_ask = self.safe_best_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def recent_own_trade_pressure(self, state: TradingState, symbol: str) -> float:
        recent_cutoff = state.timestamp - 200
        pressure = 0.0
        for trade in state.own_trades.get(symbol, []):
            if trade.timestamp < recent_cutoff:
                continue
            quantity = abs(trade.quantity)
            if trade.buyer == "SUBMISSION":
                pressure -= quantity
            elif trade.seller == "SUBMISSION":
                pressure += quantity
        return pressure

    def _round3_candidate_tte(self, days_left: float, timestamp: int) -> float:
        day_fraction = max(0.0, min(1.0, timestamp / ROUND3_DAY_TICKS))
        return max((days_left - day_fraction) / 365.0, 1e-6)

    def clamp_iv(self, iv: float) -> float:
        return max(0.05, min(1.5, float(iv)))

    def option_mid_iv(
        self,
        order_depth,
        spot: float,
        strike: int,
        tte: float,
        model: Optional["BlackScholes"] = None,
    ) -> Optional[float]:
        mid_price = self.book_mid(order_depth)
        if mid_price is None or spot <= 0 or tte <= 0:
            return None

        intrinsic = max(0.0, spot - strike)
        if mid_price <= intrinsic + 0.05:
            return None

        if model is None:
            model = BlackScholes()

        iv = model.implied_volatility(mid_price, spot, strike, tte)
        if not math.isfinite(iv):
            return None
        return self.clamp_iv(iv)

    def round3_tte(self, state: TradingState) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if Round3StateCache.timestamp == state.timestamp and Round3StateCache.tte is not None:
            return Round3StateCache.tte, Round3StateCache.days_left, Round3StateCache.underlying_mid

        underlying_mid = self.book_mid(state.order_depths.get(VELVETFRUIT_EXTRACT))
        Round3StateCache.timestamp = state.timestamp
        Round3StateCache.underlying_mid = underlying_mid

        if underlying_mid is None or underlying_mid <= 0:
            Round3StateCache.days_left = ROUND3_DAYS_LEFT_CANDIDATES[0]
            Round3StateCache.tte = self._round3_candidate_tte(Round3StateCache.days_left, state.timestamp)
            return Round3StateCache.tte, Round3StateCache.days_left, Round3StateCache.underlying_mid

        best_days_left = ROUND3_DAYS_LEFT_CANDIDATES[0]
        best_score = float("inf")

        for candidate_days_left in ROUND3_DAYS_LEFT_CANDIDATES:
            tte = self._round3_candidate_tte(candidate_days_left, state.timestamp)
            errors: List[float] = []
            for strike in VEV_TTE_STRIKES:
                symbol = f"VEV_{strike}"
                mid_price = self.book_mid(state.order_depths.get(symbol))
                if mid_price is None:
                    continue
                moneyness = np.log(strike / underlying_mid) / np.sqrt(tte)
                theoretical_iv = (
                    VEV_MID_PARAMS["c"]
                    + VEV_MID_PARAMS["b"] * moneyness
                    + VEV_MID_PARAMS["a"] * moneyness * moneyness
                )
                theoretical_price = BlackScholes().black_scholes_call(underlying_mid, strike, tte, theoretical_iv)
                errors.append(abs(mid_price - theoretical_price))
            if len(errors) >= 3:
                score = float(sum(errors) / len(errors))
                if score < best_score:
                    best_score = score
                    best_days_left = candidate_days_left

        Round3StateCache.days_left = best_days_left
        Round3StateCache.tte = self._round3_candidate_tte(best_days_left, state.timestamp)
        return Round3StateCache.tte, Round3StateCache.days_left, Round3StateCache.underlying_mid

    def round3_iv_surface(
        self, state: TradingState
    ) -> Tuple[Optional[float], Optional[float], Dict[int, float], Dict[int, float]]:
        tte, _, underlying_mid = self.round3_tte(state)
        if (
            Round3StateCache.timestamp == state.timestamp
            and Round3StateCache.surface_intercept is not None
            and Round3StateCache.surface_slope is not None
        ):
            return (
                Round3StateCache.surface_intercept,
                Round3StateCache.surface_slope,
                Round3StateCache.market_ivs,
                Round3StateCache.fair_ivs,
            )

        if tte is None or tte <= 0 or underlying_mid is None or underlying_mid <= 0:
            return None, None, {}, {}

        model = BlackScholes()
        xs: List[float] = []
        ys: List[float] = []
        market_ivs: Dict[int, float] = {}

        for strike in VEV_SURFACE_FIT_STRIKES:
            symbol = f"VEV_{strike}"
            order_depth = state.order_depths.get(symbol)
            if not order_depth:
                continue

            best_bid = self.safe_best_bid(order_depth)
            best_ask = self.safe_best_ask(order_depth)
            if best_bid is None or best_ask is None:
                continue
            if best_ask - best_bid > VEV_MAX_FIT_SPREAD:
                continue

            market_iv = self.option_mid_iv(order_depth, underlying_mid, strike, tte, model)
            if market_iv is None:
                continue

            x = np.log(strike / underlying_mid) / np.sqrt(tte)
            xs.append(float(x))
            ys.append(float(market_iv))
            market_ivs[strike] = float(market_iv)

        if not ys:
            return None, None, {}, {}

        if len(ys) >= 3:
            slope, intercept = np.polyfit(xs, ys, 1)
            surface_slope = float(slope)
            surface_intercept = float(intercept)
        else:
            surface_slope = 0.0
            surface_intercept = float(sum(ys) / len(ys))

        fair_ivs: Dict[int, float] = {}
        for strike in VEV_ACTIVE_STRIKES:
            x = np.log(strike / underlying_mid) / np.sqrt(tte)
            fair_iv = surface_intercept + surface_slope * x + VEV_STRIKE_IV_OFFSETS.get(strike, 0.0)
            fair_ivs[strike] = self.clamp_iv(fair_iv)

        Round3StateCache.surface_intercept = surface_intercept
        Round3StateCache.surface_slope = surface_slope
        Round3StateCache.market_ivs = market_ivs
        Round3StateCache.fair_ivs = fair_ivs
        return surface_intercept, surface_slope, market_ivs, fair_ivs

    def round3_target_fair_iv(self, state: TradingState, target_strike: int) -> Optional[float]:
        tte, _, underlying_mid = self.round3_tte(state)
        if tte is None or tte <= 0 or underlying_mid is None or underlying_mid <= 0:
            return None

        model = BlackScholes()
        xs: List[float] = []
        ys: List[float] = []

        for strike in VEV_SURFACE_FIT_STRIKES:
            if strike == target_strike:
                continue

            symbol = f"VEV_{strike}"
            order_depth = state.order_depths.get(symbol)
            if not order_depth:
                continue

            best_bid = self.safe_best_bid(order_depth)
            best_ask = self.safe_best_ask(order_depth)
            if best_bid is None or best_ask is None:
                continue
            if best_ask - best_bid > VEV_MAX_FIT_SPREAD:
                continue

            market_iv = self.option_mid_iv(order_depth, underlying_mid, strike, tte, model)
            if market_iv is None:
                continue

            x = np.log(strike / underlying_mid) / np.sqrt(tte)
            xs.append(float(x))
            ys.append(float(market_iv))

        if not ys:
            return None

        if len(ys) >= 3:
            slope, intercept = np.polyfit(xs, ys, 1)
            fair_iv = float(intercept + slope * (np.log(target_strike / underlying_mid) / np.sqrt(tte)))
        else:
            fair_iv = float(sum(ys) / len(ys))

        fair_iv += VEV_STRIKE_IV_OFFSETS.get(target_strike, 0.0)
        return self.clamp_iv(fair_iv)


class InteractionBlocks:
    """Wraps strategy helper functions: mean reversion, taking, making, zero-EV."""

    def __init__(self, utils: MarketUtils):
        self.u = utils

    def mean_reversion_taker(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        price_array: List[float],
        fair_price: float,
        period: int,
        z_score_threshold: float,
        fixed_threshold: float,
    ) -> None:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal

        if len(price_array) < period:
            return

        rolling_mean = sum(price_array[-period:]) / period
        rolling_std = (sum((x - rolling_mean) ** 2 for x in price_array[-period:]) / period) ** 0.5

        if rolling_std == 0:
            return

        deviation = fair_price - rolling_mean
        z_score = deviation / rolling_std

        if z_score < -z_score_threshold and deviation < -fixed_threshold and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            amount_to_buy = min(
                -order_depth.sell_orders[best_ask],
                limit - position - parent.total_buying_amount,
            )
            if amount_to_buy > 0 and best_ask > 0:
                parent.buy(best_ask, amount_to_buy)
                parent.total_buying_amount += amount_to_buy

        elif z_score > z_score_threshold and deviation > fixed_threshold and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            amount_to_sell = min(
                order_depth.buy_orders[best_bid],
                limit + position - parent.total_selling_amount,
            )
            if amount_to_sell > 0 and best_bid > 0:
                parent.sell(best_bid, amount_to_sell)
                parent.total_selling_amount += amount_to_sell

    def market_taking_strategy(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        fair_buying_price: float,
        fair_selling_price: float,
        max_size: int,
    ) -> Tuple[int, int]:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal
        sizes = [0, 0]

        # Market taking: selling orders
        max_buy_amount = max_size
        if order_depth.sell_orders:
            asks = sorted(order_depth.sell_orders.keys())
            for best_buying_price in asks:
                if best_buying_price <= fair_buying_price:
                    best_buying_amount = -order_depth.sell_orders[best_buying_price]
                    amount_to_buy = min(
                        best_buying_amount, limit - position - parent.total_buying_amount, max_buy_amount
                    )
                    if amount_to_buy > 0:
                        parent.buy(best_buying_price, amount_to_buy)
                        sizes[0] += amount_to_buy
                        max_buy_amount -= amount_to_buy

        # Market taking: buying orders
        max_sell_amount = max_size
        if order_depth.buy_orders:
            bids = sorted(order_depth.buy_orders.keys(), reverse=True)
            for best_selling_price in bids:
                if best_selling_price >= fair_selling_price:
                    best_selling_amount = order_depth.buy_orders[best_selling_price]
                    amount_to_sell = min(
                        best_selling_amount, limit + position - parent.total_selling_amount, max_sell_amount
                    )
                    if amount_to_sell > 0:
                        parent.sell(best_selling_price, amount_to_sell)
                        sizes[1] += amount_to_sell
                        max_sell_amount -= amount_to_sell

        return sizes[0], sizes[1]

    def market_making_strategy(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        zero_ev_bid: float,
        zero_ev_ask: float,
        pos_ev_bid: float,
        pos_ev_ask: float,
        max_bid_size: int,
        max_ask_size: int,
    ) -> None:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal

        # Market making: buy orders
        remaining_buy_capacity = min(limit - position - parent.total_buying_amount, max_bid_size)
        if remaining_buy_capacity > 0:
            max_buy_price = (
                max(
                    [price for price in order_depth.buy_orders.keys() if price < zero_ev_bid],
                    default=pos_ev_bid,
                )
                + 1
            )
            buy_price = min(max_buy_price, pos_ev_bid)
            parent.buy(buy_price, remaining_buy_capacity)

        # Market making: sell orders
        remaining_sell_capacity = min(limit + position - parent.total_selling_amount, max_ask_size)
        if remaining_sell_capacity > 0:
            min_sell_price = (
                min(
                    [price for price in order_depth.sell_orders.keys() if price > zero_ev_ask],
                    default=pos_ev_ask,
                )
                - 1
            )
            sell_price = max(min_sell_price, pos_ev_ask)
            parent.sell(sell_price, remaining_sell_capacity)

    def zero_ev_trades(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        fair_buying_price: float,
        fair_selling_price: float,
    ) -> None:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal

        # Zero EV sell trades
        if position > 0 and fair_selling_price in order_depth.buy_orders:
            amount_to_sell = min(
                position,
                order_depth.buy_orders[fair_selling_price],
                limit + position - parent.total_selling_amount,
            )
            if amount_to_sell > 0:
                parent.sell(fair_selling_price, amount_to_sell)

        # Zero EV buy trades
        elif position < 0 and fair_buying_price in order_depth.sell_orders:
            amount_to_buy = min(
                -position,
                -order_depth.sell_orders[fair_buying_price],
                limit - position - parent.total_buying_amount,
            )
            if amount_to_buy > 0:
                parent.buy(fair_buying_price, amount_to_buy)


class Strategy(ABC):
    """Base class unchanged in logic; provides order handling and internal book simulation."""

    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.total_buying_amount = 0
        self.total_selling_amount = 0
        self.order_depth_internal = None
        self.orders = []
        self.conversions = 0

    @abstractmethod
    def act(self, state) -> None:
        raise NotImplementedError()

    def run(self, state) -> Tuple[List, int]:
        self.total_buying_amount = 0
        self.total_selling_amount = 0
        self.orders = []
        self.conversions = 0

        self.order_depth_internal = deepcopy(state.order_depths.get(self.symbol, []))

        self.act(state)

        return self.orders, self.conversions

    def popular_price_calculator(self, order_depth) -> float:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if not buy_orders and not sell_orders:
            return 0
        elif not buy_orders:
            popular_buying_price = min(sell_orders.keys()) - 1
        else:
            popular_buying_price = max(buy_orders.keys())

        if not sell_orders:
            popular_selling_price = max(buy_orders.keys()) + 1
        else:
            popular_selling_price = min(sell_orders.keys())

        popular_price = (popular_buying_price + popular_selling_price) / 2

        return popular_price

    def buy(self, price: int, quantity: int) -> None:
        assert isinstance(price, int)
        assert isinstance(quantity, int)
        assert price > 0
        assert quantity > 0

        self.orders.append(Order(self.symbol, price, quantity))
        self.total_buying_amount += quantity

        remaining_quantity = quantity
        while (
            price >= min(self.order_depth_internal.sell_orders.keys(), default=float("inf")) and remaining_quantity > 0
        ):
            ask_price = min(self.order_depth_internal.sell_orders.keys())
            ask_size = -self.order_depth_internal.sell_orders[ask_price]
            if ask_size > remaining_quantity:
                self.order_depth_internal.sell_orders[ask_price] -= remaining_quantity
                remaining_quantity = 0
            else:
                self.order_depth_internal.sell_orders.pop(ask_price)
                remaining_quantity -= ask_size

    def sell(self, price: int, quantity: int) -> None:
        assert isinstance(price, int)
        assert isinstance(quantity, int)
        assert price > 0
        assert quantity > 0

        self.orders.append(Order(self.symbol, price, -quantity))
        self.total_selling_amount += quantity

        remaining_quantity = quantity
        while (
            price <= max(self.order_depth_internal.buy_orders.keys(), default=float("-inf")) and remaining_quantity > 0
        ):
            bid_price = max(self.order_depth_internal.buy_orders.keys())
            bid_size = self.order_depth_internal.buy_orders[bid_price]
            if bid_size > remaining_quantity:
                self.order_depth_internal.buy_orders[bid_price] -= remaining_quantity
                remaining_quantity = 0
            else:
                self.order_depth_internal.buy_orders.pop(bid_price)
                remaining_quantity -= bid_size

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class BlackScholes:

    def black_scholes_call(self, spot, strike, time_to_expiry, volatility):
        if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
            return max(0.0, spot - strike)
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    def delta(self, spot, strike, time_to_expiry, volatility):
        if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        return NormalDist().cdf(d1)

    def vega(self, spot, strike, time_to_expiry, volatility):
        if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        return spot * NormalDist().pdf(d1) * sqrt(time_to_expiry)

    def implied_volatility(self, call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = self.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class HydrogelMMStrategy(Strategy):
    _TAKE_EDGE = 2
    _MAKE_EDGE = 3
    _INVENTORY_SKEW_TICKS = 4.0
    _SKEW_DEAD_ZONE = 0.50
    _SKEW_POWER = 2.0

    _MEAN_ALPHA = 0.005
    _MEAN_REVERSION_STRENGTH = 0.40
    _MEAN_REVERSION_CAP = 7.0
    _MIN_MEAN_SAMPLES = 40

    _SIDE_BIAS_THRESHOLD = 1.0
    _SIDE_BIAS_TICKS = 1
    _FLATTEN_EDGE = 2
    _FLATTEN_FRACTION = 0.35

    def __init__(self, symbol: str, limit: int, half_width: int, max_trade_size: int, utils: MarketUtils, blocks: InteractionBlocks) -> None:
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.half_width = half_width
        self.max_trade_size = max_trade_size
        self.utils = utils
        self.blocks = blocks
        self._ema_mid: Optional[float] = None
        self._ema_samples: int = 0
        self._last_dev: float = 0.0

    def _wall_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        wall_bid = max(order_depth.buy_orders, key=order_depth.buy_orders.__getitem__)
        wall_ask = min(order_depth.sell_orders, key=order_depth.sell_orders.__getitem__)
        return (wall_bid + wall_ask) / 2.0

    def get_fair_value(self, order_depth: OrderDepth) -> Optional[float]:
        wm = self._wall_mid(order_depth)
        if wm is None:
            return None

        prev_ema = self._ema_mid
        self._update_ema(wm)
        if prev_ema is None or self._ema_samples < self._MIN_MEAN_SAMPLES:
            self._last_dev = 0.0
            return wm

        self._last_dev = prev_ema - wm
        raw_tilt = self._last_dev * self._MEAN_REVERSION_STRENGTH
        tilt = max(-self._MEAN_REVERSION_CAP, min(self._MEAN_REVERSION_CAP, raw_tilt))
        return wm + tilt

    def _update_ema(self, wall_mid: float) -> None:
        if self._ema_mid is None:
            self._ema_mid = wall_mid
            self._ema_samples = 1
            return

        self._ema_mid = self._MEAN_ALPHA * wall_mid + (1.0 - self._MEAN_ALPHA) * self._ema_mid
        self._ema_samples += 1

    def _position_after_orders(self, state: TradingState) -> int:
        return state.position.get(self.symbol, 0) + self.total_buying_amount - self.total_selling_amount

    def _inventory_skewed_fair_value(self, state: TradingState, fair_value: float) -> float:
        if self.limit <= 0:
            return fair_value

        inventory_ratio = self._position_after_orders(state) / self.limit
        abs_ratio = abs(inventory_ratio)
        if abs_ratio <= self._SKEW_DEAD_ZONE:
            return fair_value

        excess_ratio = (abs_ratio - self._SKEW_DEAD_ZONE) / (1.0 - self._SKEW_DEAD_ZONE)
        skew = math.copysign(self._INVENTORY_SKEW_TICKS * (excess_ratio ** self._SKEW_POWER), inventory_ratio)
        return fair_value - skew

    def _quote_edges(self) -> Tuple[float, float]:
        bid_edge = float(self._MAKE_EDGE)
        ask_edge = float(self._MAKE_EDGE)
        if self._last_dev > self._SIDE_BIAS_THRESHOLD:
            bid_edge = max(1.0, bid_edge - self._SIDE_BIAS_TICKS)
            ask_edge += self._SIDE_BIAS_TICKS
        elif self._last_dev < -self._SIDE_BIAS_THRESHOLD:
            bid_edge += self._SIDE_BIAS_TICKS
            ask_edge = max(1.0, ask_edge - self._SIDE_BIAS_TICKS)
        return bid_edge, ask_edge

    def _near_fair_flatten(self, state: TradingState, fair_value: float) -> None:
        pos_after_orders = self._position_after_orders(state)
        if pos_after_orders == 0:
            return

        target_qty = max(1, int(abs(pos_after_orders) * self._FLATTEN_FRACTION))

        if pos_after_orders > 0 and self.order_depth_internal.buy_orders:
            best_bid = max(self.order_depth_internal.buy_orders)
            if best_bid >= fair_value - self._FLATTEN_EDGE:
                qty = min(target_qty, pos_after_orders, self.order_depth_internal.buy_orders[best_bid])
                if qty > 0:
                    self.sell(best_bid, qty)

        elif pos_after_orders < 0 and self.order_depth_internal.sell_orders:
            best_ask = min(self.order_depth_internal.sell_orders)
            if best_ask <= fair_value + self._FLATTEN_EDGE:
                qty = min(target_qty, -pos_after_orders, -self.order_depth_internal.sell_orders[best_ask])
                if qty > 0:
                    self.buy(best_ask, qty)

    def act(self, state) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return

        fv = self.get_fair_value(order_depth)
        if fv is None:
            return

        for price in sorted(self.order_depth_internal.sell_orders):
            if price > fv - self._TAKE_EDGE:
                break
            available = -self.order_depth_internal.sell_orders[price]
            qty = min(available, self.limit - state.position.get(self.symbol, 0) - self.total_buying_amount)
            if qty > 0:
                self.buy(price, qty)

        for price in sorted(self.order_depth_internal.buy_orders, reverse=True):
            if price < fv + self._TAKE_EDGE:
                break
            available = self.order_depth_internal.buy_orders[price]
            qty = min(available, self.limit + state.position.get(self.symbol, 0) - self.total_selling_amount)
            if qty > 0:
                self.sell(price, qty)

        fv_int = int(fv)
        pos_after = self._position_after_orders(state)
        if pos_after > 0 and fv_int in self.order_depth_internal.buy_orders:
            qty = min(pos_after, self.order_depth_internal.buy_orders[fv_int])
            if qty > 0:
                self.sell(fv_int, qty)
        elif pos_after < 0 and fv_int in self.order_depth_internal.sell_orders:
            qty = min(-pos_after, -self.order_depth_internal.sell_orders[fv_int])
            if qty > 0:
                self.buy(fv_int, qty)

        self._near_fair_flatten(state, fv)

        skewed_fv = self._inventory_skewed_fair_value(state, fv)
        bid_edge, ask_edge = self._quote_edges()
        max_bid = int(skewed_fv) - int(bid_edge)
        min_ask = int(skewed_fv) + int(ask_edge)

        bid_price = max_bid
        for price in sorted(self.order_depth_internal.buy_orders, reverse=True):
            if price < max_bid:
                candidate = price + 1 if self.order_depth_internal.buy_orders[price] > 1 else price
                bid_price = min(candidate, max_bid)
                break

        ask_price = min_ask
        for price in sorted(self.order_depth_internal.sell_orders):
            if price > min_ask:
                candidate = price - 1 if -self.order_depth_internal.sell_orders[price] > 1 else price
                ask_price = max(candidate, min_ask)
                break

        remaining_buy = self.limit - state.position.get(self.symbol, 0) - self.total_buying_amount
        remaining_sell = self.limit + state.position.get(self.symbol, 0) - self.total_selling_amount
        if remaining_buy > 0:
            self.buy(bid_price, remaining_buy)
        if remaining_sell > 0:
            self.sell(ask_price, remaining_sell)

        logger.print(self.symbol, "fv", round(fv, 2), "skewed_fv", round(skewed_fv, 2), "dev", round(self._last_dev, 3), "quotes", (bid_price, ask_price), "pos", self._position_after_orders(state))

    def save(self) -> JSON:
        return {"ema_mid": self._ema_mid, "ema_samples": self._ema_samples, "last_dev": self._last_dev}

    def load(self, data) -> None:
        if isinstance(data, dict):
            ema_mid = data.get("ema_mid")
            self._ema_mid = float(ema_mid) if ema_mid is not None else None
            self._ema_samples = int(data.get("ema_samples", 0))
            self._last_dev = float(data.get("last_dev", 0.0))


class VelvetMMStrategy(Strategy):
    """
    Market-making strategy for VELVETFRUIT_EXTRACT (the underlying spot).

    Data profile (all 3 days, consistent):
      spread = 5 pts  (min=1, max=6)
      microprice signal: nonzero 29% of ticks, corr(micro_dev, ret1) = 0.17-0.19
      ACF(1) of returns = -0.15 to -0.17  (mild mean-reversion)
      best hw from fill analysis: hw=1 (6-8k/day est. at 30 units/fill)

    Logic per tick:
      1. Compute fair = mid + VELVET_MM_MICRO * (micro - mid)
                            - VELVET_MM_INV_SKEW * (pos / limit)
      2. bid_quote = floor(fair) - hw    (hw = 1, step inside bot)
         ask_quote = ceil(fair)  + hw
      3. Skip if spread < 2 (no room to improve)
      4. Inventory-skewed sizing: reduce the heavy side
    """
    HALF_WIDTH  = 1
    MICRO_COEFF = 0.55     # microprice weight (corr=0.18)
    INV_SKEW    = 4.0      # ticks of fair-price shift per unit of limit used
    BASE_SIZE   = 30       # units per side per tick
    MAX_TAKE    = 20       # aggressive take size when bot crosses our edge
    TAKE_EDGE   = 2        # take if bot price is ≥ TAKE_EDGE ticks through fair

    def __init__(self, symbol: str, limit: int, utils: MarketUtils) -> None:
        super().__init__(symbol, limit)
        self.u = utils

    def act(self, state) -> None:
        od = state.order_depths.get(self.symbol)
        if not self.u.has_liquidity(od):
            return
        best_bid = self.u.safe_best_bid(od)
        best_ask = self.u.safe_best_ask(od)
        if best_bid is None or best_ask is None:
            return
        spread = best_ask - best_bid
        if spread < 2:
            return

        mid   = self.u.book_mid(od)
        micro = self.u.microprice(od)
        if mid is None or micro is None:
            return

        position = state.position.get(self.symbol, 0)
        inv_ratio = position / self.limit

        fair = mid + self.MICRO_COEFF * (micro - mid) - self.INV_SKEW * inv_ratio

        hw = self.HALF_WIDTH
        bid_q = math.floor(fair) - hw
        ask_q = math.ceil(fair)  + hw
        if bid_q >= ask_q:
            return

        # Opportunistic take when bot crosses our fair by TAKE_EDGE
        if best_ask <= fair - self.TAKE_EDGE:
            qty = min(-self.order_depth_internal.sell_orders.get(best_ask, 0),
                      self.limit - position - self.total_buying_amount,
                      self.MAX_TAKE)
            if qty > 0:
                self.buy(best_ask, qty)

        if best_bid >= fair + self.TAKE_EDGE:
            qty = min(self.order_depth_internal.buy_orders.get(best_bid, 0),
                      self.limit + position - self.total_selling_amount,
                      self.MAX_TAKE)
            if qty > 0:
                self.sell(best_bid, qty)

        # Passive quotes — inventory-skewed size
        projected = position + self.total_buying_amount - self.total_selling_amount
        proj_ratio = projected / self.limit
        bid_size = max(4, int(round(self.BASE_SIZE * (1.0 - max(0.0,  proj_ratio)))))
        ask_size = max(4, int(round(self.BASE_SIZE * (1.0 - max(0.0, -proj_ratio)))))
        bid_size = min(bid_size, self.limit - projected)
        ask_size = min(ask_size, self.limit + projected)

        if bid_size > 0 and bid_q > 0:
            self.buy(bid_q, bid_size)
        if ask_size > 0 and ask_q > bid_q:
            self.sell(ask_q, ask_size)

        logger.print(
            self.symbol, "mm",
            "mid", round(mid, 1),
            "micro", round(micro, 1),
            "fair", round(fair, 2),
            "quotes", (bid_q, ask_q),
            "pos", projected,
        )

    # def save(self) -> JSON:
    #     return {"fair_ema": self.fair_ema, "flow_ema": self.flow_ema, "fair_samples": self.fair_samples, "last_dev": self.last_dev}

    # def load(self, data) -> None:
    #     if isinstance(data, dict):
    #         fair_ema = data.get("fair_ema")
    #         self.fair_ema = float(fair_ema) if fair_ema is not None else None
    #         self.flow_ema = float(data.get("flow_ema", 0.0))
    #         self.fair_samples = int(data.get("fair_samples", 0))
    #         self.last_dev = float(data.get("last_dev", 0.0))


class Round3VoucherStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, strike: int, utils: MarketUtils) -> None:
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.strike = strike
        self.u = utils
        self.options_model = BlackScholes()
        self.flow_ema = 0.0

    def _fair_iv(self, state: TradingState, spot: float, tte: float) -> Optional[float]:
        fair_iv = self.u.round3_target_fair_iv(state, self.strike)
        if fair_iv is not None:
            return fair_iv
        if spot <= 0 or tte <= 0:
            return None
        x = np.log(self.strike / spot) / np.sqrt(tte)
        return self.u.clamp_iv(VEV_MID_PARAMS["c"] + VEV_MID_PARAMS["b"] * x + VEV_MID_PARAMS["a"] * x * x)

    def _context(self, state: TradingState) -> Optional[Dict[str, float]]:
        order_depth = state.order_depths.get(self.symbol)
        if not self.u.has_liquidity(order_depth):
            return None

        tte, days_left, spot = self.u.round3_tte(state)
        if tte is None or tte <= 0 or spot is None or spot <= 0:
            return None

        best_bid = self.u.safe_best_bid(order_depth)
        best_ask = self.u.safe_best_ask(order_depth)
        mid_price = self.u.book_mid(order_depth)
        if best_bid is None or best_ask is None or mid_price is None:
            return None

        fair_iv = self._fair_iv(state, spot, tte)
        if fair_iv is None:
            return None

        theoretical_price = self.options_model.black_scholes_call(spot, self.strike, tte, fair_iv)
        delta = self.options_model.delta(spot, self.strike, tte, fair_iv)
        vega = self.options_model.vega(spot, self.strike, tte, fair_iv)
        recent_flow = self.u.recent_own_trade_pressure(state, self.symbol) / max(1.0, VEV_ITM_MAX_TAKE)
        self.flow_ema = (1.0 - VEV_FLOW_ALPHA) * self.flow_ema + VEV_FLOW_ALPHA * recent_flow

        return {
            "tte": tte,
            "days_left": days_left if days_left is not None else 0.0,
            "spot": spot,
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "mid_price": mid_price,
            "fair_iv": fair_iv,
            "theoretical_price": theoretical_price,
            "delta": delta,
            "vega": vega,
        }

    def _quote_position_skew(self, state: TradingState) -> float:
        projected_position = state.position.get(self.symbol, 0) + self.total_buying_amount - self.total_selling_amount
        return VEV_POSITION_SKEW * projected_position / self.limit

    def _register_delta(self, state: TradingState, delta: float) -> None:
        final_position = state.position.get(self.symbol, 0) + self.total_buying_amount - self.total_selling_amount
        Round3StateCache.option_delta_target += final_position * delta


# These strike lists are empty so these classes are never instantiated.
# Stubs kept so the Trader.__init__ loops compile without error.
class ITMOptionMeanReversionStrategy(Round3VoucherStrategy):
    def act(self, state) -> None: pass

class IVOptionArbitrageStrategy(Round3VoucherStrategy):
    def act(self, state) -> None: pass

class StructuralDirectionalOptionStrategy(Round3VoucherStrategy):
    def __init__(self, symbol, limit, strike, direction, utils):
        super().__init__(symbol, limit, strike, utils)
    def act(self, state) -> None: pass


class RegressionOptionStrategy(Round3VoucherStrategy):
    def __init__(self, symbol: str, limit: int, strike: int, utils: MarketUtils) -> None:
        super().__init__(symbol, limit, strike, utils)

    def _regression_fair(self, state: TradingState, spot_mid: float, spot_micro: float, best_bid: int, best_ask: int) -> float:
        alpha, beta = VEV_REG_COEFS[self.strike]
        spot_signal = spot_mid + VEV_REG_SPOT_MICRO_COEFF * (spot_micro - spot_mid)
        return alpha + beta * spot_signal

    def act(self, state) -> None:
        ctx = self._context(state)
        if ctx is None:
            return

        underlying_depth = state.order_depths.get(VELVETFRUIT_EXTRACT)
        spot_micro = self.u.microprice(underlying_depth)
        if spot_micro is None:
            spot_micro = ctx["spot"]

        best_bid = int(ctx["best_bid"])
        best_ask = int(ctx["best_ask"])
        position = state.position.get(self.symbol, 0)
        fair_price = self._regression_fair(state, ctx["spot"], spot_micro, best_bid, best_ask)
        fair_price += VEV_REG_FLOW_SKEW * self.flow_ema
        fair_price -= VEV_REG_INV_SKEW * (position / self.limit)

        open_edge = VEV_REG_OPEN_EDGE[self.strike]
        close_edge = VEV_REG_CLOSE_EDGE[self.strike]
        base_size = VEV_REG_BASE_SIZE[self.strike]
        max_take = VEV_REG_MAX_TAKE[self.strike]
        soft_limit = VEV_REG_SOFT_LIMIT[self.strike]
        max_make = VEV_REG_MAX_MAKE[self.strike]

        buy_edge = fair_price - best_ask
        sell_edge = best_bid - fair_price

        if position > 0 and best_bid >= fair_price - close_edge:
            amount_to_sell = min(
                self.order_depth_internal.buy_orders.get(best_bid, 0),
                position,
                max(1, base_size // 2),
            )
            if amount_to_sell > 0:
                self.sell(best_bid, amount_to_sell)
        elif position < 0 and best_ask <= fair_price + close_edge:
            amount_to_buy = min(
                -self.order_depth_internal.sell_orders.get(best_ask, 0),
                -position,
                max(1, base_size // 2),
            )
            if amount_to_buy > 0:
                self.buy(best_ask, amount_to_buy)

        if buy_edge >= open_edge and position < soft_limit:
            scale = max(0.0, buy_edge - open_edge)
            amount_to_buy = min(
                -self.order_depth_internal.sell_orders.get(best_ask, 0),
                self.limit - position - self.total_buying_amount,
                soft_limit - max(position, 0),
                min(max_take, int(base_size + 18 * scale)),
            )
            if amount_to_buy > 0:
                self.buy(best_ask, amount_to_buy)

        if sell_edge >= open_edge and -position < soft_limit:
            scale = max(0.0, sell_edge - open_edge)
            amount_to_sell = min(
                self.order_depth_internal.buy_orders.get(best_bid, 0),
                self.limit + position - self.total_selling_amount,
                soft_limit - max(-position, 0),
                min(max_take, int(base_size + 18 * scale)),
            )
            if amount_to_sell > 0:
                self.sell(best_bid, amount_to_sell)

        projected_position = position + self.total_buying_amount - self.total_selling_amount
        if best_ask - best_bid >= 2:
            bid_quote = max(best_bid + 1, min(best_ask - 1, int(math.floor(fair_price - open_edge / 2))))
            ask_quote = min(best_ask - 1, max(best_bid + 1, int(math.ceil(fair_price + open_edge / 2))))
            bid_make_size = min(max_make, soft_limit - max(projected_position, 0), self.limit - projected_position)
            ask_make_size = min(max_make, soft_limit - max(-projected_position, 0), self.limit + projected_position)
            if bid_make_size > 0 and bid_quote > 0 and bid_quote < best_ask:
                self.buy(bid_quote, bid_make_size)
            if ask_make_size > 0 and ask_quote > best_bid:
                self.sell(ask_quote, ask_make_size)

        self._register_delta(state, ctx["delta"])
        logger.print(
            self.symbol,
            "mode",
            "regress",
            "spot",
            round(ctx["spot"], 2),
            "spot_micro",
            round(spot_micro, 2),
            "fair",
            round(fair_price, 2),
            "buy_edge",
            round(buy_edge, 3),
            "sell_edge",
            round(sell_edge, 3),
            "flow",
            round(self.flow_ema, 3),
            "pos",
            state.position.get(self.symbol, 0) + self.total_buying_amount - self.total_selling_amount,
        )

    def save(self) -> JSON:
        return {"flow_ema": self.flow_ema}

    def load(self, data) -> None:
        if isinstance(data, dict):
            self.flow_ema = float(data.get("flow_ema", 0.0))


class Trader:

    def __init__(self):
        self.utils = MarketUtils()
        self.blocks = InteractionBlocks(self.utils)

        self.limit = {
            HYDROGEL_PACK: 200,
            VELVETFRUIT_EXTRACT: 200,
            **{symbol: 300 for symbol in VEV_ALL_SYMBOLS},
        }

        self.strategies = {
            HYDROGEL_PACK: HydrogelMMStrategy(
                symbol=HYDROGEL_PACK,
                limit=self.limit[HYDROGEL_PACK],
                half_width=4,
                max_trade_size=80,
                utils=self.utils,
                blocks=self.blocks,
            ),
        }
        for strike in VEV_ITM_STRIKES:
            symbol = f"VEV_{strike}"
            self.strategies[symbol] = ITMOptionMeanReversionStrategy(
                symbol=symbol,
                limit=self.limit[symbol],
                strike=strike,
                utils=self.utils,
            )
        for strike in VEV_IV_ARB_STRIKES:
            symbol = f"VEV_{strike}"
            self.strategies[symbol] = IVOptionArbitrageStrategy(
                symbol=symbol,
                limit=self.limit[symbol],
                strike=strike,
                utils=self.utils,
            )
        for strike in VEV_STRUCTURAL_LONG_STRIKES:
            symbol = f"VEV_{strike}"
            self.strategies[symbol] = StructuralDirectionalOptionStrategy(
                symbol=symbol,
                limit=self.limit[symbol],
                strike=strike,
                direction=1,
                utils=self.utils,
            )
        for strike in VEV_STRUCTURAL_SHORT_STRIKES:
            symbol = f"VEV_{strike}"
            self.strategies[symbol] = StructuralDirectionalOptionStrategy(
                symbol=symbol,
                limit=self.limit[symbol],
                strike=strike,
                direction=-1,
                utils=self.utils,
            )
        for strike in VEV_REGRESSION_STRIKES:
            symbol = f"VEV_{strike}"
            self.strategies[symbol] = RegressionOptionStrategy(
                symbol=symbol,
                limit=self.limit[symbol],
                strike=strike,
                utils=self.utils,
            )
        self.strategies[VELVETFRUIT_EXTRACT] = VelvetMMStrategy(
            symbol=VELVETFRUIT_EXTRACT,
            limit=self.limit[VELVETFRUIT_EXTRACT],
            utils=self.utils,
        )

    def run(self, state) -> Tuple[Dict[str, List], int, str]:
        logger.print(state.position)

        conversions = 0

        old_trader_data = jsonpickle.decode(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        Round3StateCache.reset_for_run()

        orders = {symbol: [] for symbol in self.limit}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                for order in strategy_orders:
                    orders.setdefault(order.symbol, []).append(order)
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        trader_data = jsonpickle.encode(new_trader_data)

        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data