# ======================================================================
# PPO + SAINTv2 — SCALPING BTCUSD M1 (SINGLE-HEAD + ACTION MASK + H1)
# Version "Loup Scalpeur" :
#   - Reward Sharpe-like sur log-return d'equity (excess_ret + Sharpe local)
#   - Position sizing dynamique via ATR_14 + risk_scale 1x / 1.8x
#   - Action masking single-head (5 actions, pas d'action CLOSE manuelle)
#   - StopLoss & TakeProfit automatiques basés sur ATR (fermeture uniquement SL/TP)
#   - ATR figé à l'entrée du trade (self.entry_atr, sans look-ahead)
#   - SAINTv2 allégé avec input_proj * sqrt(d_model)
#   - Early stopping sur Calmar rolling (fenêtre 30 epochs)
#   - Drawdown sur equity + max_dd intrépisode
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List

import MetaTrader5 as mt5
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Categorical
import wandb

# Optimisations PyTorch
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ============================================================
# SEED GLOBAL
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# CONSTANTES
# ============================================================

# 0:BUY1, 1:SELL1, 2:BUY1.8, 3:SELL1.8, 4:HOLD
N_ACTIONS = 5
MASK_VALUE = -1e4  # valeur de masquage compatible float16


# ============================================================
# CONFIG
# ============================================================

@dataclass
class PPOConfig:
    # Données
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_H1
    n_bars: int = 16180
    lookback: int = 26

    # PPO Training
    epochs: int = 618
    episodes_per_epoch: int = 3
    episode_length: int = 256
    updates_per_epoch: int = 6

    batch_size: int = 256
    gamma: float = 0.97
    lambda_gae: float = 0.95
    clip_eps: float = 0.12
    lr: float = 3e-4
    target_kl: float = 0.03
    value_coef: float = 0.5
    entropy_coef: float = 0.035      # coef de base, avec scheduler décroissant
    max_grad_norm: float = 1.0

    # SAINT
    d_model: int = 80                # plus large pour 50+ features

    # Trading
    initial_capital: float = 100.0
    position_size: float = 0.06
    leverage: float = 6.0
    fee_rate: float = 0.0004
    min_capital_frac: float = 0.2
    max_drawdown: float = 0.8

    # Risk management / position sizing
    risk_per_trade: float = 0.012    # 1.2% du capital par trade (avant risk_scale)
    max_position_frac: float = 0.35
    position_vol_penalty: float = 1e-3

    # StopLoss / TakeProfit basés sur ATR
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    # Microstructure
    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    # Scalp
    scalping_max_holding: int = 12
    scalping_holding_penalty: float = 2.5e-5
    scalping_flat_penalty: float = 5e-6
    scalping_flat_bonus: float = 5e-5

    # Curriculum vol
    use_vol_curriculum: bool = True

    # Device
    force_cpu: bool = False
    use_amp: bool = True


# ============================================================
# WANDB
# ============================================================

def init_wandb(cfg: PPOConfig):
    wandb.login()
    wandb.init(
        project="ppo_saint_m1_singlehead_scalping_pro",
        name=f"SAINTv2_SingleHead_Loup_{datetime.now().strftime('%m%d_%H%M')}",
        config=vars(cfg)
    )


# ============================================================
# INDICATEURS (M1 & H1)
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    # returns
    df["ret_1"] = c.pct_change(1)
    df["ret_3"] = c.pct_change(3)
    df["ret_5"] = c.pct_change(5)
    df["ret_15"] = c.pct_change(15)
    df["ret_60"] = c.pct_change(60)

    # realized vol
    ret = c.pct_change()
    df["realized_vol_20"] = ret.rolling(20).std()

    # Volatility regime (z-score vol réalisée long terme)
    roll_mean = df["realized_vol_20"].rolling(500).mean()
    roll_std = df["realized_vol_20"].rolling(500).std()
    df["vol_regime"] = (df["realized_vol_20"] - roll_mean) / (roll_std + 1e-8)

    # EMAs
    df["ema_5"] = c.ewm(span=5, adjust=False).mean()
    df["ema_10"] = c.ewm(span=10, adjust=False).mean()
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()

    # RSI
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - 100 / (1 + rs)

    df["rsi_7"] = rsi(c, 7)
    df["rsi_14"] = rsi(c, 14)

    # ATR(14)
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Stoch
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = (c - low14) / (high14 - low14 + 1e-8) * 100
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal

    # Ichimoku
    conv_period = 9
    base_period = 26
    span_b_period = 52

    conv_line = (h.rolling(conv_period).max() + l.rolling(conv_period).min()) / 2
    base_line = (h.rolling(base_period).max() + l.rolling(base_period).min()) / 2
    span_a = ((conv_line + base_line) / 2).shift(base_period)
    span_b = ((h.rolling(span_b_period).max() + l.rolling(span_b_period).min()) / 2).shift(base_period)

    df["ichimoku_tenkan"] = conv_line
    df["ichimoku_kijun"] = base_line
    df["ichimoku_span_a"] = span_a
    df["ichimoku_span_b"] = span_b

    df["dist_tenkan"] = (c - conv_line) / (c + 1e-8)
    df["dist_kijun"] = (c - base_line) / (c + 1e-8)
    df["dist_span_a"] = (c - span_a) / (c + 1e-8)
    df["dist_span_b"] = (c - span_b) / (c + 1e-8)

    ma_100 = c.rolling(100).mean()
    std_100 = c.rolling(100).std()
    df["ma_100"] = ma_100
    df["zscore_100"] = (c - ma_100) / (std_100 + 1e-8)

    idx = df.index
    hours = idx.hour.values
    dows = idx.dayofweek.values

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dows / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 7)

    if "tick_volume" in df.columns:
        df["tick_volume_log"] = np.log1p(df["tick_volume"])
    else:
        df["tick_volume_log"] = 0.0

    return df


# ============================================================
# CHARGEMENT M1 + H1
# ============================================================

def load_mt5_data(cfg: PPOConfig) -> pd.DataFrame:
    print("Connexion MT5…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.init()")

    rates_m1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe, 0, cfg.n_bars
    )

    n_h1 = max(cfg.n_bars // 30, 2000)
    rates_h1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.htf_timeframe, 0, n_h1
    )

    mt5.shutdown()

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1")

    # M1
    df_m1 = pd.DataFrame(rates_m1)
    df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
    df_m1.set_index("time", inplace=True)
    df_m1 = df_m1[["open", "high", "low", "close", "tick_volume"]]
    df_m1 = add_indicators(df_m1)

    # H1
    df_h1 = pd.DataFrame(rates_h1)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    df_h1.set_index("time", inplace=True)
    df_h1 = df_h1[["open", "high", "low", "close", "tick_volume"]]
    df_h1 = add_indicators(df_h1)
    df_h1 = df_h1.add_suffix("_h1")

    df_m1_reset = df_m1.reset_index()
    df_h1_reset = df_h1.reset_index()
    df_h1_reset = df_h1_reset.rename(columns={"time_h1": "time"})

    merged = pd.merge_asof(
        df_m1_reset.sort_values("time"),
        df_h1_reset.sort_values("time"),
        on="time",
        direction="backward"
    )

    merged = merged.dropna().reset_index(drop=True)
    print(f"{len(merged)} bougies M1 alignées avec H1 après indicateurs.")
    return merged

# ============================================================
# FEATURES / NORMALISATION
# ============================================================

FEATURE_COLS_M1 = [
    "open", "high", "low", "close",
    "ret_1", "ret_3", "ret_5", "ret_15", "ret_60",
    "realized_vol_20", "vol_regime",
    "ema_5", "ema_10", "ema_20",
    "rsi_7", "rsi_14",
    "atr_14",
    "stoch_k", "stoch_d",
    "macd", "macd_signal",
    "dist_tenkan", "dist_kijun", "dist_span_a", "dist_span_b",
    "ma_100", "zscore_100",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "tick_volume_log",
]

FEATURE_COLS_H1 = [
    "close_h1",
    "ema_20_h1",
    "rsi_14_h1",
    "macd_h1",
    "macd_signal_h1",
    "zscore_100_h1",
    "dist_tenkan_h1",
    "dist_kijun_h1",
    "dist_span_a_h1",
    "dist_span_b_h1",
    "realized_vol_20_h1",
]

FEATURE_COLS = FEATURE_COLS_M1 + FEATURE_COLS_H1

N_BASE_FEATURES = len(FEATURE_COLS)
OBS_N_FEATURES = N_BASE_FEATURES + 1 + 1 + 3  # +unreal + last_real + pos_onehot(3)

NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"


class MarketData:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 stats: Optional[Dict[str, np.ndarray]] = None):
        X = df[feature_cols].values.astype(np.float32)

        if stats is not None:
            mean, std = stats["mean"], stats["std"]
            std = np.where(std < 1e-8, 1.0, std)
            X = (X - mean) / std

        self.features = X
        self.close = df["close"].values.astype(np.float32)
        self.length = len(df)

        # Données brutes utiles pour risk management / SL/TP
        self.atr14 = df["atr_14"].values.astype(np.float32) if "atr_14" in df.columns else np.zeros(len(df), np.float32)
        self.ema20_h1 = df["ema_20_h1"].values.astype(np.float32) if "ema_20_h1" in df.columns else np.zeros(len(df), np.float32)
        self.high = df["high"].values.astype(np.float32) if "high" in df.columns else np.zeros(len(df), np.float32)
        self.low = df["low"].values.astype(np.float32) if "low" in df.columns else np.zeros(len(df), np.float32)

    def __len__(self):
        return self.length


def create_datasets(df: pd.DataFrame, feature_cols: List[str]):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_train = df[:train_end].reset_index(drop=True)
    df_val = df[train_end:val_end].reset_index(drop=True)
    df_test = df[val_end:].reset_index(drop=True)

    Xtrain = df_train[feature_cols].values.astype(np.float32)
    mean, std = Xtrain.mean(0), Xtrain.std(0)
    stats = {"mean": mean, "std": std}

    train_data = MarketData(df_train, feature_cols, stats)
    val_data = MarketData(df_val, feature_cols, stats)
    test_data = MarketData(df_test, feature_cols, stats)

    print(f"SPLIT : train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return train_data, val_data, test_data, stats


# ======================================================================
# ENVIRONNEMENT
# ======================================================================

class BTCTradingEnvDiscrete(gym.Env):
    """
    Actions internes env :
        0 = BUY
        1 = SELL
        2 = HOLD

    Actions agent (single-head + mask, 5 actions) :
        0 = BUY  (risk 1x)
        1 = SELL (risk 1x)
        2 = BUY  (risk 1.8x)
        3 = SELL (risk 1.8x)
        4 = HOLD

      - si position == 0 :
            actions valides = {0,1,2,3,4}
      - si position != 0 :
            actions valides = {4} uniquement, la fermeture se fait
            UNIQUEMENT par SL/TP ATR (pas d'action CLOSE manuelle).

    Reward :
        - log-return d'equity (log(E_t / E_{t-1}) - expected_ret)
        - Sharpe local sur les 5 derniers trades
        - pénalité overholding + bonus flat quand le marché bouge
    """

    metadata = {"render_modes": []}

    def __init__(self, data: MarketData, cfg: PPOConfig):
        super().__init__()
        self.data = data
        self.cfg = cfg
        self.lookback = cfg.lookback

        # 0=BUY,1=SELL,2=HOLD
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, OBS_N_FEATURES),
            dtype=np.float32
        )

        if self.cfg.use_vol_curriculum:
            self._init_vol_curriculum()
        else:
            self.low_vol_starts = None
            self.high_vol_starts = None

        self.risk_scale = 1.0
        self.reset()

    # ---------- Curriculum vol ----------

    def _init_vol_curriculum(self):
        close = self.data.close
        ret = np.diff(close) / (close[:-1] + 1e-8)
        vol20 = pd.Series(ret).rolling(20).std().to_numpy()
        vol20 = np.concatenate([[np.nan], vol20])

        valid = ~np.isnan(vol20)
        if valid.sum() < 30:
            self.low_vol_starts = None
            self.high_vol_starts = None
            return

        q_low, q_high = np.quantile(vol20[valid], [0.3, 0.7])

        candidate_low = np.where((vol20 <= q_low) & valid)[0]
        candidate_high = np.where((vol20 >= q_high) & valid)[0]

        max_start = self.data.length - self.cfg.episode_length - 2
        low = candidate_low[
            (candidate_low >= self.lookback) &
            (candidate_low <= max_start)
        ]
        high = candidate_high[
            (candidate_high >= self.lookback) &
            (candidate_high <= max_start)
        ]

        self.low_vol_starts = low if len(low) > 0 else None
        self.high_vol_starts = high if len(high) > 0 else None

    def set_risk_scale(self, scale: float):
        self.risk_scale = float(max(scale, 0.0))

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self.data.length - self.cfg.episode_length - 2

        start_idx = None
        if self.cfg.use_vol_curriculum and self.low_vol_starts is not None and self.high_vol_starts is not None:
            if np.random.rand() < 0.5 and len(self.low_vol_starts) > 0:
                start_idx = int(np.random.choice(self.low_vol_starts))
            elif len(self.high_vol_starts) > 0:
                start_idx = int(np.random.choice(self.high_vol_starts))

        if start_idx is None:
            start_idx = np.random.randint(self.lookback, max_start)

        self.start_idx = start_idx
        self.end_idx = self.start_idx + self.cfg.episode_length
        self.idx = self.start_idx

        self.capital = self.cfg.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.current_size = 0.0

        self.sl_price = 0.0
        self.tp_price = 0.0
        self.entry_idx = -1
        self.entry_atr = 0.0

        self.last_realized_pnl = 0.0
        self.peak_capital = self.capital
        self.trades_pnl: List[float] = []

        self.bars_in_position = 0
        self.risk_scale = 1.0

        # max drawdown intrépisode
        self.max_dd = 0.0

        obs = self._get_obs()
        return obs, {
            "capital": self.capital,
            "position": self.position,
            "drawdown": 0.0,
            "done_reason": None
        }

    def _get_obs(self):
        start = self.idx - self.lookback
        feat = self.data.features[start:self.idx]
        last_price = self.data.close[self.idx - 1]

        if self.position != 0:
            unreal = (
                self.position *
                (last_price - self.entry_price) *
                self.current_size *
                self.cfg.leverage
            )
        else:
            unreal = 0.0

        unreal_norm = np.full(
            (self.lookback, 1),
            unreal / self.cfg.initial_capital,
            dtype=np.float32
        )

        last_real_norm = np.full(
            (self.lookback, 1),
            self.last_realized_pnl / self.cfg.initial_capital,
            dtype=np.float32
        )

        pos = np.zeros((self.lookback, 3), np.float32)
        if self.position == -1:
            pos[:, 0] = 1.0
        elif self.position == 0:
            pos[:, 1] = 1.0
        else:
            pos[:, 2] = 1.0

        obs = np.concatenate(
            [feat, unreal_norm, last_real_norm, pos],
            axis=1
        )
        return obs.astype(np.float32)

    def _apply_micro(self, price: float, side: int) -> float:
        spread = self.cfg.spread_bps / 10_000.0
        slip = self.cfg.slippage_bps / 10_000.0

        if side != 0:
            price *= (1 + side * spread * 0.5)
            price *= (1 + np.random.uniform(-slip, slip))

        return price

    def _compute_dynamic_size(self, price: float) -> float:
        """
        Position sizing aligné avec le SL ATR :
          - on prend la distance de SL = atr_sl_mult * ATR_effectif
          - risk_per_trade correspond à la perte max si SL touché.
          - risk_scale (1x / 1.8x) modifie effectivement le risque.
        """
        if self.capital <= 0:
            return 0.0

        atr = float(self.data.atr14[self.idx - 1]) if self.idx - 1 >= 0 else 0.0
        fallback = 0.0015 * price
        eff_atr = max(atr, fallback, 1e-8)

        stop_distance = self.cfg.atr_sl_mult * eff_atr

        risk_base = self.cfg.risk_per_trade
        effective_risk = risk_base * max(self.risk_scale, 1e-6)

        size = effective_risk * self.capital / (stop_distance * self.cfg.leverage + 1e-8)
        size = size / max(price, 1e-8)

        max_notional = self.cfg.max_position_frac * self.capital
        max_size = max_notional / max(price, 1e-8)

        size = float(np.clip(size, 1e-4, max_size))
        return size

    def step(self, action: int):
        """
        action (env) :
          0 = BUY, 1 = SELL, 2 = HOLD
        """
        price = self.data.close[self.idx]
        high_bar = self.data.high[self.idx]
        low_bar = self.data.low[self.idx]

        old_pos = self.position
        prev_capital = self.capital

        # equity à t-1 (avant action)
        if self.idx > 0:
            prev_price = self.data.close[self.idx - 1]
        else:
            prev_price = price

        if old_pos != 0 and self.current_size > 0 and self.entry_price > 0:
            prev_latent = (
                old_pos *
                (prev_price - self.entry_price) *
                self.current_size *
                self.cfg.leverage
            )
        else:
            prev_latent = 0.0

        prev_equity = prev_capital + prev_latent

        realized = 0.0
        realized_trade = 0.0

        # --------- Mise à jour du temps en position (AVANT fermetures) ---------
        if old_pos != 0:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0

        # --------- exécution action courante ---------

        # Ouverture (uniquement si flat)
        if action in (0, 1) and old_pos == 0:
            side = 1 if action == 0 else -1

            if side != 0:
                size = self._compute_dynamic_size(price)
                if size > 0.0:
                    self.current_size = size
                    self.position = side

                    exec_price = self._apply_micro(price, side)
                    self.entry_price = exec_price
                    self.entry_idx = self.idx

                    # ATR figé à l'entrée pour SL/TP — sans look-ahead
                    atr_raw = float(self.data.atr14[self.idx - 1]) if self.idx - 1 >= 0 else 0.0
                    fallback = 0.0015 * exec_price
                    self.entry_atr = max(atr_raw, fallback, 1e-8)

                    sl_dist = self.cfg.atr_sl_mult * self.entry_atr
                    tp_dist = self.cfg.atr_tp_mult * self.entry_atr

                    if side == 1:  # long
                        self.sl_price = max(1e-8, exec_price - sl_dist)
                        self.tp_price = max(1e-8, exec_price + tp_dist)
                    else:          # short
                        self.sl_price = max(1e-8, exec_price + sl_dist)
                        self.tp_price = max(1e-8, exec_price - tp_dist)

                    fee = self.cfg.fee_rate * exec_price * size
                    self.capital -= fee
                else:
                    self.position = 0
                    self.current_size = 0.0
                    self.entry_price = 0.0
                    self.sl_price = 0.0
                    self.tp_price = 0.0
                    self.entry_idx = -1
                    self.entry_atr = 0.0

        # Sinon HOLD : on ne touche pas à la position

        # --------- FERMETURE AUTOMATIQUE SL/TP (autorisé dès la bougie suivante) ---------
        hit_sl = False
        hit_tp = False

        if (
            self.position != 0 and
            self.current_size > 0 and
            self.entry_price > 0 and
            self.entry_idx >= 0 and
            self.idx > self.entry_idx   # dès la bougie suivante
        ):
            exit_price = None

            if self.position == 1:  # long
                if self.sl_price > 0 and low_bar <= self.sl_price:
                    exit_price = self.sl_price
                    hit_sl = True
                elif self.tp_price > 0 and high_bar >= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True

            elif self.position == -1:  # short
                if self.sl_price > 0 and high_bar >= self.sl_price:
                    exit_price = self.sl_price
                    hit_sl = True
                elif self.tp_price > 0 and low_bar <= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True

            if exit_price is not None:
                exec_price = self._apply_micro(exit_price, -self.position)

                pnl = (
                    self.position *
                    (exec_price - self.entry_price) *
                    self.current_size *
                    self.cfg.leverage
                )
                fee = self.cfg.fee_rate * exec_price * self.current_size
                realized = pnl - fee
                realized_trade = realized

                self.capital += realized
                self.last_realized_pnl = realized
                self.trades_pnl.append(realized)

                self.position = 0
                self.current_size = 0.0
                self.entry_price = 0.0
                self.sl_price = 0.0
                self.tp_price = 0.0
                self.entry_idx = -1
                self.entry_atr = 0.0
                # bars_in_position sera remis à 0 au prochain step

        # --------- EQUITY & REWARD (Sharpe-like sur log-return) ---------

        latent = 0.0
        if self.position != 0 and self.current_size > 0 and self.entry_price > 0:
            latent = (
                self.position *
                (price - self.entry_price) *
                self.current_size *
                self.cfg.leverage
            )

        equity = self.capital + latent
        equity_clamped = max(equity, 1e-8)
        prev_equity_clamped = max(prev_equity, 1e-8)

        log_ret = math.log(equity_clamped / prev_equity_clamped)

        # Coût de portage / opportunité par barre (4% annuel approximatif)
        risk_free_rate_per_bar = 0.04 / (365.0 * 1440.0)
        expected_ret = (
            risk_free_rate_per_bar * self.cfg.leverage * abs(self.position)
            + 5e-6  # petite constante pour forcer la rentabilité
        )

        excess_ret = log_ret - expected_ret
        reward = excess_ret

        # Sharpe local sur les 5 derniers trades (en PnL / capital)
        if len(self.trades_pnl) > 5:
            recent = np.array(self.trades_pnl[-5:], dtype=np.float32) / self.cfg.initial_capital
            sharpe_local = recent.mean() / (recent.std() + 1e-8)
            reward += 1e-4 * float(sharpe_local)

        # Pénalité si trop longtemps en position (anti-overholding)
        if self.bars_in_position > self.cfg.scalping_max_holding:
            reward -= self.cfg.scalping_holding_penalty * (
                self.bars_in_position - self.cfg.scalping_max_holding
            )

        # Bonus si flat + marché bouge un minimum
        if self.position == 0 and abs(log_ret) > 3e-4:
            reward += self.cfg.scalping_flat_bonus

        # Bonus/malus sur TP / SL (ratio gain/risque)
        if hit_tp:
            reward += 0.005
        elif hit_sl:
            reward -= 0.002

        # Clip plus léger pour ne pas tuer le signal
        reward = float(np.clip(reward, -0.02, 0.02))

        # --------- DRAWDOWN SUR EQUITY + MAX_DD ---------

        self.peak_capital = max(self.peak_capital, equity)
        dd = (self.peak_capital - equity) / (self.peak_capital + 1e-8)
        self.max_dd = max(self.max_dd, dd)

        if dd > self.cfg.max_drawdown:
            reward -= 1e-3 * (dd - self.cfg.max_drawdown)

        # Terminaison
        self.idx += 1
        done = False
        done_reason = None

        if self.idx >= self.end_idx:
            done = True
            done_reason = "episode_end"

        if dd > self.cfg.max_drawdown:
            done = True
            done_reason = "max_drawdown"

        if self.capital <= self.cfg.initial_capital * self.cfg.min_capital_frac:
            done = True
            done_reason = "min_capital"

        obs = self._get_obs()

        return obs, float(reward), done, False, {
            "capital": self.capital,
            "drawdown": self.max_dd,  # max DD intrépisode
            "done_reason": done_reason,
            "position": self.position
        }

# ======================================================================
# SAINT v2 — SINGLE-HEAD (ACTOR + CRITIC) ALLÉGÉ
# ======================================================================

class GatedFFN(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.05):
        super().__init__()
        inner = d * mult
        self.lin1 = nn.Linear(d, inner * 2)
        self.lin2 = nn.Linear(inner, d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, gate = self.lin1(h).chunk(2, dim=-1)
        h = a * torch.sigmoid(gate)
        h = self.lin2(self.dropout(h))
        return x + h


class ColumnAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.reshape(B * T, F, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        return h.reshape(B, T, F, D)


class RowAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        h = h.reshape(B, F, T, D).permute(0, 2, 1, 3)
        return h


class SAINTv2Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, mult: int):
        super().__init__()
        self.ra1 = RowAttention(d, heads, dropout)
        self.ff1 = GatedFFN(d, mult, dropout)

        self.ra2 = RowAttention(d, heads, dropout)
        self.ff2 = GatedFFN(d, mult, dropout)

        self.ca = ColumnAttention(d, heads, dropout)
        self.ff3 = GatedFFN(d, mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ra1(x)
        x = self.ff1(x)
        x = self.ra2(x)
        x = self.ff2(x)
        x = self.ca(x)
        x = self.ff3(x)
        return x


class SAINTPolicySingleHead(nn.Module):
    """
    SAINTv2 simplifié :
      - actor: logits (N_ACTIONS)
      - critic: V(s)
    """

    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 96,
        num_blocks: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
        ff_mult: int = 2,
        max_len: int = 512,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_actions = n_actions

        self.input_proj = nn.Linear(1, d_model)
        self.scale = math.sqrt(d_model)
        self.row_emb = nn.Embedding(max_len, d_model)
        self.col_emb = nn.Embedding(n_features, d_model)

        self.blocks = nn.ModuleList([
            SAINTv2Block(d_model, heads, dropout, ff_mult)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        x : (B,T,F)
        """
        assert x.dim() == 3, f"Input x must be (B,T,F), got {x.shape}"
        B, T, F = x.shape

        tok = self.input_proj(x.unsqueeze(-1)) * self.scale  # (B,T,F,D)

        rows = torch.arange(T, device=x.device).view(1, T, 1).expand(B, T, F)
        cols = torch.arange(F, device=x.device).view(1, 1, F).expand(B, T, F)

        tok = tok + self.row_emb(rows) + self.col_emb(cols)

        for blk in self.blocks:
            tok = blk(tok)

        # Agrégation améliorée : double pooling temps / features
        h_time = tok.mean(dim=1)      # (B, F, D) - moyenne sur T
        h_feat = tok.mean(dim=2)      # (B, T, D) - moyenne sur F

        cls_time = h_time.mean(dim=1)  # (B, D)
        cls_feat = h_feat.mean(dim=1)  # (B, D)

        h = cls_time + cls_feat        # fusion
        h = self.norm(h)
        h = self.mlp(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value

# ======================================================================
# PPO UTILITAIRES
# ======================================================================

def get_device(cfg: PPOConfig):
    if cfg.force_cpu:
        print("CPU forcé.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA détecté — utilisation GPU.")
        return torch.device("cuda")
    print("Pas de CUDA — utilisation CPU.")
    return torch.device("cpu")


def compute_gae(rewards, values, dones, gamma, lam, last_value=0.0):
    values = values + [last_value]
    gae = 0.0
    adv = []

    for t in reversed(range(len(rewards))):
        mask = 1 - int(dones[t])
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv.insert(0, gae)

    returns = [adv[i] + values[i] for i in range(len(adv))]
    return adv, returns


def epsilon_greedy_from_logits(logits_masked: torch.Tensor, eps: float) -> int:
    """
    logits_masked : logits déjà masqués (MASK_VALUE sur actions invalides).
    Exploration epsilon-greedy *respectant* les masques.
    """
    with torch.no_grad():
        if np.random.rand() > eps:
            return logits_masked.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits_masked, dim=-1)
            dist = Categorical(probs)
            return dist.sample().item()


# --------- Masques d'actions ---------

def build_mask_from_pos_scalar(pos: int, device) -> torch.Tensor:
    """
    Renvoie un bool mask (N_ACTIONS,) True = action valide.
    - Flat  : toutes les actions dispo (open + HOLD)
    - En pos: seulement HOLD (index 4)
    """
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)
    if pos == 0:
        mask[:] = True
    else:
        mask[4] = True
    return mask


def build_action_mask_from_states(states: torch.Tensor) -> torch.Tensor:
    """
    states : (B, T, OBS_N_FEATURES)
    On lit la dernière ligne (T-1) sur les 3 derniers features (one-hot pos).
    Renvoie mask (B, N_ACTIONS) bool, vectorisé.
    """
    B, T, F = states.shape
    device = states.device

    pos_oh = states[:, -1, -3:]  # (B,3)
    pos_id = pos_oh.argmax(dim=-1)  # (B,)

    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool, device=device)

    flat = (pos_id == 1)
    inpos = ~flat

    if flat.any():
        mask[flat] = True  # toutes les actions disponibles

    if inpos.any():
        mask[inpos, 4] = True  # seulement HOLD

    return mask


# ======================================================================
# TRAINING PPO
# ======================================================================

def run_training(cfg: PPOConfig):
    df = load_mt5_data(cfg)
    train_data, val_data, test_data, stats = create_datasets(df, FEATURE_COLS)

    np.savez(NORM_STATS_PATH, mean=stats["mean"], std=stats["std"])
    print(f"Stats de normalisation sauvegardées → {NORM_STATS_PATH}")

    init_wandb(cfg)
    device = get_device(cfg)

    env = BTCTradingEnvDiscrete(train_data, cfg)
    val_env = BTCTradingEnvDiscrete(val_data, cfg)

    policy = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=cfg.d_model,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-8)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: max(0.05, 1 - e / cfg.epochs)
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(cfg.use_amp and device.type == "cuda")
    )

    if os.path.exists("best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth"):
        print("→ Chargement du modèle existant pour continuation…")
        policy.load_state_dict(torch.load("best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth", map_location=device))

    best_val_profit = -1e9
    best_calmar = -1e9
    best_state = None
    epochs_no_improve = 0
    patience = 150

    calmar_history: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        batch_states = []
        batch_actions = []
        batch_oldlog = []
        batch_adv = []
        batch_returns = []
        batch_values = []

        total_reward_epoch = 0.0
        epoch_pnl = []
        epoch_dd = []
        epoch_trades_pnl: List[float] = []

        action_counts_env = np.zeros(3, dtype=np.int64)  # BUY, SELL, HOLD

        policy.train()

        # --------- collecte expériences ---------
        for ep in range(cfg.episodes_per_epoch):
            state, info = env.reset()
            done = False

            ep_states = []
            ep_actions = []
            ep_logprobs = []
            ep_rewards = []      # rewards bruts (log-returns Sharpe-like)
            ep_values = []
            ep_dones = []

            while not done:
                pos = info.get("position", 0)
                s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, value = policy(s_tensor)  # logits : (1, N_ACTIONS)
                    logits = logits[0]

                    mask = build_mask_from_pos_scalar(pos, device)
                    logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                    dist = Categorical(logits=logits_masked)
                    agent_action = dist.sample()
                    logprob = dist.log_prob(agent_action).squeeze()
                    a = int(agent_action.item())

                    # -------- mapping agent -> env + risk_scale --------
                    if pos == 0:  # flat → toutes actions possibles
                        if a == 4:      # HOLD explicite
                            env_action = 2
                            env.set_risk_scale(1.0)
                        elif a in (0, 2):   # BUY
                            env_action = 0
                            risk_scale = 1.8 if a == 2 else 1.0
                            env.set_risk_scale(risk_scale)
                        elif a in (1, 3):   # SELL
                            env_action = 1
                            risk_scale = 1.8 if a == 3 else 1.0
                            env.set_risk_scale(risk_scale)
                        else:              # sécurité
                            env_action = 2
                            env.set_risk_scale(1.0)
                    else:  # déjà en position → HOLD forcé, on ne touche pas au risk_scale
                        env_action = 2

                action_counts_env[env_action] += 1

                # step env
                ns, reward, done, _, info = env.step(env_action)
                total_reward_epoch += reward

                ep_rewards.append(reward)

                # On stocke l'état PRE-step
                ep_states.append(state)  # (lookback, OBS_N_FEATURES)

                ep_actions.append(a)
                ep_logprobs.append(logprob.detach())
                ep_values.append(value.item())
                ep_dones.append(done)

                state = ns

            epoch_dd.append(info["drawdown"])
            epoch_pnl.append(env.capital - cfg.initial_capital)
            epoch_trades_pnl.extend(env.trades_pnl)

            # valeur bootstrap
            if done and info.get("done_reason") == "max_drawdown":
                last_value = 0.0
            else:
                with torch.no_grad():
                    s_last = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    _, v_last = policy(s_last)
                    last_value = v_last.item()

            # pas de normalisation par épisode → les rewards sont déjà des log-returns
            adv, ret = compute_gae(
                ep_rewards, ep_values, ep_dones,
                cfg.gamma, cfg.lambda_gae, last_value
            )

            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_oldlog.extend(ep_logprobs)
            batch_adv.extend(adv)
            batch_returns.extend(ret)
            batch_values.extend(ep_values)

        # --------- tenseurs batch ---------
        states_np = np.stack(batch_states, axis=0)  # (N_steps, lookback, OBS_N_FEATURES)
        states = torch.tensor(states_np, dtype=torch.float32, device=device)

        actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
        oldlog = torch.stack(batch_oldlog).to(device).view(-1)
        advantages = torch.tensor(batch_adv, dtype=torch.float32, device=device)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        values_old = torch.tensor(batch_values, dtype=torch.float32, device=device)

        assert states.size(0) == oldlog.size(0) == actions.size(0) == values_old.size(0), \
            f"Len mismatch: states={states.size(0)}, oldlog={oldlog.size(0)}, actions={actions.size(0)}, values={values_old.size(0)}"

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-10, 10)
        advantages = advantages * 1.5  # boost modéré du signal sur log-returns

        # --------- PPO update ---------
        epoch_actor_loss = []
        epoch_critic_loss = []
        epoch_entropy = []
        epoch_kl = []

        n_samples = states.size(0)
        idx = np.arange(n_samples)

        for upd in range(cfg.updates_per_epoch):
            np.random.shuffle(idx)

            for start in range(0, n_samples, cfg.batch_size):
                end = start + cfg.batch_size
                ids = idx[start:end]

                sb = states[ids]
                ab = actions[ids]
                lb_old = oldlog[ids]
                adv_b = advantages[ids]
                ret_b = returns[ids]
                val_old = values_old[ids]

                with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                    logits, value = policy(sb)
                    mask_batch = build_action_mask_from_states(sb)
                    logits_masked = logits.masked_fill(~mask_batch, MASK_VALUE)

                    dist = Categorical(logits=logits_masked)
                    new_log = dist.log_prob(ab)
                    entropy = dist.entropy().mean()

                    ratio = (new_log - lb_old).exp()
                    surr1 = adv_b * ratio
                    surr2 = adv_b * torch.clamp(
                        ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value loss PPO-style avec clipping (MSE)
                    value_pred = value.squeeze(-1)
                    v_clipped = val_old + (value_pred - val_old).clamp(-0.2, 0.2)
                    unclipped_loss = (value_pred - ret_b).pow(2)
                    clipped_loss = (v_clipped - ret_b).pow(2)
                    critic_loss = torch.max(unclipped_loss, clipped_loss).mean()

                    # Entropy scheduler DÉCROISSANT
                    entropy_coef_epoch = cfg.entropy_coef * (0.1 + 0.9 * math.exp(-epoch / 100.0))
                    entropy_bonus = entropy_coef_epoch * entropy

                    loss = (
                        actor_loss +
                        cfg.value_coef * critic_loss -
                        entropy_bonus
                    )

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    approx_kl = (lb_old - new_log).mean().item()

                epoch_actor_loss.append(actor_loss.item())
                epoch_critic_loss.append(critic_loss.item())
                epoch_entropy.append(entropy.item())
                epoch_kl.append(approx_kl)

            if np.mean(epoch_kl) > 1.5 * cfg.target_kl:
                print(f"[PPO] Early stop KL, KL={np.mean(epoch_kl):.4f}")
                break

        scheduler.step()

        # --------- stats train ---------
        profit_epoch = float(sum(epoch_pnl))
        num_trades_epoch = len(epoch_trades_pnl)
        winrate_epoch = (
            float(np.mean([p > 0 for p in epoch_trades_pnl]))
            if num_trades_epoch > 0 else 0.0
        )
        max_dd_epoch = float(max(epoch_dd) if epoch_dd else 0.0)

        total_actions_env = int(action_counts_env.sum()) if action_counts_env.sum() > 0 else 1
        buy_count, sell_count, hold_count = action_counts_env
        buy_ratio = buy_count / total_actions_env
        sell_ratio = sell_count / total_actions_env
        hold_ratio = hold_count / total_actions_env

        # --------- validation (epsilon-greedy + mask) ---------
        policy.eval()
        val_pnl = []
        val_dd = []
        val_trades = []

        with torch.no_grad():
            for _ in range(2):
                s, info = val_env.reset()
                done = False
                step_count = 0
                while not done:
                    pos = info.get("position", 0)
                    st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                    logits, _ = policy(st)
                    logits = logits[0]
                    mask = build_mask_from_pos_scalar(pos, device)
                    logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                    eps = 0.05 if step_count < 1000 else 0.0
                    a = epsilon_greedy_from_logits(logits_masked.unsqueeze(0), eps=eps)

                    if pos == 0:
                        if a == 4:        # HOLD explicite
                            env_action = 2
                            val_env.set_risk_scale(1.0)
                        elif a in (0, 2):  # BUY
                            env_action = 0
                            risk_scale = 1.8 if a == 2 else 1.0
                            val_env.set_risk_scale(risk_scale)
                        elif a in (1, 3):  # SELL
                            env_action = 1
                            risk_scale = 1.8 if a == 3 else 1.0
                            val_env.set_risk_scale(risk_scale)
                        else:
                            env_action = 2
                            val_env.set_risk_scale(1.0)
                    else:
                        env_action = 2   # HOLD uniquement (ne touche pas au risk_scale)

                    ns, r, done, _, info = val_env.step(env_action)
                    s = ns
                    step_count += 1

                val_pnl.append(val_env.capital - cfg.initial_capital)
                val_dd.append(info["drawdown"])
                val_trades.extend(val_env.trades_pnl)

        val_profit = float(sum(val_pnl))
        val_max_dd = float(max(val_dd) if val_dd else 0.0)
        val_num_trades = len(val_trades)
        val_winrate = (
            float(np.mean([t > 0 for t in val_trades]))
            if val_num_trades > 0 else 0.0
        )
        calmar = val_profit / (val_max_dd + 0.05)

        calmar_history.append(calmar)
        if len(calmar_history) >= 30:
            recent_calmar = float(np.mean(calmar_history[-30:]))
        else:
            recent_calmar = float(np.mean(calmar_history))

        # --------- logging (tous les 5 epochs) ---------
        if epoch % 5 == 0:
            wandb.log({
                "epoch": epoch,

                "train/profit": profit_epoch,
                "train/num_trades": num_trades_epoch,
                "train/winrate": winrate_epoch,
                "train/max_dd": max_dd_epoch,
                "train/entropy": float(np.mean(epoch_entropy)),
                "train/kl": float(np.mean(epoch_kl)),
                "train/actor_loss": float(np.mean(epoch_actor_loss)),
                "train/critic_loss": float(np.mean(epoch_critic_loss)),
                "train/total_reward": total_reward_epoch,

                "train/env_actions/buy_count": int(buy_count),
                "train/env_actions/sell_count": int(sell_count),
                "train/env_actions/hold_count": int(hold_count),
                "train/env_actions/buy_ratio": buy_ratio,
                "train/env_actions/sell_ratio": sell_ratio,
                "train/env_actions/hold_ratio": hold_ratio,

                "val/profit": val_profit,
                "val/num_trades": val_num_trades,
                "val/winrate": val_winrate,
                "val/max_dd": val_max_dd,
                "val/calmar": calmar,
                "val/calmar_recent30": recent_calmar,

                "lr": scheduler.get_last_lr()[0],
            })

        print(
            f"[EPOCH {epoch:03d}] "
            f"TrainPNL={profit_epoch:8.2f}  "
            f"Trades={num_trades_epoch:4d}  "
            f"Win={winrate_epoch:5.1%}  "
            f"DD={max_dd_epoch:5.1%}  "
            f"ValPNL={val_profit:8.2f}  "
            f"ValTrades={val_num_trades:4d}  "
            f"ValWin={val_winrate:5.1%}  "
            f"ValDD={val_max_dd:5.1%}  "
            f"Calmar={calmar:6.3f}  "
            f"Calmar30={recent_calmar:6.3f}  "
            f"ENV B:{buy_ratio:4.1%} S:{sell_ratio:4.1%} H:{hold_ratio:4.1%}  "
            f"KL={np.mean(epoch_kl):.4f}"
        )

        # --------- early stopping sur Calmar rolling ---------
        if recent_calmar > best_calmar:
            best_calmar = recent_calmar
            best_val_profit = val_profit
            best_state = policy.state_dict().copy()
            torch.save(best_state, "best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth")
            wandb.save("best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth")
            epochs_no_improve = 0
            print(f"[EPOCH {epoch:03d}] Nouveau best model (Calmar30={recent_calmar:.3f}).")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping après {epoch} epochs (Calmar rolling ne progresse plus).")
                break

    # --------- sauvegarde finale ---------
    torch.save(policy.state_dict(), "last_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth")
    wandb.save("last_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth")

    print("Entraînement terminé, passage en TEST (epsilon-greedy + mask)…")

    # ================= TEST =================
    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()

    test_env = BTCTradingEnvDiscrete(test_data, cfg)
    all_trades = []
    all_dd = []
    all_equity = []

    with torch.no_grad():
        for ep in range(5):
            s, info = test_env.reset()
            done = False
            step_count = 0
            while not done:
                pos = info.get("position", 0)
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = policy(st)
                logits = logits[0]
                mask = build_mask_from_pos_scalar(pos, device)
                logits_masked = logits.masked_fill(~mask, MASK_VALUE)
                eps = 0.05 if step_count < 1000 else 0.0
                a = epsilon_greedy_from_logits(logits_masked.unsqueeze(0), eps=eps)

                if pos == 0:
                    if a == 4:        # HOLD explicite
                        env_action = 2
                        test_env.set_risk_scale(1.0)
                    elif a in (0, 2):  # BUY
                        env_action = 0
                        risk_scale = 1.8 if a == 2 else 1.0
                        test_env.set_risk_scale(risk_scale)
                    elif a in (1, 3):  # SELL
                        env_action = 1
                        risk_scale = 1.8 if a == 3 else 1.0
                        test_env.set_risk_scale(risk_scale)
                    else:
                        env_action = 2
                        test_env.set_risk_scale(1.0)
                else:
                    env_action = 2   # HOLD uniquement

                ns, r, done, _, info = test_env.step(env_action)
                s = ns
                step_count += 1

            dd_ep = info["drawdown"]
            all_dd.append(dd_ep)
            all_trades.extend(test_env.trades_pnl)
            all_equity.append(test_env.capital - cfg.initial_capital)

    test_profit = float(sum(all_equity))
    test_num_trades = len(all_trades)
    test_winrate = (
        float(np.mean([p > 0 for p in all_trades]))
        if test_num_trades > 0 else 0.0
    )
    test_max_dd = float(max(all_dd) if all_dd else 0.0)

    print(
        f"[TEST] Profit={test_profit:.2f} $, "
        f"trades={test_num_trades}, "
        f"winrate={test_winrate:2.0%}, "
        f"max_dd={test_max_dd:.3f}"
    )

    wandb.log({
        "test/profit": test_profit,
        "test/num_trades": test_num_trades,
        "test/winrate": test_winrate,
        "test/max_dd": test_max_dd,
    })

    wandb.finish()
    print("Fin du script.")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    cfg = PPOConfig()
    run_training(cfg)
