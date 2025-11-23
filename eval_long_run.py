# ======================================================================
# EVAL LONGUE DURÉE — SAINTv2 SINGLE-HEAD SCALPING (BTCUSD M1, M1+H1)
#
# - Même modèle que le training "Loup Scalpeur"
# - Même features : M1 + H1 (OHLC + indicateurs + Ichimoku + temps + vol_regime)
# - Même normalisation : norm_stats_ohlc_indics.npz
# - Même action space : 5 actions (0..4) + masking
#   0 = BUY  1x
#   1 = SELL 1x
#   2 = BUY  1.8x
#   3 = SELL 1.8x
#   4 = HOLD
# - Backtest offline continu sur historique MT5.
#   SL/TP uniquement via ATR, pas d’action CLOSE manuelle.
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

N_ACTIONS = 5
MASK_VALUE = -1e4  # même valeur que le training (compatible float16)


@dataclass
class EvalConfig:
    symbol: str = "BTCUSD"
    timeframe_m1: int = mt5.TIMEFRAME_M1
    timeframe_h1: int = mt5.TIMEFRAME_H1

    # Nombre de bougies d’historique pour le backtest
    n_bars_m1: int = 250_000
    n_bars_h1: int = 25_000

    lookback: int = 26

    # Hypothèses RL identiques au training
    initial_capital: float = 100.0
    leverage: float = 6.0
    fee_rate: float = 0.0004
    risk_per_trade: float = 0.012      # 1.2% comme dans cfg
    max_position_frac: float = 0.35

    # Param SL/TP ATR comme dans l'env
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    # Fichiers du modèle et de la normalisation
    model_path: str = "best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth"
    norm_stats_path: str = "norm_stats_ohlc_indics.npz"

    # Device
    use_cuda: bool = True


cfg = EvalConfig()

device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")
print(f"[EVAL] Device = {device}")

# ----------------------------------------------------------------------
# FEATURES & INDICATEURS — identiques au training
# ----------------------------------------------------------------------

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

LIVE_MEAN: Optional[np.ndarray] = None
LIVE_STD: Optional[np.ndarray] = None
USE_NORM = False

if os.path.exists(cfg.norm_stats_path):
    try:
        norm = np.load(cfg.norm_stats_path)
        LIVE_MEAN = norm["mean"]
        LIVE_STD = norm["std"]
        LIVE_STD = np.where(LIVE_STD < 1e-8, 1.0, LIVE_STD)
        USE_NORM = True
        print(f"[{datetime.now()}] Stats de normalisation chargées depuis {cfg.norm_stats_path}")
        print(f"  → mean shape = {LIVE_MEAN.shape}, std shape = {LIVE_STD.shape}, nb features = {len(FEATURE_COLS)}")
    except Exception as e:
        print(f"[{datetime.now()}] Erreur chargement {cfg.norm_stats_path} : {e}")
        print("→ Backtest sans normalisation (différent du training, à éviter).")
        USE_NORM = False
else:
    print(f"[{datetime.now()}] Fichier {cfg.norm_stats_path} introuvable, pas de normalisation.")
    USE_NORM = False


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

    # Volatility regime
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

# ----------------------------------------------------------------------
# SAINTv2 SINGLE-HEAD — identique au training actuel
# ----------------------------------------------------------------------

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
    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 80,          # comme cfg.d_model du training
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

        # Agrégation temps/features comme dans le training
        h_time = tok.mean(dim=1)      # (B, F, D)
        h_feat = tok.mean(dim=2)      # (B, T, D)

        cls_time = h_time.mean(dim=1)  # (B, D)
        cls_feat = h_feat.mean(dim=1)  # (B, D)

        h = cls_time + cls_feat
        h = self.norm(h)
        h = self.mlp(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value


def load_policy(path: str) -> SAINTPolicySingleHead:
    policy = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=80,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    state_dict = torch.load(path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"[{datetime.now()}] Modèle SAINTv2 Single-Head chargé depuis {path}")
    return policy

# ----------------------------------------------------------------------
# MT5 — chargement de l’historique et merge M1 + H1
# ----------------------------------------------------------------------

def init_mt5():
    print("Connexion MT5…")
    if not mt5.initialize():
        raise RuntimeError(f"MT5.initialize() a échoué: {mt5.last_error()}")
    info = mt5.account_info()
    if info:
        print(f"Compte #{info.login} — balance={info.balance}, equity={info.equity}")


def shutdown_mt5():
    print("Fermeture MT5…")
    mt5.shutdown()


def load_merged_history() -> pd.DataFrame:
    rates_m1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe_m1, 0, cfg.n_bars_m1
    )
    rates_h1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe_h1, 0, cfg.n_bars_h1
    )

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("Pas assez de données M1 ou H1.")

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

    # important : ne dropper que sur les features utilisables
    merged_before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(
        f"[{datetime.now()}] Historique aligné M1+H1 : {len(merged)} bougies M1 utilisables "
        f"(avant dropna={merged_before}, après={len(merged)})"
    )
    return merged

# ----------------------------------------------------------------------
# MASK + POLICY DECISION (greedy + epsilon tiny)
# ----------------------------------------------------------------------

def build_mask_from_pos_scalar(pos: int, device) -> torch.Tensor:
    """
    pos: -1, 0, 1
    True = action valide.
    - Flat  : 0..4 (BUY/SELL 1x/1.8x + HOLD)
    - En pos: seule action 4 (HOLD)
    """
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)
    if pos == 0:
        mask[:] = True
    else:
        mask[4] = True
    return mask


def epsilon_greedy_from_logits(logits_masked: torch.Tensor, eps: float) -> int:
    """
    logits_masked : (1, N_ACTIONS) déjà masqués (MASK_VALUE sur actions invalides).
    """
    with torch.no_grad():
        if np.random.rand() > eps:
            return logits_masked.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits_masked, dim=-1)
            dist = Categorical(probs)
            return dist.sample().item()

# ----------------------------------------------------------------------
# POSITION SIZING BACKTEST — même logique que l'env
# ----------------------------------------------------------------------

def compute_size_units(price: float, atr14_prev: float, risk_scale: float, capital: float) -> float:
    """
    Même formule que _compute_dynamic_size() de l'env :
      - ATR basé sur idx-1
      - SL distance = atr_sl_mult * ATR
      - risk_per_trade * risk_scale
      - clamp par max_position_frac
    """
    if price <= 0 or capital <= 0:
        return 0.0

    fallback = 0.0015 * price
    eff_atr = max(atr14_prev, fallback, 1e-8)

    stop_distance = cfg.atr_sl_mult * eff_atr

    risk_base = cfg.risk_per_trade
    effective_risk = risk_base * max(risk_scale, 1e-6)

    size = effective_risk * capital / (stop_distance * cfg.leverage + 1e-8)
    size = size / max(price, 1e-8)

    max_notional = cfg.max_position_frac * capital
    max_size = max_notional / max(price, 1e-8)

    size = float(np.clip(size, 1e-4, max_size))
    return size

# ----------------------------------------------------------------------
# BUILD OBS (comme l’ENV) pour un index donné
# ----------------------------------------------------------------------

def build_observation_from_index(merged: pd.DataFrame,
                                 i: int,
                                 pos_dir: int,
                                 entry_price: float,
                                 current_size: float,
                                 last_realized_pnl: float) -> np.ndarray:
    """
    i : index de la bougie "courante" utilisée pour la décision.
    On reproduit :
      - env._get_obs() → features sur [idx-lookback .. idx-1]
    """
    start = i - cfg.lookback
    end = i
    df_lb = merged.iloc[start:end]
    feats = df_lb[FEATURE_COLS].values.astype(np.float32)

    if USE_NORM and LIVE_MEAN is not None and LIVE_STD is not None:
        if LIVE_MEAN.shape[0] == feats.shape[1]:
            feats = (feats - LIVE_MEAN) / LIVE_STD
        else:
            print(f"[WARN] mismatch normalisation : feats={feats.shape[1]}, mean={LIVE_MEAN.shape[0]}")

    last_price = float(df_lb["close"].iloc[-1])

    # Unrealized PnL comme dans l'env (en $ / capital initial)
    if pos_dir != 0 and current_size > 0.0 and entry_price > 0.0:
        unreal = (
            pos_dir *
            (last_price - entry_price) *
            current_size * cfg.leverage
        )
    else:
        unreal = 0.0

    unreal_norm = np.full(
        (cfg.lookback, 1),
        unreal / cfg.initial_capital,
        dtype=np.float32
    )

    last_real_norm = np.full(
        (cfg.lookback, 1),
        last_realized_pnl / cfg.initial_capital,
        dtype=np.float32
    )

    pos_oh = np.zeros((cfg.lookback, 3), dtype=np.float32)
    if pos_dir == -1:
        pos_oh[:, 0] = 1.0
    elif pos_dir == 0:
        pos_oh[:, 1] = 1.0
    else:
        pos_oh[:, 2] = 1.0

    obs = np.concatenate(
        [feats, unreal_norm, last_real_norm, pos_oh],
        axis=1
    ).astype(np.float32)

    return obs

# ----------------------------------------------------------------------
# BACKTEST PRINCIPAL — simulation SL/TP comme l'env
# ----------------------------------------------------------------------

def run_long_backtest():
    print("=== DÉMARRAGE BACKTEST LONGUE DURÉE SAINTv2 SINGLE-HEAD SCALPING ===")
    init_mt5()
    merged = load_merged_history()
    shutdown_mt5()

    if len(merged) <= cfg.lookback + 10:
        print("Pas assez de données pour backtest.")
        return

    policy = load_policy(cfg.model_path)

    # --- état simulé ---
    capital = cfg.initial_capital
    peak_equity = capital
    max_dd = 0.0

    pos_dir = 0          # -1 short, 0 flat, 1 long
    entry_price = 0.0
    current_size = 0.0   # en unités BTC

    sl_price = 0.0
    tp_price = 0.0
    entry_idx = -1
    entry_atr = 0.0

    last_realized_pnl = 0.0

    trades_pnl: List[float] = []
    equity_curve: List[float] = []

    n_steps = len(merged)
    step_count = 0

    # risk_scale interne (comme env.set_risk_scale)
    risk_scale = 1.0

    # On commence à idx = lookback pour avoir une fenêtre complète
    for i in range(cfg.lookback, n_steps):
        # ----- construction de l'obs (comme env._get_obs) -----
        obs = build_observation_from_index(
            merged, i, pos_dir, entry_price, current_size, last_realized_pnl
        )
        state_t = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1,T,F)

        # ----- policy + masking + epsilon (comme val) -----
        with torch.no_grad():
            logits, _ = policy(state_t)
            mask = build_mask_from_pos_scalar(pos_dir, device)
            logits_masked = logits.masked_fill(~mask, MASK_VALUE)
            eps = 0.05 if step_count < 1_000 else 0.0
            a = epsilon_greedy_from_logits(logits_masked, eps=eps)

        step_count += 1

        # ----- mapping agent → env_action + risk_scale (identique training) -----
        if pos_dir == 0:
            if a == 4:        # HOLD explicite
                env_action = 2
                risk_scale = 1.0
            elif a in (0, 2):  # BUY 1x / 1.8x
                env_action = 0
                risk_scale = 1.8 if a == 2 else 1.0
            elif a in (1, 3):  # SELL 1x / 1.8x
                env_action = 1
                risk_scale = 1.8 if a == 3 else 1.0
            else:
                env_action = 2
                risk_scale = 1.0
        else:
            # déjà en position → HOLD forcé, on ne touche pas au risk_scale
            env_action = 2

        # ----- simulation de la bougie courante (i) -----
        price = float(merged["close"].iloc[i])
        high_bar = float(merged["high"].iloc[i])
        low_bar = float(merged["low"].iloc[i])
        atr14_prev = float(merged["atr_14"].iloc[i - 1]) if "atr_14" in merged.columns else 0.0

        # Mise à jour du temps en position (bars_in_position)
        # (sert si tu veux analyser plus tard, ici on ne l'utilise que pour la cohérence)
        # on pourrait garder bars_in_position si besoin, mais pas indispensable pour le PnL

        # ----- ouverture éventuelle -----
        if env_action in (0, 1) and pos_dir == 0:
            side = 1 if env_action == 0 else -1
            size = compute_size_units(price, atr14_prev, risk_scale, capital)
            if size > 0.0:
                current_size = size
                pos_dir = side
                entry_price = price
                entry_idx = i

                # ATR figé à l'entrée
                fallback = 0.0015 * price
                entry_atr = max(atr14_prev, fallback, 1e-8)

                sl_dist = cfg.atr_sl_mult * entry_atr
                tp_dist = cfg.atr_tp_mult * entry_atr

                if side == 1:
                    sl_price = max(1e-8, entry_price - sl_dist)
                    tp_price = max(1e-8, entry_price + tp_dist)
                else:
                    sl_price = max(1e-8, entry_price + sl_dist)
                    tp_price = max(1e-8, entry_price - tp_dist)

                fee = cfg.fee_rate * price * size
                capital -= fee
            else:
                # pas de taille → reste flat
                pos_dir = 0
                current_size = 0.0
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
                entry_idx = -1
                entry_atr = 0.0

        # ----- fermeture automatique SL/TP (dès la bougie suivante) -----
        realized = 0.0
        hit_sl = False
        hit_tp = False

        if (
            pos_dir != 0 and
            current_size > 0.0 and
            entry_price > 0.0 and
            entry_idx >= 0 and
            i > entry_idx
        ):
            exit_price = None

            if pos_dir == 1:  # long
                if sl_price > 0 and low_bar <= sl_price:
                    exit_price = sl_price
                    hit_sl = True
                elif tp_price > 0 and high_bar >= tp_price:
                    exit_price = tp_price
                    hit_tp = True
            else:  # short
                if sl_price > 0 and high_bar >= sl_price:
                    exit_price = sl_price
                    hit_sl = True
                elif tp_price > 0 and low_bar <= tp_price:
                    exit_price = tp_price
                    hit_tp = True

            if exit_price is not None:
                pnl = (
                    pos_dir *
                    (exit_price - entry_price) *
                    current_size * cfg.leverage
                )
                fee = cfg.fee_rate * exit_price * current_size
                realized = pnl - fee
                capital += realized
                trades_pnl.append(realized)
                last_realized_pnl = realized

                # flat
                pos_dir = 0
                current_size = 0.0
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
                entry_idx = -1
                entry_atr = 0.0

        # ----- equity & drawdown -----
        latent = 0.0
        if pos_dir != 0 and current_size > 0.0 and entry_price > 0.0:
            latent = (
                pos_dir *
                (price - entry_price) *
                current_size * cfg.leverage
            )

        equity = capital + latent
        equity_curve.append(equity)

        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / (peak_equity + 1e-8)
        max_dd = max(max_dd, dd)

        if (i % 5_000) == 0:
            print(
                f"[{merged['time'].iloc[i]}] i={i}/{n_steps}, "
                f"equity={equity:.2f}, capital={capital:.2f}, pos={pos_dir}, size={current_size:.6f}"
            )

    # Si encore en position à la fin, on ferme au dernier prix (réalisation latente)
    if pos_dir != 0 and current_size > 0.0 and entry_price > 0.0:
        last_price = float(merged["close"].iloc[-1])
        pnl = (
            pos_dir *
            (last_price - entry_price) *
            current_size * cfg.leverage
        )
        fee = cfg.fee_rate * last_price * current_size
        realized = pnl - fee
        capital += realized
        trades_pnl.append(realized)
        last_realized_pnl = realized
        equity = capital
        equity_curve.append(equity)

    # ------------------------------------------------------------------
    # STATS FINALES
    # ------------------------------------------------------------------
    final_profit = capital - cfg.initial_capital
    final_return_pct = final_profit / cfg.initial_capital * 100.0

    n_trades = len(trades_pnl)
    if n_trades > 0:
        wins = [p for p in trades_pnl if p > 0]
        losses = [p for p in trades_pnl if p <= 0]
        winrate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        expectancy = (winrate * avg_win + (1 - winrate) * avg_loss)
    else:
        winrate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        expectancy = 0.0

    print("\n================== RÉSULTATS BACKTEST LONGUE DURÉE ==================")
    print(f"Symbol          : {cfg.symbol}")
    print(f"Nb bougies M1   : {len(merged)}")
    print(f"Capital initial : {cfg.initial_capital:.2f} $")
    print(f"Capital final   : {capital:.2f} $")
    print(f"Profit total    : {final_profit:.2f} $ ({final_return_pct:.2f} %)")
    print(f"Max drawdown    : {max_dd*100:.2f} %")
    print(f"Nb trades       : {n_trades}")
    print(f"Winrate         : {winrate*100:.2f} %")
    print(f"Gain moyen      : {avg_win:.2f} $")
    print(f"Perte moyenne   : {avg_loss:.2f} $")
    print(f"Expectancy/trade: {expectancy:.2f} $")
    print("=====================================================================")


if __name__ == "__main__":
    """
    ⚠️ Ce script ne passe AUCUN ordre réel.
       Il utilise MT5 uniquement pour télécharger l'historique,
       puis simule les décisions du modèle sur une longue période.

    Assure-toi que :
      - best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth
      - norm_stats_ohlc_indics.npz
    sont dans le même dossier.
    Et que l’historique BTCUSD M1/H1 est bien chargé dans ton MT5.
    """
    run_long_backtest()
