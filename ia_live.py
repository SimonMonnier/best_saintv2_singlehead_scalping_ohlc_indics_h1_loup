# ======================================================================
# LIVE TRADING — SAINTv2 SINGLE-HEAD SCALPING (MODE SAFE)
# Aligné avec le training "Loup Scalpeur"
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ======================================================================
# CONFIG LIVE (cohérent avec le training)
# ======================================================================

N_ACTIONS = 5
MASK_VALUE = -1e4

@dataclass
class LiveConfig:
    symbol: str = "BTCUSD"
    timeframe_m1: int = mt5.TIMEFRAME_M1
    timeframe_h1: int = mt5.TIMEFRAME_H1

    lookback: int = 26

    hist_bars_m1: int = 2000
    hist_bars_h1: int = 1000

    model_path: str = "best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth"
    norm_stats_path: str = "norm_stats_ohlc_indics.npz"

    initial_capital: float = 100.0
    leverage: float = 6.0
    fee_rate: float = 0.0004

    risk_per_trade: float = 0.012
    max_position_frac: float = 0.35

    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    base_volume_lots: float = 0.01
    deviation: int = 40
    magic: int = 987654321


cfg = LiveConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[LIVE] Device = {device}")

# ======================================================================
# FEATURES — identiques au training
# ======================================================================

FEATURE_COLS_M1 = [
    "open","high","low","close",
    "ret_1","ret_3","ret_5","ret_15","ret_60",
    "realized_vol_20","vol_regime",
    "ema_5","ema_10","ema_20",
    "rsi_7","rsi_14",
    "atr_14",
    "stoch_k","stoch_d",
    "macd","macd_signal",
    "dist_tenkan","dist_kijun","dist_span_a","dist_span_b",
    "ma_100","zscore_100",
    "hour_sin","hour_cos","dow_sin","dow_cos",
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
OBS_N_FEATURES = N_BASE_FEATURES + 1 + 1 + 3   # unreal + last_realized + pos_onehot

# ======================================================================
# NORMALISATION — SAFE : arrêt si mismatch
# ======================================================================

if not os.path.exists(cfg.norm_stats_path):
    raise RuntimeError(
        f"❌ Fichier de normalisation introuvable : {cfg.norm_stats_path}"
    )

norm = np.load(cfg.norm_stats_path)
LIVE_MEAN = norm["mean"]
LIVE_STD = norm["std"]
LIVE_STD = np.where(LIVE_STD < 1e-8, 1.0, LIVE_STD)

if LIVE_MEAN.shape[0] != len(FEATURE_COLS):
    raise RuntimeError(
        f"❌ Normalisation invalide : mean.shape[0]={LIVE_MEAN.shape[0]} "
        f"mais {len(FEATURE_COLS)} features définies.\n"
        f"Le fichier {cfg.norm_stats_path} ne correspond pas au training utilisé."
    )

print(f"[SAFE] Normalisation chargée ({len(FEATURE_COLS)} features).")

# ======================================================================
# INDICATEURS — identiques au training (version compacte)
# ======================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]

    df["ret_1"]  = c.pct_change(1)
    df["ret_3"]  = c.pct_change(3)
    df["ret_5"]  = c.pct_change(5)
    df["ret_15"] = c.pct_change(15)
    df["ret_60"] = c.pct_change(60)

    ret = c.pct_change()
    df["realized_vol_20"] = ret.rolling(20).std()

    roll_mean = df["realized_vol_20"].rolling(500).mean()
    roll_std  = df["realized_vol_20"].rolling(500).std()
    df["vol_regime"] = (df["realized_vol_20"] - roll_mean) / (roll_std + 1e-8)

    df["ema_5"]  = c.ewm(span=5, adjust=False).mean()
    df["ema_10"] = c.ewm(span=10, adjust=False).mean()
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()

    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - 100 / (1 + rs)

    df["rsi_7"]  = rsi(c, 7)
    df["rsi_14"] = rsi(c, 14)

    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = (c - low14) / (high14 - low14 + 1e-8) * 100
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    conv_period = 9
    base_period = 26
    span_b_period = 52

    conv_line = (h.rolling(conv_period).max() + l.rolling(conv_period).min()) / 2
    base_line = (h.rolling(base_period).max() + l.rolling(base_period).min()) / 2
    span_a = ((conv_line + base_line) / 2).shift(base_period)
    span_b = ((h.rolling(span_b_period).max() + l.rolling(span_b_period).min()) / 2).shift(base_period)

    df["ichimoku_tenkan"] = conv_line
    df["ichimoku_kijun"]  = base_line
    df["ichimoku_span_a"] = span_a
    df["ichimoku_span_b"] = span_b

    df["dist_tenkan"] = (c - conv_line) / (c + 1e-8)
    df["dist_kijun"]  = (c - base_line) / (c + 1e-8)
    df["dist_span_a"] = (c - span_a) / (c + 1e-8)
    df["dist_span_b"] = (c - span_b) / (c + 1e-8)

    ma_100  = c.rolling(100).mean()
    std_100 = c.rolling(100).std()
    df["ma_100"]    = ma_100
    df["zscore_100"] = (c - ma_100) / (std_100 + 1e-8)

    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 7)

    if "tick_volume" in df.columns:
        df["tick_volume_log"] = np.log1p(df["tick_volume"])
    else:
        df["tick_volume_log"] = 0.0

    return df

# ======================================================================
# GLOBALS RL
# ======================================================================

last_realized_pnl_rl: float = 0.0
rl_capital: float = cfg.initial_capital  # capital "virtuel" utilisé pour le sizing

# ======================================================================
# MT5 HELPERS
# ======================================================================

def init_mt5():
    global rl_capital
    print("[MT5] Connexion…")
    if not mt5.initialize():
        raise RuntimeError(f"MT5.initialize() a échoué: {mt5.last_error()}")

    acc = mt5.account_info()
    if acc:
        print(f"[MT5] Compte #{acc.login} — balance={acc.balance} equity={acc.equity}")
        # on pourrait aligner rl_capital sur la balance réelle si tu veux
        # rl_capital = float(acc.balance)
    else:
        print("[MT5] Impossible de lire les infos de compte.")

    si = mt5.symbol_info(cfg.symbol)
    if si is None:
        raise RuntimeError(f"Symbole {cfg.symbol} introuvable.")
    if not si.visible:
        if not mt5.symbol_select(cfg.symbol, True):
            raise RuntimeError(f"Impossible d'activer le symbole {cfg.symbol}.")


def shutdown_mt5():
    print("[MT5] Fermeture MT5…")
    mt5.shutdown()


def get_current_position() -> Tuple[int, float, float]:
    """
    Retourne (direction, entry_price, volume_net)
    direction : -1 short, 0 flat, 1 long
    """
    positions = mt5.positions_get(symbol=cfg.symbol)
    if positions is None or len(positions) == 0:
        return 0, 0.0, 0.0

    vol_long = 0.0
    vol_short = 0.0
    price_long = 0.0
    price_short = 0.0

    for p in positions:
        if p.type == mt5.POSITION_TYPE_BUY:
            vol_long += p.volume
            price_long = p.price_open
        elif p.type == mt5.POSITION_TYPE_SELL:
            vol_short += p.volume
            price_short = p.price_open

    if vol_long > vol_short:
        return 1, price_long, vol_long - vol_short
    elif vol_short > vol_long:
        return -1, price_short, vol_short - vol_long
    else:
        return 0, 0.0, 0.0


def open_position(decision: str, sl: float, tp: float, volume: float):
    tick = mt5.symbol_info_tick(cfg.symbol)
    if tick is None:
        print("[MT5] Tick indisponible, pas d'ouverture.")
        return

    if decision == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    volume = float(round(volume, 2))
    if volume <= 0:
        print("[MT5] Volume <= 0, pas d'ouverture.")
        return

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": cfg.deviation,
        "magic": cfg.magic,
        "comment": "SAINTv2_SH_Open",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    res = mt5.order_send(req)
    if res is None:
        print("[MT5] order_send a renvoyé None, last_error:", mt5.last_error())
    else:
        print(f"[MT5] Ouverture {decision}, volume={volume:.2f}, retcode={res.retcode}, comment={res.comment}")

# ======================================================================
# SAINT v2 — SINGLE-HEAD (même archi que le training)
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
    Même architecture que dans le script de training
    (actor: logits N_ACTIONS, critic: V(s), mais ici on utilise seulement l'actor en live).
    """

    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 80,
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
        self.scale = np.sqrt(d_model)
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

        h_time = tok.mean(dim=1)      # (B,F,D)
        h_feat = tok.mean(dim=2)      # (B,T,D)

        cls_time = h_time.mean(dim=1)  # (B,D)
        cls_feat = h_feat.mean(dim=1)  # (B,D)

        h = cls_time + cls_feat
        h = self.norm(h)
        h = self.mlp(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


def load_model() -> SAINTPolicySingleHead:
    model = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=80,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    if not os.path.exists(cfg.model_path):
        raise RuntimeError(f"❌ Modèle introuvable : {cfg.model_path}")

    sd = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[MODEL] SAINTPolicySingleHead chargé depuis {cfg.model_path}")
    return model

# ======================================================================
# MASKING & MERGE M1+H1
# ======================================================================

def build_mask_from_pos(pos: int, device) -> torch.Tensor:
    """
    Identique au training :
    - Flat : 5 actions valides (0..4)
    - En position : uniquement HOLD (4)
    """
    m = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)
    if pos == 0:
        m[:] = True
    else:
        m[4] = True
    return m


def fetch_merged_m1_h1() -> Optional[pd.DataFrame]:
    try:
        m1 = mt5.copy_rates_from_pos(cfg.symbol, cfg.timeframe_m1, 0, cfg.hist_bars_m1)
        h1 = mt5.copy_rates_from_pos(cfg.symbol, cfg.timeframe_h1, 0, cfg.hist_bars_h1)
    except Exception as e:
        print(f"[ERR] MT5 copy_rates_from_pos: {e}")
        return None

    if m1 is None or h1 is None:
        print("[ERR] Pas assez de données M1 ou H1.")
        return None

    df_m1 = pd.DataFrame(m1)
    df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
    df_m1 = df_m1.set_index("time")
    df_m1 = df_m1[["open","high","low","close","tick_volume"]]
    df_m1 = add_indicators(df_m1)

    df_h1 = pd.DataFrame(h1)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    df_h1 = df_h1.set_index("time")
    df_h1 = df_h1[["open","high","low","close","tick_volume"]]
    df_h1 = add_indicators(df_h1)
    df_h1 = df_h1.add_suffix("_h1")

    merged = pd.merge_asof(
        df_m1.sort_index(),
        df_h1.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    merged = merged.dropna(subset=FEATURE_COLS)
    if len(merged) < cfg.lookback:
        return None
    return merged

# ======================================================================
# OBSERVATION — identique à l'env
# ======================================================================

def build_observation(pos_dir: int, entry_price: float, volume_net: float):
    global last_realized_pnl_rl

    merged = fetch_merged_m1_h1()
    if merged is None or len(merged) < cfg.lookback:
        print("[OBS] Pas assez de merged M1+H1.")
        return None, None, None

    df_lb = merged.iloc[-cfg.lookback:]
    feats = df_lb[FEATURE_COLS].values.astype(np.float32)

    feats = (feats - LIVE_MEAN) / LIVE_STD

    last_price = float(df_lb["close"].iloc[-1])
    last_atr = float(df_lb["atr_14"].iloc[-2]) if len(df_lb) >= 2 else 0.0

    if pos_dir != 0 and entry_price > 0 and volume_net > 0:
        latent = pos_dir * (last_price - entry_price) * volume_net * cfg.leverage
    else:
        latent = 0.0

    unreal_norm = latent / cfg.initial_capital
    last_real_norm = last_realized_pnl_rl / cfg.initial_capital

    unreal_arr = np.full((cfg.lookback, 1), unreal_norm, dtype=np.float32)
    last_real_arr = np.full((cfg.lookback, 1), last_real_norm, dtype=np.float32)

    pos_oh = np.zeros((cfg.lookback, 3), dtype=np.float32)
    if pos_dir == -1:
        pos_oh[:, 0] = 1.0
    elif pos_dir == 0:
        pos_oh[:, 1] = 1.0
    else:
        pos_oh[:, 2] = 1.0

    obs = np.concatenate([feats, unreal_arr, last_real_arr, pos_oh], axis=1)
    return obs.astype(np.float32), last_price, last_atr

# ======================================================================
# SIZING & MAPPING & SL/TP (alignés env)
# ======================================================================

def compute_live_volume_lots(price: float, atr14: float, risk_scale: float) -> float:
    global rl_capital
    if price <= 0 or rl_capital <= 0:
        return 0.0

    fallback = 0.0015 * price
    eff_atr = max(atr14, fallback, 1e-8)
    stop_distance = cfg.atr_sl_mult * eff_atr

    risk_base = cfg.risk_per_trade
    effective_risk = risk_base * max(risk_scale, 1e-6)

    size_units = effective_risk * rl_capital / (stop_distance * cfg.leverage + 1e-8)
    max_notional = cfg.max_position_frac * rl_capital
    max_units = max_notional / max(price, 1e-8)
    size_units = float(np.clip(size_units, 1e-4, max_units))

    volume_lots = size_units  # 1 lot ≈ 1 BTC sur BTCUSD
    volume_lots = max(cfg.base_volume_lots, volume_lots)
    volume_lots = float(round(volume_lots, 2))
    return volume_lots * 10


def map_action(a: int, pos_dir: int):
    if pos_dir == 0:
        if a == 4:
            return "HOLD", 1.0
        if a == 0:
            return "BUY", 1.0
        if a == 2:
            return "BUY", 1.8
        if a == 1:
            return "SELL", 1.0
        if a == 3:
            return "SELL", 1.8
        return "HOLD", 1.0
    else:
        return "HOLD", 1.0


def compute_sl_tp(decision: str, price: float, atr14: float):
    if atr14 <= 0:
        return None, None
    if decision == "BUY":
        sl = price - cfg.atr_sl_mult * atr14
        tp = price + cfg.atr_tp_mult * atr14
    elif decision == "SELL":
        sl = price + cfg.atr_sl_mult * atr14
        tp = price - cfg.atr_tp_mult * atr14
    else:
        return None, None
    return sl, tp

# ======================================================================
# SYNC M1
# ======================================================================

def sleep_until_next_m1_close():
    now = datetime.now()
    sec = now.second + now.microsecond / 1e6
    remaining = 60.0 - sec
    if remaining < 0:
        remaining += 60.0
    time.sleep(remaining + 0.3)

# ======================================================================
# RL → MT5
# ======================================================================

def apply_decision(model: SAINTPolicySingleHead):
    global last_realized_pnl_rl, rl_capital

    acc = mt5.account_info()
    if acc:
        # on peut utiliser la diff vs initial_capital comme proxy PnL réalisé
        last_realized_pnl_rl = float(acc.balance - cfg.initial_capital)
        # si tu veux que le sizing suive la courbe de balance :
        # rl_capital = float(acc.balance)

    pos_dir, entry_price, vol_net = get_current_position()
    obs, last_price, last_atr = build_observation(pos_dir, entry_price, vol_net)
    if obs is None:
        return

    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,F)

    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[0]
        mask = build_mask_from_pos(pos_dir, device)
        logits = logits.masked_fill(~mask, MASK_VALUE)
        a = logits.argmax().item()

    decision, risk_scale = map_action(a, pos_dir)

    print(
        f"[DECISION] action={a}, decision={decision}, pos={pos_dir}, "
        f"price={last_price:.2f}, atr={last_atr:.6f}, "
        f"vol_net={vol_net:.3f}, risk_scale={risk_scale:.2f}"
    )

    if decision in ("BUY", "SELL") and pos_dir == 0:
        volume = compute_live_volume_lots(last_price, last_atr, risk_scale)
        sl, tp = compute_sl_tp(decision, last_price, last_atr)
        if sl is not None and volume > 0:
            open_position(decision, sl, tp, volume)
    else:
        # HOLD ou déjà en position : gestion des sorties par SL/TP MT5
        pass

# ======================================================================
# MAIN LOOP
# ======================================================================

def live_loop():
    print("=== DÉMARRAGE LIVE SAINTv2 SINGLE-HEAD (SAFE) ===")
    init_mt5()
    model = load_model()

    heartbeat_last = datetime.now()

    try:
        while True:
            sleep_until_next_m1_close()
            try:
                apply_decision(model)
            except Exception as e:
                print(f"[ERR] apply_decision: {e}")

            now = datetime.now()
            if (now - heartbeat_last).total_seconds() > 600:
                print(f"[HEARTBEAT] {now} — script vivant.")
                heartbeat_last = now

    except KeyboardInterrupt:
        print("[STOP] Interruption manuelle (Ctrl+C).")

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    live_loop()
