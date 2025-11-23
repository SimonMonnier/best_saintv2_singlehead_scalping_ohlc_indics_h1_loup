# ======================================================================
# LIVE TRADING — SAINTv2 SINGLE-HEAD SCALPING (BTCUSD M1, M1+H1 FEATURES)
#  - Architecture identique au TRAINING "Loup Scalpeur"
#  - Mêmes features : M1+H1 (OHLC + indicateurs + Ichimoku + temps)
#  - Même normalisation : norm_stats_ohlc_indics.npz (FEATURE_COLS M1+H1)
#  - Même action space : 5 actions (0..4) comme en training :
#       0 : BUY  (risk 1.0)
#       1 : SELL (risk 1.0)
#       2 : BUY  (risk 1.8)
#       3 : SELL (risk 1.8)
#       4 : HOLD
#  - Fermeture uniquement via SL/TP ATR (pas d’action CLOSE manuelle)
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ======================================================================
# CONFIG LIVE (COHÉRENTE AVEC LE TRAINING "LOUP SCALPEUR")
# ======================================================================

# 0 : BUY 1x
# 1 : SELL 1x
# 2 : BUY 1.8x
# 3 : SELL 1.8x
# 4 : HOLD
N_ACTIONS = 5
MASK_VALUE = -1e4  # même valeur que training (compatible float16)


@dataclass
class LiveConfig:
    # symboles / timeframes
    symbol: str = "BTCUSD"
    timeframe_m1: int = mt5.TIMEFRAME_M1
    timeframe_h1: int = mt5.TIMEFRAME_H1

    lookback: int = 26         # identique au training
    hist_bars_m1: int = 2000    # un peu plus long pour les indicateurs
    hist_bars_h1: int = 1000

    # chemins des fichiers entraînés
    model_path: str = "best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth"
    norm_stats_path: str = "norm_stats_ohlc_indics.npz"

    # Hypothèses RL (alignées sur le training)
    initial_capital: float = 100.0   # même échelle que l'env
    leverage: float = 6.0
    fee_rate: float = 0.0004

    # Risk management live (approx de l'env)
    risk_per_trade: float = 0.012      # 1.2 % du capital par trade (avant risk_scale)
    max_position_frac: float = 0.35    # max 35 % du capital en notional

    # ATR SL/TP (identique à l'env)
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    # Exécution réelle
    base_volume_lots: float = 0.01     # volume minimal de base, sera modulé
    deviation: int = 40
    magic: int = 987654321

    # Epsilon-greedy (non utilisé en prod, eps=0)
    eps_warmup_steps: int = 1000


cfg = LiveConfig()

# ======================================================================
# DEVICE
# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[LIVE] Device = {device}")

# ======================================================================
# FEATURES & NORMALISATION — MÊMES COLONNES QUE LE TRAINING FINAL
# ======================================================================

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
# + unreal_norm + last_realized_norm + pos_onehot(3)
OBS_N_FEATURES = N_BASE_FEATURES + 1 + 1 + 3

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
        print("→ Live sans normalisation (différent du training, à éviter).")
        USE_NORM = False
else:
    print(f"[{datetime.now()}] Fichier {cfg.norm_stats_path} introuvable, pas de normalisation.")
    USE_NORM = False

# ======================================================================
# INDICATEURS — COPIÉS DU TRAINING (M1 & H1)
# ======================================================================

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

# ======================================================================
# SAINTv2 SINGLE-HEAD — IDENTIQUE AU TRAINING
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

    Identique à la version training.
    """

    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 96,        # doit matcher cfg.d_model du training
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

        # Agrégation double temps / features (comme training)
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


def load_policy_live(path: str) -> SAINTPolicySingleHead:
    policy = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=80,           # IMPORTANT : même que cfg.d_model du training
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

# ======================================================================
# MT5 HELPERS
# ======================================================================

def init_mt5():
    global rl_capital
    print("Connexion à MetaTrader 5…")
    if not mt5.initialize():
        raise RuntimeError(f"MT5.initialize() a échoué: {mt5.last_error()}")
    info = mt5.account_info()
    if info:
        print(f"Compte #{info.login} — balance={info.balance}, equity={info.equity}")
        # Capital RL initial aligné sur la balance réelle (comme capital de l'env)
        rl_capital = float(info.balance)
    si = mt5.symbol_info(cfg.symbol)
    if si is None:
        raise RuntimeError(f"Symbole {cfg.symbol} introuvable.")
    if not si.visible:
        if not mt5.symbol_select(cfg.symbol, True):
            raise RuntimeError(f"Impossible d'activer le symbole {cfg.symbol}.")


def shutdown_mt5():
    print("Fermeture de MT5…")
    mt5.shutdown()


def get_current_position(symbol: str):
    """
    Retourne :
        direction : -1 short, 0 flat, 1 long
        entry_price : prix moyen d'entrée (approx)
        volume_net : volume net (lots, long - short)
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return 0, 0.0, 0.0

    vol_long = 0.0
    vol_short = 0.0
    price_long = 0.0
    price_short = 0.0

    for pos in positions:
        if pos.type == mt5.POSITION_TYPE_BUY:
            vol_long += pos.volume
            price_long = pos.price_open
        elif pos.type == mt5.POSITION_TYPE_SELL:
            vol_short += pos.volume
            price_short = pos.price_open

    if vol_long > vol_short:
        return 1, price_long, vol_long - vol_short
    elif vol_short > vol_long:
        return -1, price_short, vol_short - vol_long
    else:
        return 0, 0.0, 0.0


def open_position(order_type, symbol: str, volume: float, sl: float, tp: float):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Tick indisponible, impossible d'ouvrir.")
        return

    # volume normalisé à 2 décimales
    volume = float(round(volume, 2))
    if volume <= 0:
        print("Volume <= 0, pas d'ouverture.")
        return

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": cfg.deviation,
        "magic": cfg.magic,
        "comment": "SAINTv2_SH_Open",  # <= 31 caractères
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    res = mt5.order_send(req)
    if res is None:
        le = mt5.last_error()
        print(f"Ouverture {symbol}, volume={volume:.2f} — order_send a renvoyé None (last_error={le})")
        return

    print(f"Ouverture {symbol}, type={order_type}, volume={volume:.2f}, retcode={res.retcode}")
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Erreur ouverture: {res.comment}")


def close_all_positions(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        print("Erreur positions_get pour fermeture.")
        return

    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("Tick indisponible, impossible de fermer.")
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY

        volume = float(round(pos.volume, 2))

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": cfg.deviation,
            "magic": cfg.magic,
            "comment": "SAINTv2_SH_Close",  # <= 31 caractères
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        res = mt5.order_send(req)
        if res is None:
            le = mt5.last_error()
            print(
                f"Fermeture ticket #{pos.ticket}, volume={volume:.2f} — "
                f"order_send a renvoyé None (last_error={le})"
            )
            continue

        print(f"Fermeture ticket #{pos.ticket}, volume={volume:.2f}, retcode={res.retcode}")
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Erreur fermeture: {res.comment}")

# ======================================================================
# RL-SIDE CAPITAL (POUR CONSISTANCE AVEC L'ENV)
# ======================================================================

last_realized_pnl_rl: float = 0.0     # PnL réalisé en $
rl_capital: float = cfg.initial_capital
live_step_count: int = 0

# ======================================================================
# ACTION MASKING & EPSILON-GREEDY (MÊME LOGIQUE QUE TRAINING)
# ======================================================================

def build_mask_from_pos_scalar(pos: int, device) -> torch.Tensor:
    """
    Masque Solution A :
      - Flat  : actions valides = {0:BUY1x, 1:SELL1x, 4:HOLD}
      - En pos: seulement HOLD (4)
    """
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)
    if pos == 0:
        mask[0] = True   # BUY 1x
        mask[1] = True   # SELL 1x
        mask[4] = True   # HOLD
    else:
        mask[4] = True   # seulement HOLD lorsque déjà en position
    return mask


def epsilon_greedy_from_logits(logits_masked: torch.Tensor, eps: float) -> int:
    """
    logits_masked : (1, N_ACTIONS) déjà masqués avec MASK_VALUE.
    Exploration epsilon-greedy respectant les masques.
    """
    with torch.no_grad():
        if np.random.rand() > eps:
            return logits_masked.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits_masked, dim=-1)
            dist = Categorical(probs)
            return dist.sample().item()

# ======================================================================
# BUILD OBS LIVE — ALIGNEMENT M1 + H1, MÊMES FEATURES QUE TRAINING
# ======================================================================

def fetch_merged_m1_h1() -> Optional[pd.DataFrame]:
    try:
        rates_m1 = mt5.copy_rates_from_pos(
            cfg.symbol, cfg.timeframe_m1, 0, cfg.hist_bars_m1
        )
        rates_h1 = mt5.copy_rates_from_pos(
            cfg.symbol, cfg.timeframe_h1, 0, cfg.hist_bars_h1
        )
    except Exception as e:
        print(f"[{datetime.now()}] Erreur MT5 copy_rates_from_pos: {e}")
        return None

    if rates_m1 is None or rates_h1 is None:
        print(f"[{datetime.now()}] Pas assez de données M1 ou H1.")
        return None

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

    # 🔧 On ne droppe que sur les features réellement utilisées
    merged_before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    merged_after = len(merged)

    if merged_after < cfg.lookback:
        print(
            f"[{datetime.now()}] Pas assez de merged M1+H1 pour l'obs. "
            f"(avant dropna={merged_before}, après={merged_after}, lookback={cfg.lookback})"
        )
        return None

    return merged


def build_observation(position_dir: int, entry_price: float, volume_net: float):
    """
    Construit l'observation (lookback, OBS_N_FEATURES) exactement comme l'env :
        - features normalisés (FEATURE_COLS)
        - unreal_norm (PnL latent / initial_capital)
        - last_realized_norm (PnL réalisé / initial_capital)
        - position one-hot (T,3)
    """
    merged = fetch_merged_m1_h1()
    if merged is None or len(merged) < cfg.lookback:
        print(f"[{datetime.now()}] Pas assez de merged M1+H1 pour l'obs.")
        return None, None, None

    df_lb = merged.iloc[-cfg.lookback:]
    feats = df_lb[FEATURE_COLS].values.astype(np.float32)

    if USE_NORM and LIVE_MEAN is not None and LIVE_STD is not None:
        if LIVE_MEAN.shape[0] == feats.shape[1]:
            feats = (feats - LIVE_MEAN) / LIVE_STD
        else:
            print(f"[WARN] Dimension mismatch normalisation : feats={feats.shape[1]}, mean={LIVE_MEAN.shape[0]}")

    last_price = float(df_lb["close"].iloc[-1])

    # Utiliser l'ATR de la bougie *précédente* pour le sizing et le SL/TP
    if "atr_14" in df_lb.columns and len(df_lb) >= 2:
        last_atr14 = float(df_lb["atr_14"].iloc[-2])
    else:
        last_atr14 = 0.0

    # Unrealized PnL latent (comme l'env) normalisé par initial_capital
    if position_dir != 0 and entry_price > 0.0 and volume_net > 0.0:
        latent = (
            position_dir *
            (last_price - entry_price) *
            volume_net *
            cfg.leverage
        )
    else:
        latent = 0.0

    unreal_norm_arr = np.full(
        (cfg.lookback, 1),
        latent / cfg.initial_capital,
        dtype=np.float32
    )

    last_real_norm_arr = np.full(
        (cfg.lookback, 1),
        last_realized_pnl_rl / cfg.initial_capital,
        dtype=np.float32
    )

    pos_oh = np.zeros((cfg.lookback, 3), dtype=np.float32)
    if position_dir == -1:
        pos_oh[:, 0] = 1.0
    elif position_dir == 0:
        pos_oh[:, 1] = 1.0
    else:
        pos_oh[:, 2] = 1.0

    obs = np.concatenate(
        [feats, unreal_norm_arr, last_real_norm_arr, pos_oh],
        axis=1
    ).astype(np.float32)

    return obs, last_price, last_atr14

# ======================================================================
# DYNAMIC POSITION SIZING LIVE (MIRROR DE L'ENV)
# ======================================================================

def compute_live_volume_lots(price: float, atr14: float, risk_scale: float) -> float:
    """
    Approximation de _compute_dynamic_size de l'env, mais en lots MT5.
    On suppose 1 lot = 1 BTC sur BTCUSD.
    """
    global rl_capital
    if price <= 0 or rl_capital <= 0:
        return cfg.base_volume_lots

    fallback = 0.0015 * price
    eff_atr = max(atr14, fallback, 1e-8)

    stop_distance = cfg.atr_sl_mult * eff_atr

    risk_base = cfg.risk_per_trade
    effective_risk = risk_base * max(risk_scale, 1e-6)

    # taille en unités instrument (BTC)
    size_units = effective_risk * rl_capital / (stop_distance * cfg.leverage + 1e-8)

    max_notional = cfg.max_position_frac * rl_capital
    max_units = max_notional / max(price, 1e-8)

    size_units = float(np.clip(size_units, 1e-4, max_units))

    volume_lots = size_units

    volume_lots = float(max(cfg.base_volume_lots, volume_lots))
    volume_lots = float(round(volume_lots, 2)) * 10
    return volume_lots


def compute_sl_tp_from_atr(price: float, atr14: float, side: int):
    """
    SL/TP identiques à l'env :
      sl_dist = atr_sl_mult * eff_atr
      tp_dist = atr_tp_mult * eff_atr
    """
    fallback = 0.0015 * price
    eff_atr = max(atr14, fallback, 1e-8)

    sl_dist = cfg.atr_sl_mult * eff_atr
    tp_dist = cfg.atr_tp_mult * eff_atr

    if side == 1:   # long
        sl = max(1e-8, price - sl_dist)
        tp = max(1e-8, price + tp_dist)
    else:           # short
        sl = max(1e-8, price + sl_dist)
        tp = max(1e-8, price - tp_dist)

    return sl, tp

# ======================================================================
# DECISION LOOP RL → MT5 (MÊME MAPPING QUE TRAINING)
# ======================================================================

def apply_decision(policy: SAINTPolicySingleHead):
    """
    - Récupère position MT5
    - Met à jour le capital RL via la balance (PnL réalisé)
    - Construit l'observation live (M1+H1, mêmes features que training)
    - Applique policy single-head + masking (Solution A) + eps=0
    - Map l'action vers BUY/SELL/HOLD + risk_scale
    - Ouvre avec SL/TP ATR, fermeture *uniquement* via SL/TP (pas de CLOSE manuel)
    """
    global last_realized_pnl_rl, rl_capital, live_step_count

    # MAJ capital RL via la balance (PnL réalisé)
    info_acc = mt5.account_info()
    if info_acc is not None:
        new_cap = float(info_acc.balance)
        last_realized_pnl_rl = new_cap - rl_capital
        rl_capital = new_cap

    pos_dir, entry_price, vol_net = get_current_position(cfg.symbol)

    obs, last_price, last_atr = build_observation(pos_dir, entry_price, vol_net)
    if obs is None:
        return

    state_t = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1,T,F)

    with torch.no_grad():
        logits, _ = policy(state_t)  # (1,N_ACTIONS)
        logits_single = logits[0]

        mask = build_mask_from_pos_scalar(pos_dir, device)
        logits_masked = logits_single.masked_fill(~mask, MASK_VALUE)

        # En production : eps = 0 → on suit la policy greedy
        eps = 0.0
        a = epsilon_greedy_from_logits(logits_masked.unsqueeze(0), eps=eps)

    # Mapping agent action -> décision + risk_scale (identique à training)
    if pos_dir == 0:
        if a == 4:        # HOLD explicite
            decision = "HOLD"
            risk_scale = 1.0
        elif a in (0, 2):  # BUY
            decision = "BUY"
            risk_scale = 1.8 if a == 2 else 1.0
        elif a in (1, 3):  # SELL
            decision = "SELL"
            risk_scale = 1.8 if a == 3 else 1.0
        else:
            decision = "HOLD"
            risk_scale = 1.0
    else:
        # En position : HOLD uniquement, la fermeture est gérée par SL/TP
        decision = "HOLD"
        risk_scale = 1.0

    live_step_count += 1

    print(
        f"[{datetime.now()}] DECISION {decision} "
        f"(pos={pos_dir}, action={a}, price={last_price:.2f}, "
        f"atr14={last_atr:.6f}, vol_net={vol_net:.2f}, risk_scale={risk_scale:.2f})"
    )
    print(
        f"[STATE RL] pos={pos_dir}, entry={entry_price:.2f}, "
        f"last_realized_pnl={last_realized_pnl_rl:.2f}, rl_capital={rl_capital:.2f}"
    )

    # --- Application de la décision sur le compte réel ---
    if pos_dir == 0:
        # FLAT -> ouverture potentielle
        if decision in ("BUY", "SELL"):
            volume = compute_live_volume_lots(last_price, last_atr, risk_scale)
            side = 1 if decision == "BUY" else -1
            sl, tp = compute_sl_tp_from_atr(last_price, last_atr, side)
            order_type = mt5.ORDER_TYPE_BUY if decision == "BUY" else mt5.ORDER_TYPE_SELL
            open_position(order_type, cfg.symbol, volume, sl, tp)
        else:
            # HOLD
            pass
    else:
        # EN POSITION -> on laisse SL/TP gérer la fermeture, pas de CLOSE manuel
        pass

# ======================================================================
# SYNC M1
# ======================================================================

def sleep_until_next_m1_close():
    now = datetime.now()
    secs = now.second + now.microsecond / 1e6
    remaining = 60.0 - secs
    if remaining < 0:
        remaining += 60.0
    # petite marge de sécurité
    time.sleep(remaining + 0.3)

# ======================================================================
# MAIN LIVE LOOP
# ======================================================================

def live_loop():
    print("=== DÉMARRAGE LIVE SAINTv2 SINGLE-HEAD SCALPING (M1+H1 FEATURES, LOUP SCALPEUR) ===")
    init_mt5()
    policy = load_policy_live(cfg.model_path)

    heartbeat_last = datetime.now()

    try:
        while True:
            sleep_until_next_m1_close()
            try:
                apply_decision(policy)
            except Exception as e:
                print(f"[{datetime.now()}] ERREUR dans apply_decision: {e}")

            # Heartbeat toutes les 10 minutes
            now = datetime.now()
            if (now - heartbeat_last).total_seconds() > 600:
                print(f"[{now}] HEARTBEAT — EA vivant, steps={live_step_count}, rl_capital≈{rl_capital:.2f}")
                heartbeat_last = now

    except KeyboardInterrupt:
        print("Arrêt manuel (Ctrl+C).")
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    """
    ⚠️ Lancer sur COMPTE DÉMO d'abord.
    Assure-toi que :
      - best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth
      - norm_stats_ohlc_indics.npz
    sont présents dans le même dossier.
    """
    live_loop()
