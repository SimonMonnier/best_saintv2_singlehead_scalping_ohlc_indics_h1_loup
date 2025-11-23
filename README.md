````markdown
# Loup Scalpeur — SAINTv2 RL Trader (BTCUSD M1)

Ce dépôt contient tout le pipeline pour entraîner, évaluer et faire tourner en live un agent de trading RL basé sur l’architecture **SAINTv2 Single-Head** pour du **scalping BTCUSD en M1**, avec features **M1 + H1 (OHLC + indicateurs)** et exécution via **MetaTrader 5**.

---

## 🧩 Structure du projet

- `model_saint.py`  
  Script d’**entraînement** du modèle RL (PPO) sur un environnement de trading simulé.
- `ia_live.py`  
  Script de **trading live** : connexion à MT5, décision du modèle, passage d’ordres réels (ou démo).
- `eval_long_run.py`  
  Script d’**évaluation longue durée (backtest offline)** sur historique M1+H1 téléchargé depuis MT5.

Des fichiers auxiliaires sont utilisés :

- `norm_stats_ohlc_indics.npz` — statistiques de normalisation des features (mean/std)
- `best_saintv2_*.pth` — poids du modèle entraîné (checkpoints)
- éventuels fichiers de config (JSON/INI/YAML) selon ton organisation

---

## ⚙️ Prérequis

### Python & dépendances

- Python 3.9+ recommandé
- Librairies principales :
  - `torch`
  - `numpy`
  - `pandas`
  - `MetaTrader5`
  - `tqdm` (éventuellement)
  - `matplotlib` / `seaborn` (si tu fais des plots)
  - etc.

Installation type :

```bash
pip install -r requirements.txt
````

*(à adapter selon ton fichier `requirements.txt`)*

### MetaTrader 5

* MT5 installé sur la machine
* Compte (démo ou réel)
* Symbol **BTCUSD** disponible
* Autoriser la connexion Python :

  * Terminal MT5 ouvert
  * Connexion au bon compte
  * Historique M1 et H1 chargé (scroller dans les graphes si besoin)

---

## 🧠 Architecture du modèle — SAINTv2 Single-Head

Le modèle est un SAINT (Self-Attention for Tabular) adapté au time series :

* **Input** : séquence de longueur `lookback` sur les features M1 + H1
* **Features M1** :

  * OHLC, retours (ret_1, ret_3, …), volatilité réalisée,
  * EMAs, RSI, ATR, Stoch, MACD,
  * Ichimoku (distances tenkan/kijun/span A/B),
  * MA100, zscore_100,
  * embedding temporel (hour_sin/cos, dow_sin/cos),
  * volume (tick_volume_log).
* **Features H1** :

  * dérivés en suffixe `_h1` (close_h1, ema_20_h1, rsi_14_h1, macd_h1, etc.)
* **Context RL** injecté dans l’observation :

  * PnL latent (unrealized) normalisé,
  * dernier PnL réalisé (last_realized),
  * one-hot de direction de position : short / flat / long.

Le modèle sort :

* **Policy (actor)** : logits sur `N_ACTIONS = 6`
* **Value (critic)** : estimation de la valeur V(s)

Espace d’actions :

* `0` : Ouvrir **BUY** (risk scale 1x)
* `1` : Ouvrir **SELL** (risk scale 1x)
* `2` : Ouvrir **BUY** (risk scale 1.8x)
* `3` : Ouvrir **SELL** (risk scale 1.8x)
* `4` : **CLOSE** (si en position)
* `5` : **HOLD**

Un **masquage des actions** est appliqué pour interdire les actions incohérentes (pas d’ouverture si déjà en position, pas de CLOSE si flat, etc.).

---

## 📦 1. Entraînement — `model_saint.py`

Script de training PPO sur un environnement de trading simulé, avec :

* Architecture **SAINTv2 Single-Head**
* Optimisation via **PPO** (KL monitor, clipping, entropy, GAE, etc.)
* Reward basé sur la performance du portefeuille (retours, drawdown, Calmar, …)
* Séparation **train/validation**
* Sauvegarde des meilleurs modèles selon la métrique **Calmar30**

### Caractéristiques principales

* **Symbol** : BTCUSD
* **Timeframe principal** : M1
* **Context H1** fusionné dans l’observation
* **Gestion de risque** :

  * `initial_capital` (ex: 10 000$)
  * `leverage` (ex: 6x)
  * `fee_rate` (ex: 0.0004)
  * `risk_per_trade` (ex: 0.9% du capital)
  * stop basé sur **ATR14**

### Lance l’entraînement

Exemple simple :

```bash
python model_saint.py
```

Tu peux gérer :

* le nombre d’epochs
* la longueur de l’historique
* les hyperparamètres PPO
* les chemins de sauvegarde des modèles

directement dans le script ou via des arguments / fichier de config (à adapter à ton implémentation).

### Sorties typiques

Le script logge par epoch :

* `TrainPNL`, `ValPNL`
* `Trades`, `ValTrades`
* `Win%`, `ValWin%`
* `DD`, `ValDD`
* `Calmar`, `Calmar30`
* `ENV B/S/H` (répartition des actions BUY/SELL/HOLD)
* `KL` (stabilité PPO)
* Signaux du type :

  * `Nouveau best model (Calmar30=...)`
    → sauvegarde du meilleur checkpoint `.pth`

Le meilleur modèle est ensuite utilisé pour le live et l’évaluation longue.

---

## 💹 2. Trading Live — `ia_live.py`

Script de **trading automatique** en temps réel via MetaTrader 5 :

* Charge un modèle SAINTv2 pré-entraîné (`.pth`)
* Se connecte à MT5 (`MetaTrader5.initialize`)
* Récupère les dernières bougies M1 et H1
* Construit l’observation (features + context RL)
* Applique la policy du modèle (avec éventuellement un peu de random / epsilon)
* Traduit l’action en **ordre MT5** :

  * ouverture position (BUY / SELL, taille calculée)
  * fermeture position (CLOSE)
  * HOLD → pas d’action

### Lancer le live

Assure-toi d’avoir :

* MT5 ouvert, connecté au bon compte
* symbol BTCUSD disponible
* le fichier modèle (ex: `best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth`)
* le fichier de normalisation `norm_stats_ohlc_indics.npz`

Exemple :

```bash
python ia_live.py
```

Selon ton implémentation, tu peux avoir :

* un mode **dry-run** / **paper trading**
* un paramètre pour définir le **lot minimum**, slippage, etc.
* des logs console / fichier pour suivre les décisions en temps réel

⚠️ **Important** :
Toujours tester en **démo** avant de brancher sur un compte réel.
Vérifie la cohérence des tailles d’ordres, du levier, des stops, et des frais.

---

## 📊 3. Backtest Longue Durée — `eval_long_run.py`

Script d’**évaluation offline** d’un modèle SAINTv2 sur un long historique M1+H1.

### Objectif

* Ne **passe aucun ordre réel**
* Utilise MT5 seulement pour **télécharger l’historique**
* Simule les décisions du modèle bougie par bougie
* Reproduit le sizing, les fees, et la logique de l’environnement live

### Pipeline du script

1. **Connexion MT5**

   * `init_mt5()` → `mt5.initialize()`
   * log des infos de compte

2. **Téléchargement de l’historique**

   * M1 : `mt5.copy_rates_from_pos(symbol, TIMEFRAME_M1, 0, n_bars_m1)`
   * H1 : `mt5.copy_rates_from_pos(symbol, TIMEFRAME_H1, 0, n_bars_h1)`

3. **Construction des features**

   * `add_indicators(df_m1)` + `add_indicators(df_h1)`
   * suffixe `_h1` pour les features H1
   * merge asof M1/H1 → `merged` (bougies M1 alignées sur dernier H1 connu)

4. **Normalisation**

   * Chargement de `norm_stats_ohlc_indics.npz`
   * Application de `(x - mean) / std` sur les features

5. **Simulation RL**

   * Boucle `for i in range(lookback, n_steps-1)` :

     * build observation avec `build_observation_from_index(...)`
     * passage dans le modèle → logits, value
     * application du mask selon `pos_dir` (`build_mask_from_pos_scalar`)
     * sélection d’action via `greedy_action_from_logits` (epsilon-greedy léger)
     * simulation du trade :

       * ouverture BUY/SELL avec `compute_size_units`
       * CLOSE → calcul PnL réalisé (fees inclus)
       * HOLD → rien
     * mise à jour :

       * `capital`, `equity`, `peak_equity`, `max_dd`
       * ajout des PnL de trades à `trades_pnl`
       * sauvegarde de la courbe d’equity

6. **Fermeture de position finale**

   * Si en position à la fin du backtest → fermeture sur la dernière bougie

7. **Stats finales**

   * Capital final vs initial
   * Profit total absolu et en %
   * Max drawdown
   * Nombre de trades
   * Winrate
   * Gain moyen, perte moyenne
   * Expectancy par trade

### Lancer le backtest

```bash
python eval_long_run.py
```

Assure-toi que :

* `cfg.model_path` pointe sur le bon modèle `.pth`
* `cfg.norm_stats_path` existe (`norm_stats_ohlc_indics.npz`)
* MT5 a suffisamment d’historique BTCUSD M1/H1

---

## 🧾 Exemple de configuration (dans `EvalConfig`)

Dans `eval_long_run.py`, tu as une dataclass du type :

```python
@dataclass
class EvalConfig:
    symbol: str = "BTCUSD"
    timeframe_m1: int = mt5.TIMEFRAME_M1
    timeframe_h1: int = mt5.TIMEFRAME_H1

    n_bars_m1: int = 250_000
    n_bars_h1: int = 25_000

    lookback: int = 26

    initial_capital: float = 10_000.0
    leverage: float = 6.0
    fee_rate: float = 0.0004
    risk_per_trade: float = 0.009
    max_position_frac: float = 0.35

    model_path: str = "best_saintv2_clamar92.pth"
    norm_stats_path: str = "norm_stats_ohlc_indics.npz"

    use_cuda: bool = True
```

Tu peux ajuster ces paramètres directement dans le script ou les rendre configurables via arguments, si tu préfères.

---

## ✅ Bonnes pratiques

* Toujours vérifier :

  * cohérence des features entre **training**, **eval**, **live**
  * même ordre de colonnes `FEATURE_COLS`
  * même normalisation (`norm_stats_ohlc_indics.npz`)
* Entraîner sur suffisamment de données
* Evaluer sur une période **différente** de celle d’entraînement
* Commencer le live en **compte démo**
* Surveiller les métriques :

  * `Calmar30`
  * `max drawdown`
  * `winrate` vs payoff
  * comportement HOLD vs B/S

---

## 📚 Licence / Avertissement

Ce code est fourni à titre expérimental.
Le trading comporte des risques importants de perte en capital.
Utilisation à tes propres risques, surtout en compte réel.

---

```
```
