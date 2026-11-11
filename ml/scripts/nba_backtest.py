"""
OmniPicks — NBA v2 (N5): backtest walk-forward del modelo completo y entrenamiento final.

Cada día se predice solo con lo que se sabía antes. Compara:
  · nba_players_v2 (ratings de jugador + bajas + contexto + calibración) y sus ablaciones
  · nba_possessions_v1 (el modelo actual, reconstruido con el histórico propio)
  · Elo con margen de victoria
El mercado no entra: no hay líneas históricas gratis de NBA (se medirá hacia adelante).

Bajas en el histórico: no hay reportes de lesiones viejos, así que se usa lo que habría
dicho un reporte: jugadores de rotación que no jugaron (y quién regresa).

Al final entrena con todo y guarda ml/saved_models/nba_v2.json (lo usa producción) y
escribe el reporte en docs/NBA_BACKTEST.md.

Uso:
    python ml/scripts/nba_backtest.py                 # backtest + entrenamiento + reporte
    python ml/scripts/nba_backtest.py --ablation      # además, sin RAPM y sin box score
"""
import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sqlalchemy import create_engine

from app.config import get_settings
from app.services.nba_game_model import (MARGIN_FEATURES, TOTAL_FEATURES, ModelParams, PaceState, ScheduleState,
                                         game_possessions, normal_cdf, structural, team_strength)
from app.services.nba_lineups import ROTATION_MIN, RosterTracker, expected_minutes
from app.services.nba_ratings import Prepared, RatingConfig, RatingEngine, load_data
from nba_ratings_eval import elo_predictions, team_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nba_backtest")

WARMUP_OUTPUT = date(2022, 12, 1)       # desde aquí se generan predicciones (para entrenar correcciones)
REPORT_FROM = date(2023, 10, 1)         # desde aquí se mide
SETTINGS = ("sin_info", "con_bajas", "sabe_quien")
KEY_MPG = 28.0
REPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "NBA_BACKTEST.md"))


# ══════════════════════════════════════════════════════════════════════
# nba_possessions_v1 reconstruido (mismas fórmulas que simulation_engine._simulate_nba)
# ══════════════════════════════════════════════════════════════════════

class V1State:
    K, HCA, B2B = 10.0, 2.5, 1.0

    def __init__(self):
        self.cur, self.prev, self.season = {}, {}, None

    def new_season(self, season):
        if season != self.season:
            if self.season is not None:
                self.prev = {t: self._raw(v) for t, v in self.cur.items()}
            self.cur, self.season = {}, season

    @staticmethod
    def _raw(v):
        return {"off": 100 * v["pf"] / v["poss"], "def": 100 * v["pa"] / v["poss"], "pace": v["poss"] / v["gp"]}

    def add(self, team, pf, pa, poss):
        v = self.cur.setdefault(team, {"pf": 0.0, "pa": 0.0, "poss": 0.0, "gp": 0})
        v["pf"] += pf
        v["pa"] += pa
        v["poss"] += poss
        v["gp"] += 1

    def league(self):
        src = self.cur if sum(v["gp"] for v in self.cur.values()) >= 60 else None
        if src:
            pf = sum(v["pf"] for v in src.values())
            poss = sum(v["poss"] for v in src.values())
            gp = sum(v["gp"] for v in src.values())
            return 100 * pf / poss, poss / gp
        if self.prev:
            return float(np.mean([v["off"] for v in self.prev.values()])), float(np.mean([v["pace"] for v in self.prev.values()]))
        return 114.0, 99.0

    def team(self, t, lo, lp):
        v = self.cur.get(t)
        gp = v["gp"] if v else 0
        raw = self._raw(v) if v else {"off": lo, "def": lo, "pace": lp}
        p = self.prev.get(t)
        out = {}
        for key, mean in (("off", lo), ("def", lo), ("pace", lp)):
            prior = mean if p is None else (2 * p[key] + mean) / 3
            out[key] = (raw[key] * gp + prior * self.K) / (gp + self.K)
        return out

    def predict(self, h, a, b2b_h, b2b_a):
        lo, lp = self.league()
        H, A = self.team(h, lo, lp), self.team(a, lo, lp)
        poss = H["pace"] * A["pace"] / lp
        hp = poss * (H["off"] * A["def"] / lo) / 100 + self.HCA / 2
        ap = poss * (A["off"] * H["def"] / lo) / 100 - self.HCA / 2
        if b2b_h:
            hp, ap = hp - self.B2B, ap + self.B2B / 2
        if b2b_a:
            ap, hp = ap - self.B2B, hp + self.B2B / 2
        return hp, ap


# ══════════════════════════════════════════════════════════════════════
# Recolección de predicciones estructurales, día por día
# ══════════════════════════════════════════════════════════════════════

def collect(prep: Prepared, cfg: RatingConfig) -> pd.DataFrame:
    actual, _ = team_logs(prep)
    box_by_game = {gid: g for gid, g in prep.box.groupby("game_id")}
    games_by_day = {d: g for d, g in prep.games.groupby("game_date")}
    day_of = {d: i for i, d in enumerate(prep.days)}
    games = prep.games
    sched = ScheduleState()
    for season, g in games.groupby("season"):
        reg = g[g.season_type == "Regular Season"]
        sched.season_start[season] = reg.game_date.min()
        sched.season_end[season] = reg.game_date.max()
    pace, v1, tracker = PaceState(), V1State(), RosterTracker()
    fed = [0]
    rows = []

    def feed_until(i):
        while fed[0] < i:
            d = prep.days[fed[0]]
            for g in games_by_day[d].itertuples():
                h, a = int(g.home_team_id), int(g.away_team_id)
                for tid in (h, a):
                    if (g.game_id, tid) in actual:
                        tracker.update(d, tid, *actual[(g.game_id, tid)])
                sched.add(d, h, a)
                poss = game_possessions(box_by_game.get(g.game_id, pd.DataFrame()))
                if poss:
                    pace.new_season(g.season)
                    pace.add(h, a, poss, g.home_pts + g.away_pts)
                    v1.new_season(g.season)
                    v1.add(h, g.home_pts, g.away_pts, poss)
                    v1.add(a, g.away_pts, g.home_pts, poss)
            fed[0] += 1

    def on_day(r):
        feed_until(day_of[r.day])
        for g in games_by_day[r.day].itertuples():
            h, a = int(g.home_team_id), int(g.away_team_id)
            pace.new_season(g.season)
            v1.new_season(g.season)
            ctx = sched.game(r.day, g.season, g.season_type, h, a)
            rec = {"game_id": g.game_id, "game_date": g.game_date, "season": g.season, "season_type": g.season_type,
                   "home": h, "away": a, "home_pts": g.home_pts, "away_pts": g.away_pts, **ctx,
                   "outs": 0, "key_out": False, "miss_min": 0.0}
            strengths = {s: {} for s in SETTINGS}
            ok = True
            for side, tid in (("h", h), ("a", a)):
                cands = tracker.candidates(tid, r.day)
                if (g.game_id, tid) not in actual or len(cands) < 8:
                    ok = False
                    break
                act_p, act_min = actual[(g.game_id, tid)]
                act = set(act_p.tolist())
                ids = np.array([c.player for c in cands])
                outs = {c.player: 0.0 for c in cands if c.player not in act and c.rate >= 0.6 and c.base >= ROTATION_MIN}
                back = {c.player for c in cands if c.player in act and c.rate <= 0.2 and c.base >= ROTATION_MIN}
                absent = {c.player: 0.0 for c in cands if c.player not in act}
                present = {c.player: 1.0 for c in cands if c.player in act}
                proj = {"sin_info": expected_minutes(cands)[0],
                        "con_bajas": expected_minutes(cands, outs, back)[0],
                        "sabe_quien": expected_minutes(cands, {**absent, **present})[0]}
                for s, m in proj.items():
                    strengths[s][side] = team_strength(m, r.o[ids], r.d[ids])
                sign = 1.0 if side == "h" else -1.0
                rec["outs"] += len(outs)
                rec["miss_min"] += sign * sum(c.base for c in cands if c.player in outs) / 10.0
                rec["key_out"] |= any(c.base >= KEY_MPG for c in cands if c.player in outs)
            if not ok:
                continue
            for s in SETTINGS:
                hp, ap = structural(strengths[s]["h"], strengths[s]["a"], pace.pace(h), pace.pace(a),
                                    pace.league_pace(), pace.league_ppp(), r.league_eff, r.home_edge)
                rec[f"{s}_hp"], rec[f"{s}_ap"] = hp, ap
            rec["v1_hp"], rec["v1_ap"] = v1.predict(h, a, ctx["b2b_h"], ctx["b2b_a"])
            rows.append(rec)

    RatingEngine(prep, cfg).run(first_output=WARMUP_OUTPUT, on_day=on_day)
    df = pd.DataFrame(rows)
    df["margin"] = df.home_pts - df.away_pts
    df["total"] = df.home_pts + df.away_pts
    elo = pd.Series(elo_predictions(games), index=games.game_id)
    df["elo_m"] = df.game_id.map(elo)
    return df


# ══════════════════════════════════════════════════════════════════════
# Correcciones walk-forward (por mes), σ y calibración
# ══════════════════════════════════════════════════════════════════════

def _fit_corr(train, base, feats_m, feats_t, method):
    rm = train.margin - (train[f"{base}_hp"] - train[f"{base}_ap"])
    rt = train.total - (train[f"{base}_hp"] + train[f"{base}_ap"])
    if method == "lgbm":
        import lightgbm as lgb
        kw = dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=60,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
        mm = lgb.LGBMRegressor(**kw).fit(train[feats_m], rm)
        mt = lgb.LGBMRegressor(**kw).fit(train[feats_t], rt)
    else:
        mm = Ridge(alpha=50.0).fit(train[feats_m], rm)
        mt = Ridge(alpha=50.0).fit(train[feats_t], rt)
    return mm, mt


def walk_forward(df: pd.DataFrame, base: str = "con_bajas", method: Optional[str] = "ridge",
                 calibrate: bool = True, feats_m=None) -> pd.DataFrame:
    """Predicciones corregidas fuera de muestra: cada mes se entrena con lo anterior."""
    feats_m = feats_m or MARGIN_FEATURES
    out = df[["game_id", "game_date", "season", "margin", "total", "key_out", "outs"]].copy()
    m = (df[f"{base}_hp"] - df[f"{base}_ap"]).to_numpy(float)
    t = (df[f"{base}_hp"] + df[f"{base}_ap"]).to_numpy(float)
    month = pd.to_datetime(df.game_date).dt.to_period("M")
    pm, pt = m.copy(), t.copy()
    if method:
        for mo in sorted(month[pd.to_datetime(df.game_date) >= pd.Timestamp(REPORT_FROM)].unique()):
            tr = (month < mo).to_numpy()
            te = (month == mo).to_numpy()
            if tr.sum() < 300:
                continue
            mm, mt = _fit_corr(df[tr], base, feats_m, TOTAL_FEATURES, method)
            pm[te] = m[te] + mm.predict(df.loc[te, feats_m])
            pt[te] = t[te] + mt.predict(df.loc[te, TOTAL_FEATURES])
    out["pred_m"], out["pred_t"] = pm, pt
    # σ del margen: RMSE de los errores anteriores (en el calentamiento se usa 13)
    err = (out.margin - out.pred_m) ** 2
    sig = np.sqrt(err.expanding().mean().shift(1)).fillna(13.0).clip(11.0, 16.0)
    out["p_raw"] = normal_cdf(out.pred_m / sig)
    out["p"] = out["p_raw"]
    if calibrate:
        for season in sorted(out.season.unique()):
            te = (out.season == season) & (pd.to_datetime(out.game_date) >= pd.Timestamp(REPORT_FROM))
            tr = pd.to_datetime(out.game_date) < pd.to_datetime(out.loc[te, "game_date"]).min() if te.any() else None
            if not te.any() or tr.sum() < 500:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(out.p_raw[tr], out.margin[tr] > 0)
            # Mezcla con la probabilidad cruda: la isotónica con pocos datos hace escalones bruscos
            out.loc[te, "p"] = 0.5 * iso.predict(out.p_raw[te]) + 0.5 * out.p_raw[te]
    return out


# ══════════════════════════════════════════════════════════════════════
# Métricas
# ══════════════════════════════════════════════════════════════════════

def metrics(d: pd.DataFrame) -> dict:
    p = d.p.clip(1e-4, 1 - 1e-4).to_numpy(float)
    win = (d.margin > 0).to_numpy(float)
    res = {
        "n": len(d),
        "logloss": float(-(win * np.log(p) + (1 - win) * np.log(1 - p)).mean()),
        "Brier": float(((p - win) ** 2).mean()),
        "acierto": float(((d.pred_m > 0) == (d.margin > 0)).mean()),
        "MAE margen": float((d.pred_m - d.margin).abs().mean()),
        "MAE total": float((d.pred_t - d.total).abs().mean()) if d.pred_t.notna().all() else float("nan"),
    }
    return res


def from_pts(df, hp, ap, label):
    out = df[["game_id", "game_date", "season", "margin", "total", "key_out", "outs"]].copy()
    out["pred_m"] = df[hp] - df[ap] if ap else df[hp]
    out["pred_t"] = df[hp] + df[ap] if ap else np.nan
    err = (out.margin - out.pred_m) ** 2
    sig = np.sqrt(err.expanding().mean().shift(1)).fillna(13.0).clip(11.0, 16.0)
    out["p"] = normal_cdf(out.pred_m / sig)
    out["modelo"] = label
    return out


def calibration_table(d: pd.DataFrame) -> pd.DataFrame:
    bins = np.array([0, .2, .3, .4, .5, .6, .7, .8, 1.0])
    p = np.maximum(d.p, 1 - d.p)                    # probabilidad del favorito
    fav_won = np.where(d.p >= 0.5, d.margin > 0, d.margin < 0)
    cut = pd.cut(p, [.5, .55, .6, .65, .7, .75, .8, .85, 1.0], include_lowest=True)
    t = pd.DataFrame({"tramo": cut, "p": p, "gano": fav_won}).groupby("tramo", observed=True).agg(
        partidos=("p", "size"), prob_media=("p", "mean"), gano_favorito=("gano", "mean"))
    return t


def fmt_table(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    lines = ["| " + " | ".join([t.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in t.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{int(v):,}" if c in ("n", "partidos") else (f"{v:.1%}" if c in ("acierto", "prob_media", "gano_favorito") else f"{v:.3f}" if c in ("logloss", "Brier") else f"{v:.2f}"))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Entrenamiento final (con todo) para producción
# ══════════════════════════════════════════════════════════════════════

def train_final(df: pd.DataFrame, metrics_summary: dict, feats_m, use_iso: bool) -> ModelParams:
    mm, mt = _fit_corr(df, "con_bajas", feats_m, TOTAL_FEATURES, "ridge")
    m = (df.con_bajas_hp - df.con_bajas_ap) + mm.predict(df[feats_m])
    t = (df.con_bajas_hp + df.con_bajas_ap) + mt.predict(df[TOTAL_FEATURES])
    em, et = df.margin - m, df.total - t
    sig_m = float(np.sqrt((em ** 2).mean()))
    p_raw = normal_cdf(m / sig_m)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(p_raw, df.margin > 0)
    grid = np.linspace(0.01, 0.99, 50)
    return ModelParams(
        margin_coef=dict(zip(feats_m, map(float, mm.coef_))), total_coef=dict(zip(TOTAL_FEATURES, map(float, mt.coef_))),
        margin_bias=float(mm.intercept_), total_bias=float(mt.intercept_),
        sigma_margin=sig_m, sigma_total=float(np.sqrt((et ** 2).mean())), rho=float(np.corrcoef(em, et)[0, 1]),
        iso_x=grid.tolist() if use_iso else [], iso_y=(0.5 * iso.predict(grid) + 0.5 * grid).tolist() if use_iso else [],
        trained_until=str(df.game_date.max()), metrics=metrics_summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    engine = create_engine(get_settings().mysql_url_sync)
    cfg = RatingConfig()
    data = load_data(engine)
    prep = Prepared(data, cfg)
    last_stint = prep.days[prep.last_stint_day]
    logger.info(f"{len(prep.games)} partidos; stints confiables de {prep.games_with_stints} "
                f"({prep.games_bad_stints} descartados); último día con stints {last_stint}")
    df = collect(prep, cfg)
    df = df[df.game_date <= last_stint].reset_index(drop=True)
    logger.info(f"{len(df)} partidos predichos en {time.time() - t0:.0f}s")

    variants = {
        "v2 completo": walk_forward(df),
        "v2 sin bajas (no sabe quién falta)": walk_forward(df, base="sin_info"),
        "v2 sin contexto": walk_forward(df, method=None),
        "v2 sin calibración": walk_forward(df, calibrate=False),
        "v2 contexto con LightGBM": walk_forward(df, method="lgbm"),
        "v2 + uso de los ausentes (3c)": walk_forward(df, feats_m=MARGIN_FEATURES + ["miss_min"]),
        "v2 sabiendo exactamente quién juega (techo)": walk_forward(df, base="sabe_quien"),
        "nba_possessions_v1 (actual)": from_pts(df, "v1_hp", "v1_ap", "v1"),
        "Elo": from_pts(df, "elo_m", None, "elo"),
    }
    if args.ablation:
        for label, c in (("v2 sin RAPM (solo box score)", replace(cfg, use_rapm=False)),
                         ("v2 sin box score (solo RAPM)", replace(cfg, use_box=False))):
            d2 = collect(Prepared(data, c), c)
            d2 = d2[d2.game_date <= last_stint].reset_index(drop=True)
            variants[label] = walk_forward(d2)

    rep = lambda d: d[pd.to_datetime(d.game_date) >= pd.Timestamp(REPORT_FROM)]
    ll = lambda k: metrics(rep(variants[k]))["logloss"]
    # Cada pieza opcional entra solo si mejora el log loss del backtest
    use_3c = ll("v2 + uso de los ausentes (3c)") < ll("v2 completo")
    use_iso = ll("v2 completo") < ll("v2 sin calibración")
    use_ctx = ll("v2 completo") < ll("v2 sin contexto")
    feats_m = MARGIN_FEATURES + (["miss_min"] if use_3c else [])
    variants["v2 ELEGIDO"] = walk_forward(df, method="ridge" if use_ctx else None, calibrate=use_iso, feats_m=feats_m)
    choice = f"contexto {'sí' if use_ctx else 'no'} · uso de ausentes (3c) {'sí' if use_3c else 'no'} · isotónica {'sí' if use_iso else 'no'}"
    logger.info(f"Piezas elegidas: {choice}")
    summary = pd.DataFrame({k: metrics(rep(v)) for k, v in variants.items()}).T
    summary.index.name = "modelo"
    print("\n" + summary.to_string(float_format=lambda v: f"{v:.4f}"))
    by_season = {}
    for k in ("v2 ELEGIDO", "nba_possessions_v1 (actual)", "Elo"):
        d = rep(variants[k])
        for s, g in d.groupby("season"):
            by_season[(s, k)] = metrics(g)
    bs = pd.DataFrame(by_season).T
    bs.index = [f"{s} · {k}" for s, k in bs.index]
    bs.index.name = "temporada · modelo"
    key = pd.DataFrame({k: metrics(rep(v)[rep(v).key_out]) for k, v in variants.items()
                        if k in ("v2 ELEGIDO", "v2 sin bajas (no sabe quién falta)", "nba_possessions_v1 (actual)", "Elo")}).T
    key.index.name = "modelo"
    cal = calibration_table(rep(variants["v2 ELEGIDO"]))
    print("\n" + bs.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nCon baja importante:\n" + key.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nCalibración:\n" + cal.to_string())

    if args.no_save:
        return
    v2 = metrics(rep(variants["v2 ELEGIDO"]))
    params = train_final(df, {**{k: round(v, 4) for k, v in v2.items()}, "piezas": choice}, feats_m, use_iso)
    if not use_ctx:
        params.margin_coef, params.total_coef, params.margin_bias, params.total_bias = {}, {}, 0.0, 0.0
    params.save()
    logger.info(f"Parámetros guardados (σ margen {params.sigma_margin:.2f}, σ total {params.sigma_total:.2f}, ρ {params.rho:.2f})")
    coefs = pd.DataFrame({"margen": pd.Series(params.margin_coef), "total": pd.Series(params.total_coef)})
    coefs.index.name = "variable"

    d0 = rep(variants["v2 ELEGIDO"])
    with open(REPORT_PATH, "w") as f:
        f.write(f"""# NBA v2: backtest walk-forward

**Generado:** {date.today()} con `ml/scripts/nba_backtest.py` · **Periodo medido:** {d0.game_date.min()} → {d0.game_date.max()} ({len(d0):,} partidos, temporada regular y playoffs)

Cada partido se predice solo con información anterior a ese día. Las correcciones de contexto se reentrenan cada mes con los meses previos, y la calibración de cada temporada se ajusta con las temporadas anteriores. El mercado no entra al modelo; tampoco hay líneas históricas gratis de NBA para compararlo, así que esa comparación se hará hacia adelante.

**Bajas en el histórico:** no existen reportes de lesiones viejos, así que se usa lo que habría dicho un reporte: jugadores de rotación (≥15 min y presentes en ≥60% de los partidos recientes) que no jugaron, y quién regresaba. No se sabe qué suplentes se quedaron sin jugar.

## Resultado general

{fmt_table(summary)}

**v2 ELEGIDO** es el que va a producción: {choice}. Cada pieza opcional entra solo si baja el log loss del backtest. "v2 completo" es la versión de referencia (contexto con ridge e isotónica, sin 3c); las demás filas le quitan o le cambian una pieza.

- **log loss y Brier**: calidad de la probabilidad de victoria (más bajo es mejor).
- **MAE**: error medio en puntos del margen y del total.

## Por temporada

{fmt_table(bs)}

## Partidos con una baja importante (faltó alguien de ≥{KEY_MPG:.0f} min por partido)

{fmt_table(key)}

## Calibración del modelo v2 (probabilidad del favorito)

{fmt_table(cal)}

## Parámetros entrenados con todo el histórico

σ del margen {params.sigma_margin:.2f} pts · σ del total {params.sigma_total:.2f} pts · correlación entre ambos errores {params.rho:.2f}

Corrección de contexto (puntos por unidad de cada variable; margen = local − visita):

```
{coefs.fillna(0).round(2).to_string()}
```
""")
    logger.info(f"Reporte en {REPORT_PATH} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
