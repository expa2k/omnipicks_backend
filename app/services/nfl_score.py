"""
OmniPicks — NFL: del margen y el total previstos a probabilidades (ganador, spread y over/under).

En la NFL el margen final no se reparte como una campana suave: cae muchísimo más en 3 y en 7
(un gol de campo, un touchdown con su extra) y casi nunca en 0. Por eso no es lo mismo −2.5 que
−3.5. Se parte de una normal alrededor del margen previsto y se corrige con un peso por cada
margen exacto, aprendido de los partidos jugados:

    peso[|m|] = veces que terminó por |m| / veces que la normal esperaba |m|

Así, P(cubrir −3) y P(empate en la línea) salen como en la realidad.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

SUPPORT = np.arange(-70, 71)


def _normal_mass(mu: float, sigma: float) -> np.ndarray:
    return norm.cdf((SUPPORT + 0.5 - mu) / sigma) - norm.cdf((SUPPORT - 0.5 - mu) / sigma)


@dataclass
class MarginDistribution:
    sigma: float = 13.0
    weights: Dict[int, float] = field(default_factory=dict)     # |margen| → peso

    @classmethod
    def fit(cls, mu: np.ndarray, result: np.ndarray, sigma: float, smooth: float = 20.0) -> "MarginDistribution":
        """Aprende los pesos con partidos jugados: margen previsto (mu) y margen real (result)."""
        expected = np.zeros(len(SUPPORT))
        for m in mu:
            expected += _normal_mass(float(m), sigma)
        actual = np.zeros(len(SUPPORT))
        r = np.clip(np.round(result).astype(int), SUPPORT[0], SUPPORT[-1])
        np.add.at(actual, r - SUPPORT[0], 1.0)
        w = {}
        for k in range(0, 71):
            i, j = k - SUPPORT[0], -k - SUPPORT[0]
            e = expected[i] + (expected[j] if k else 0.0)
            a = actual[i] + (actual[j] if k else 0.0)
            w[k] = (a + smooth) / (e + smooth)      # encogido hacia 1 con `smooth` partidos virtuales
        return cls(sigma=sigma, weights=w)

    def pmf(self, mu: float) -> np.ndarray:
        p = _normal_mass(mu, self.sigma)
        if self.weights:
            p = p * np.array([self.weights.get(abs(int(k)), 1.0) for k in SUPPORT])
        return p / p.sum()

    def probs(self, mu: float, home_line: Optional[float] = None) -> Dict[str, float]:
        """Probabilidades del local. home_line: spread del local como lo da el casino (−3 = favorito por 3)."""
        p = self.pmf(mu)
        win, tie = float(p[SUPPORT > 0].sum()), float(p[SUPPORT == 0].sum())
        out = {"home_win": win + 0.5 * tie, "tie": tie}
        if home_line is not None:
            adj = SUPPORT + home_line
            out["home_cover"] = float(p[adj > 0].sum())
            out["push"] = float(p[np.isclose(adj, 0)].sum())
            out["away_cover"] = 1.0 - out["home_cover"] - out["push"]
        return out


def total_probs(mu_total: float, sigma_total: float, line: float) -> Dict[str, float]:
    """Over/under con una normal (en los totales los números clave pesan mucho menos)."""
    over = 1.0 - norm.cdf((line + 0.5 - mu_total) / sigma_total) if float(line).is_integer() \
        else 1.0 - norm.cdf((line - mu_total) / sigma_total)
    push = (norm.cdf((line + 0.5 - mu_total) / sigma_total) - norm.cdf((line - 0.5 - mu_total) / sigma_total)) \
        if float(line).is_integer() else 0.0
    return {"over": float(over), "under": float(1.0 - over - push), "push": float(push)}
