# NBA v2: backtest walk-forward

**Generado:** 2026-11-11 con `ml/scripts/nba_backtest.py` · **Periodo medido:** 2023-10-24 → 2026-11-09 (4,093 partidos, temporada regular y playoffs)

Cada partido se predice solo con información anterior a ese día. Las correcciones de contexto se reentrenan cada mes con los meses previos, y la calibración de cada temporada se ajusta con las temporadas anteriores. El mercado no entra al modelo; tampoco hay líneas históricas gratis de NBA para compararlo, así que esa comparación se hará hacia adelante.

**Bajas en el histórico:** no existen reportes de lesiones viejos, así que se usa lo que habría dicho un reporte: jugadores de rotación (≥15 min y presentes en ≥60% de los partidos recientes) que no jugaron, y quién regresaba. No se sabe qué suplentes se quedaron sin jugar.

## Resultado general

| modelo | n | logloss | Brier | acierto | MAE margen | MAE total |
|---|---|---|---|---|---|---|
| v2 completo | 4,093 | 0.596 | 0.205 | 67.6% | 11.03 | 14.81 |
| v2 sin bajas (no sabe quién falta) | 4,093 | 0.603 | 0.208 | 67.1% | 11.16 | 14.89 |
| v2 sin contexto | 4,093 | 0.597 | 0.206 | 67.5% | 11.05 | 14.96 |
| v2 sin calibración | 4,093 | 0.595 | 0.205 | 67.6% | 11.03 | 14.81 |
| v2 contexto con LightGBM | 4,093 | 0.599 | 0.206 | 67.0% | 11.09 | 14.81 |
| v2 + uso de los ausentes (3c) | 4,093 | 0.595 | 0.205 | 67.7% | 11.02 | 14.81 |
| v2 + uso absorbido por compañeros (D) | 4,093 | 0.595 | 0.205 | 67.6% | 11.02 | 14.81 |
| v2 minutos con LightGBM | 4,093 | 0.592 | 0.203 | 68.1% | 10.95 | 14.80 |
| v2 minutos con LightGBM + 3c | 4,093 | 0.591 | 0.203 | 67.9% | 10.96 | 14.80 |
| v2 minutos con LightGBM + D | 4,093 | 0.591 | 0.203 | 67.9% | 10.96 | 14.80 |
| v2 sabiendo exactamente quién juega (techo) | 4,093 | 0.592 | 0.203 | 68.1% | 11.01 | 14.78 |
| nba_possessions_v1 (actual) | 4,093 | 0.607 | 0.210 | 67.0% | 11.23 | 15.03 |
| Elo | 4,093 | 0.617 | 0.214 | 65.4% | 11.38 | nan |
| v2 ELEGIDO | 4,093 | 0.591 | 0.203 | 67.9% | 10.96 | 14.80 |

**v2 ELEGIDO** es el que va a producción: minutos LightGBM · contexto sí · ajuste por ausentes uso absorbido (D) · isotónica no. Cada pieza opcional entra solo si baja el log loss del backtest. "v2 completo" es la versión de referencia (contexto con ridge e isotónica, sin 3c); las demás filas le quitan o le cambian una pieza.

- **log loss y Brier**: calidad de la probabilidad de victoria (más bajo es mejor).
- **MAE**: error medio en puntos del margen y del total.

## Por temporada

| temporada · modelo | n | logloss | Brier | acierto | MAE margen | MAE total |
|---|---|---|---|---|---|---|
| 2023-24 · v2 ELEGIDO | 1,310 | 0.590 | 0.203 | 67.2% | 10.83 | 14.67 |
| 2024-25 · v2 ELEGIDO | 1,314 | 0.597 | 0.205 | 68.2% | 10.86 | 14.55 |
| 2025-26 · v2 ELEGIDO | 1,315 | 0.589 | 0.202 | 68.6% | 11.13 | 15.13 |
| 2026-27 · v2 ELEGIDO | 154 | 0.572 | 0.197 | 66.9% | 11.58 | 15.35 |
| 2023-24 · nba_possessions_v1 (actual) | 1,310 | 0.612 | 0.212 | 66.0% | 11.09 | 15.17 |
| 2024-25 · nba_possessions_v1 (actual) | 1,314 | 0.609 | 0.211 | 66.0% | 11.09 | 14.74 |
| 2025-26 · nba_possessions_v1 (actual) | 1,315 | 0.601 | 0.207 | 69.0% | 11.47 | 15.20 |
| 2026-27 · nba_possessions_v1 (actual) | 154 | 0.590 | 0.203 | 67.5% | 11.48 | 15.04 |
| 2023-24 · Elo | 1,310 | 0.620 | 0.216 | 64.3% | 11.22 | nan |
| 2024-25 · Elo | 1,314 | 0.620 | 0.216 | 64.6% | 11.24 | nan |
| 2025-26 · Elo | 1,315 | 0.610 | 0.211 | 67.3% | 11.60 | nan |
| 2026-27 · Elo | 154 | 0.614 | 0.214 | 64.3% | 12.08 | nan |

## Partidos con una baja importante (faltó alguien de ≥28 min por partido)

| modelo | n | logloss | Brier | acierto | MAE margen | MAE total |
|---|---|---|---|---|---|---|
| v2 sin bajas (no sabe quién falta) | 2,377 | 0.609 | 0.211 | 66.8% | 11.25 | 14.83 |
| nba_possessions_v1 (actual) | 2,377 | 0.613 | 0.212 | 66.6% | 11.33 | 14.83 |
| Elo | 2,377 | 0.624 | 0.217 | 65.0% | 11.49 | nan |
| v2 ELEGIDO | 2,377 | 0.594 | 0.204 | 68.2% | 11.01 | 14.69 |

## Calibración del modelo v2 (probabilidad del favorito)

| tramo | partidos | prob_media | gano_favorito |
|---|---|---|---|
| (0.499, 0.55] | 652 | 52.5% | 52.5% |
| (0.55, 0.6] | 631 | 57.5% | 53.6% |
| (0.6, 0.65] | 611 | 62.6% | 63.8% |
| (0.65, 0.7] | 555 | 67.5% | 71.0% |
| (0.7, 0.75] | 480 | 72.5% | 73.1% |
| (0.75, 0.8] | 431 | 77.5% | 76.8% |
| (0.8, 0.85] | 355 | 82.4% | 83.1% |
| (0.85, 1.0] | 378 | 89.2% | 89.9% |

## Parámetros entrenados con todo el histórico

σ del margen 13.78 pts · σ del total 18.50 pts · correlación entre ambos errores -0.00

Corrección de contexto (puntos por unidad de cada variable; margen = local − visita):

```
                   margen  total
variable                        
altitude_visit       0.54   1.88
b2b_a                1.71   0.00
b2b_h               -1.46   0.00
b2b_sum              0.00   0.12
early_season         0.68   0.41
east_a               1.28   0.00
late_season          0.00   1.03
playoffs             1.84  -9.42
rest_diff            0.15   0.00
three_in_four_a      0.12   0.00
three_in_four_h     -0.43   0.00
three_in_four_sum    0.00  -0.24
travel_a            -0.62   0.00
travel_h            -0.41   0.00
usage_gap           -0.09   0.00
west_a               0.51   0.00
```
