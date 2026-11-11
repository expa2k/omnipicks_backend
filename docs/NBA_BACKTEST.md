# NBA v2: backtest walk-forward

**Generado:** 2026-11-11 con `ml/scripts/nba_backtest.py` · **Periodo medido:** 2023-10-24 → 2026-11-10 (4,099 partidos, temporada regular y playoffs)

Cada partido se predice solo con información anterior a ese día. Las correcciones de contexto se reentrenan cada mes con los meses previos, y la calibración de cada temporada se ajusta con las temporadas anteriores. El mercado no entra al modelo; tampoco hay líneas históricas gratis de NBA para compararlo, así que esa comparación se hará hacia adelante.

**Bajas en el histórico:** no existen reportes de lesiones viejos, así que se usa lo que habría dicho un reporte: jugadores de rotación (≥15 min y presentes en ≥60% de los partidos recientes) que no jugaron, y quién regresaba. No se sabe qué suplentes se quedaron sin jugar.

## Resultado general

| modelo | n | logloss | Brier | acierto | MAE margen | MAE total |
|---|---|---|---|---|---|---|
| v2 completo | 4,099 | 0.596 | 0.205 | 67.6% | 11.03 | 14.81 |
| v2 sin bajas (no sabe quién falta) | 4,099 | 0.603 | 0.208 | 67.2% | 11.16 | 14.89 |
| v2 sin contexto | 4,099 | 0.597 | 0.206 | 67.5% | 11.05 | 14.95 |
| v2 sin calibración | 4,099 | 0.595 | 0.205 | 67.6% | 11.03 | 14.81 |
| v2 contexto con LightGBM | 4,099 | 0.599 | 0.206 | 67.1% | 11.09 | 14.81 |
| v2 + uso de los ausentes (3c) | 4,099 | 0.594 | 0.205 | 67.7% | 11.02 | 14.81 |
| v2 + uso absorbido por compañeros (D) | 4,099 | 0.594 | 0.205 | 67.6% | 11.02 | 14.81 |
| v2 minutos con LightGBM | 4,099 | 0.591 | 0.203 | 68.1% | 10.96 | 14.80 |
| v2 minutos con LightGBM + 3c | 4,099 | 0.591 | 0.203 | 67.9% | 10.97 | 14.80 |
| v2 minutos con LightGBM + D | 4,099 | 0.591 | 0.203 | 68.0% | 10.97 | 14.80 |
| v2 minutos con LightGBM + D + ritmo por jugadores | 4,099 | 0.591 | 0.203 | 68.0% | 10.97 | 14.78 |
| v2 sabiendo exactamente quién juega (techo) | 4,099 | 0.591 | 0.203 | 68.1% | 11.01 | 14.78 |
| nba_possessions_v1 (actual) | 4,099 | 0.607 | 0.210 | 67.0% | 11.23 | 15.03 |
| Elo | 4,099 | 0.617 | 0.214 | 65.4% | 11.39 | nan |
| v2 ELEGIDO | 4,099 | 0.591 | 0.203 | 68.0% | 10.97 | 14.78 |

**v2 ELEGIDO** es el que va a producción: minutos LightGBM · contexto sí · ajuste por ausentes uso absorbido (D) · ritmo por jugadores sí · isotónica no. Cada pieza opcional entra solo si baja el log loss del backtest. "v2 completo" es la versión de referencia (minutos proporcionales, contexto con ridge e isotónica); las demás filas le quitan, le cambian o le agregan una pieza. "Minutos con LightGBM" usa el modelo de `ml/scripts/nba_minutes_model.py`, entrenado solo con temporadas anteriores; "3c" suma los minutos de rotación ausentes y "D" el uso (tiros y pérdidas) que dejan libre.

- **log loss y Brier**: calidad de la probabilidad de victoria (más bajo es mejor).
- **MAE**: error medio en puntos del margen y del total.

## Por temporada

| temporada · modelo | n | logloss | Brier | acierto | MAE margen | MAE total |
|---|---|---|---|---|---|---|
| 2023-24 · v2 ELEGIDO | 1,310 | 0.590 | 0.203 | 67.2% | 10.83 | 14.66 |
| 2024-25 · v2 ELEGIDO | 1,314 | 0.597 | 0.205 | 68.2% | 10.86 | 14.52 |
| 2025-26 · v2 ELEGIDO | 1,315 | 0.589 | 0.202 | 68.6% | 11.13 | 15.11 |
| 2026-27 · v2 ELEGIDO | 160 | 0.568 | 0.196 | 67.5% | 11.64 | 15.17 |
| 2023-24 · nba_possessions_v1 (actual) | 1,310 | 0.612 | 0.212 | 66.0% | 11.09 | 15.17 |
| 2024-25 · nba_possessions_v1 (actual) | 1,314 | 0.609 | 0.211 | 66.0% | 11.09 | 14.74 |
| 2025-26 · nba_possessions_v1 (actual) | 1,315 | 0.601 | 0.207 | 69.0% | 11.47 | 15.20 |
| 2026-27 · nba_possessions_v1 (actual) | 160 | 0.589 | 0.202 | 67.5% | 11.59 | 14.98 |
| 2023-24 · Elo | 1,310 | 0.620 | 0.216 | 64.3% | 11.22 | nan |
| 2024-25 · Elo | 1,314 | 0.620 | 0.216 | 64.6% | 11.24 | nan |
| 2025-26 · Elo | 1,315 | 0.610 | 0.211 | 67.3% | 11.60 | nan |
| 2026-27 · Elo | 160 | 0.612 | 0.213 | 64.4% | 12.17 | nan |

## Partidos con una baja importante (faltó alguien de ≥28 min por partido)

| modelo | n | logloss | Brier | acierto | MAE margen | MAE total |
|---|---|---|---|---|---|---|
| v2 sin bajas (no sabe quién falta) | 2,380 | 0.608 | 0.211 | 66.9% | 11.25 | 14.84 |
| nba_possessions_v1 (actual) | 2,380 | 0.612 | 0.212 | 66.6% | 11.33 | 14.83 |
| Elo | 2,380 | 0.624 | 0.217 | 65.0% | 11.48 | nan |
| v2 ELEGIDO | 2,380 | 0.594 | 0.204 | 68.3% | 11.00 | 14.67 |

## Calibración del modelo v2 (probabilidad del favorito)

| tramo | partidos | prob_media | gano_favorito |
|---|---|---|---|
| (0.499, 0.55] | 652 | 52.5% | 52.5% |
| (0.55, 0.6] | 632 | 57.5% | 53.6% |
| (0.6, 0.65] | 613 | 62.6% | 63.8% |
| (0.65, 0.7] | 556 | 67.5% | 71.0% |
| (0.7, 0.75] | 481 | 72.5% | 73.2% |
| (0.75, 0.8] | 431 | 77.5% | 76.8% |
| (0.8, 0.85] | 355 | 82.4% | 83.1% |
| (0.85, 1.0] | 379 | 89.2% | 90.0% |

## Parámetros entrenados con todo el histórico

σ del margen 13.79 pts · σ del total 18.48 pts · correlación entre ambos errores -0.00

Corrección de contexto (puntos por unidad de cada variable; margen = local − visita):

```
                   margen  total
variable                        
altitude_visit       0.53   1.86
b2b_a                1.69   0.00
b2b_h               -1.43   0.00
b2b_sum              0.00   0.14
early_season         0.67   0.34
east_a               1.30   0.00
late_season          0.00   1.14
pace_delta           0.00   3.11
playoffs             1.83  -9.38
rest_diff            0.16   0.00
three_in_four_a      0.10   0.00
three_in_four_h     -0.42   0.00
three_in_four_sum    0.00  -0.22
travel_a            -0.63   0.00
travel_h            -0.42   0.00
usage_gap           -0.09   0.00
west_a               0.54   0.00
```
