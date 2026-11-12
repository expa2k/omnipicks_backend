# OmniPicks: plan del modelo NFL

Hoy la predicción de la NFL es un relleno: el simulador usa `wind_speed` y `qb_diff`, que nadie calcula, así que siempre da **local por 2.5 y total de 45**. El objetivo es cambiarlo por un modelo que use datos reales, construido igual que el de la NBA.

**Regla:** el mercado no entra al modelo. La línea de cierre solo sirve para compararse.

## Fases

| Fase | Qué | Estado |
|---|---|---|
| 0 | Datos de nflverse: jugadas con EPA desde 2010, calendario con resultados, descanso, techo, clima, QB titular y líneas de cierre | ✅ |
| 1 | Rating de cada equipo: ataque y defensa en pase y en carrera, ajustados por rival, más un rating por marcador | ✅ |
| 2 | QB: rating propio de cada QB y cambio del titular cuando no juega | ✅ |
| 3 | Contexto: localía, descanso (bye, jueves), viaje y husos horarios, viento y frío | pendiente |
| 4 | De ratings a marcador: margen y total, con la distribución real del margen (números clave 3 y 7) | pendiente |
| 5 | Backtest completo semana a semana, contra el Elo y la línea de cierre (incluye % de spreads cubiertos) | pendiente |
| 6 | Producción: tareas semanales, predicción congelada, liquidación, historial y front | pendiente |

## Fase 0: datos

- `app/services/nfl_history.py`:
  - `load_pbp(seasons)`: solo jugadas de pase o carrera con EPA. Se guarda en caché en `ml/datasets/nfl/pbp_<año>.parquet` (unos 13 MB para 2010–2026); la temporada actual se vuelve a bajar con `refresh_current=True`.
  - `load_schedules()`: el calendario completo.
  - `team_games()`: una fila por equipo y partido, con EPA y número de jugadas en pase y en carrera. Quita las jugadas con el partido decidido (probabilidad de ganar fuera de 5–95%) y las de los últimos dos minutos de cada mitad.
  - Equipos que se mudaron: OAK→LV, SD→LAC, STL→LA.
- Librería: `nflreadpy==0.1.5`. Al 12-nov-2026 nflverse ya tenía la semana 9 de 2026 completa.

## Fase 1: ratings de equipo

`app/services/nfl_ratings.py`:

- **`NFLRatingEngine`**. Para pase y para carrera, con todo lo jugado antes de la semana a predecir:
  `EPA por jugada = media + localía + ataque[equipo] − defensa[rival]`
  - Mínimos cuadrados ponderados con ridge.
  - Cada partido pesa por sus jugadas y por su antigüedad.
  - La pausa entre temporadas cuenta como semanas extra.
- **`PointsRatingEngine`**: lo mismo con el marcador (tipo SRS), con el margen recortado a ±24.
- Parámetros elegidos:
  - EPA: vida media de 16 semanas, pausa de 12 y ridge de 300 jugadas.
  - Marcador: vida media de 16, pausa de 12 y ridge de 8 partidos.

### Resultado (`ml/scripts/nfl_ratings_eval.py`)

3,164 partidos de 2015 a 2026 (hasta la semana 9), caminando semana a semana. La conversión a puntos se ajusta solo con temporadas anteriores.

| Modelo | Error medio (pts) | RMSE | Brier | Acierto de ganador |
|---|---|---|---|---|
| Relleno actual (local +2.5) | 10.96 | 14.14 | 0.248 | 54.8% |
| Elo clásico | 10.19 | 13.12 | 0.222 | 64.1% |
| Ratings EPA ajustados por rival | 10.28 | 13.26 | 0.226 | 63.2% |
| Rating por marcador | 10.14 | 13.07 | 0.222 | 64.1% |
| EPA + marcador | 10.14 | 13.08 | 0.222 | 64.3% |
| **Mercado (línea de cierre)** | **9.79** | **12.69** | **0.213** | **66.2%** |

**Qué dice:**
- Con solo el rendimiento de los equipos, el error baja de 11.0 a ~10.1 puntos. Queda a la par del Elo, o un poco mejor.
- Separar pase y carrera no mejoró. Con el marcador ya incluido, el EPA tampoco agrega mucho.
- La distancia con el mercado (~0.35 puntos) es información de la semana que los ratings no tienen: sobre todo **quién juega de QB**, y luego las lesiones. Eso es la fase 2.
- Ojo: los parámetros se eligieron mirando el mismo periodo, así que el resultado es un poco optimista. El backtest de la fase 5 los va a elegir con temporadas anteriores.

## Fase 2: el QB

`app/services/nfl_qb.py`:

- **Rating del QB:** su EPA por dropback (pases, sacks y scrambles) por encima de la media de la liga.
  - Vida media de 64 semanas: la calidad de un QB cambia más despacio que la de un equipo.
  - Se encoge hacia un **nivel de suplente** (−0.10 EPA por dropback) con 100 dropbacks virtuales. Así, un QB nuevo o con pocas jugadas empieza como suplente.
- **Ajuste del partido:** el rating de pase del equipo ya trae dentro al QB que jugó esas semanas. Por eso el ajuste es la diferencia entre el titular de hoy y el QB medio con el que el equipo sumó sus números recientes, multiplicada por ~38 dropbacks.
  - Si juega el de siempre, el ajuste es ~0. Si entra el suplente, es negativo.
  - Además se usa la calidad del titular como segunda variable.
- **Titular:** en el backtest se usa el que reporta el calendario de nflverse. En producción saldrá del depth chart y del reporte de lesiones de ESPN (fase 6).

### Resultado (`ml/scripts/nfl_qb_eval.py`)

Los parámetros del QB se eligieron solo con 2015–2020. 2021–2026 queda como prueba limpia.

| Modelo | 2015–2020 | 2021–2026 (prueba limpia) | Con cambio de QB (1,019) |
|---|---|---|---|
| Elo clásico | 10.19 | 10.19 | 10.04 |
| Fase 1: EPA + marcador | 10.11 | 10.17 | 9.95 |
| **Fase 2: + QB titular** | **9.98** | **10.08** | **9.67** |
| Mercado (línea de cierre) | 9.85 | 9.74 | 9.43 |

Error medio del margen, en puntos. Con el QB, el Brier de la prueba limpia baja de 0.224 a 0.220.

**Qué dice:**
- El QB es la mejora más grande hasta ahora, sobre todo cuando cambia el titular: el error baja 0.28 puntos en esos partidos.
- En 2015–2020 el modelo queda a 0.13 puntos del mercado; en 2021–2026, a 0.34. El mercado mejoró en los últimos años y nosotros no tanto. Falta el contexto (fase 3) y la conversión a marcador (fase 4).
- Vidas medias más largas (128–256 semanas) mejoraban 2015–2020, pero empeoraban la prueba limpia, así que se quedó en 64.
- Ideas que quedan: ajustar el EPA del QB por la defensa rival y usar el % de pases completos sobre lo esperado (CPOE).
