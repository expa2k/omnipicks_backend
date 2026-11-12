# OmniPicks: plan del modelo NFL

Hoy la predicción de la NFL es un relleno: el simulador usa `wind_speed` y `qb_diff`, que nadie calcula, así que siempre da **local por 2.5 y total de 45**. El objetivo es cambiarlo por un modelo que use datos reales, construido igual que el de la NBA.

**Regla:** el mercado no entra al modelo. La línea de cierre solo sirve para compararse.

## Fases

| Fase | Qué | Estado |
|---|---|---|
| 0 | Datos de nflverse: jugadas con EPA desde 2010, calendario con resultados, descanso, techo, clima, QB titular y líneas de cierre | ✅ |
| 1 | Rating de cada equipo: ataque y defensa en pase y en carrera, ajustados por rival, más un rating por marcador | ✅ |
| 2 | QB: rating propio de cada QB y cambio del titular cuando no juega | ✅ |
| 3 | Contexto: localía, descanso (bye, jueves), viaje y husos horarios, división | ✅ probado, no mejora el margen |
| 4 | De ratings a marcador: margen y total, con la distribución real del margen (números clave 3 y 7) | ✅ |
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

## Fase 3: contexto del partido

`app/services/nfl_context.py`: variables que se saben antes del partido, todas del lado del local.
- Localía (0 en sede neutral).
- Diferencia de descanso, semana libre (bye) y semana corta (jueves) de cada equipo.
- Kilómetros de viaje del visitante, husos horarios que cruzó y si juega antes de las 11 a. m. de su reloj (un equipo del oeste a la 1 p. m. ET).
- Partido divisional, 2020 sin público y playoffs.
- Considera los estadios anteriores de Raiders (Oakland), Chargers (San Diego) y Rams (St. Louis).

### Resultado (`ml/scripts/nfl_context_eval.py`)

| Modelo | 2015–2020 | 2021–2026 (prueba limpia) |
|---|---|---|
| Fase 2 (EPA + marcador + QB) | 9.98 | 10.08 |
| Fase 3: + contexto | 9.98 | 10.08 |
| Mercado | 9.85 | 9.74 |

**El contexto no mejora el margen.** Los efectos son chicos, de 0.1 a 0.7 puntos, y ruidosos: por ejemplo, la semana libre sale negativa para los dos equipos. Además, la localía ya está dentro de los ratings: tanto el de EPA como el de marcador aprenden su propio término de local, que se va ajustando con los años. Dar más peso a las temporadas recientes al convertir a puntos tampoco ayudó.

**Decisión:** el margen se queda con la fase 2. El módulo se queda para la fase 4, donde el clima (viento y frío) sí debería pesar en el total.

### Dónde está la distancia con el mercado

Error medio del modelo de la fase 2 contra la línea de cierre, de 2015 a 2026:

| Semanas | Partidos | Modelo | Mercado | Diferencia |
|---|---|---|---|---|
| 1–4 | 762 | 10.00 | 9.85 | +0.15 |
| 5–9 | 854 | 9.97 | 9.73 | +0.24 |
| 10–14 | 808 | 10.01 | 9.77 | +0.24 |
| 15–18 | 633 | 10.15 | 9.86 | +0.29 |
| Playoffs | 107 | 10.13 | 9.69 | +0.44 |

La distancia es pareja todo el año, así que no es un problema del arranque de temporada. Es información de la semana que el mercado tiene y el modelo no:
- lesiones fuera del QB;
- titulares que descansan al final de la temporada;
- equipos ya clasificados o eliminados.

Ideas para después de la fase 4: bajas por grupo de posición (línea ofensiva, receptores, secundaria), con las que el backtest diga que ayudan, y una marca para "ya clasificado o eliminado" en las semanas 17–18.

## Fase 4: total, marcador y probabilidades

### Total de puntos

- `ScoringRatingEngine` (en `nfl_ratings.py`): por equipo y partido, `puntos = media + localía + ataque − defensa rival`, ajustado por rival y con descuento por antigüedad. Elegidos: vida media de 8 semanas, pausa de 12 y ridge de 8.
- Además: EPA esperado de los dos ataques, ritmo (jugadas por partido de cada equipo) y clima.

| Modelo | 2015–2020 | 2021–2026 (prueba limpia) |
|---|---|---|
| Media de la liga | 11.09 | 10.94 |
| Rating de anotación | 10.88 | 10.72 |
| + EPA de los ataques | 10.82 | 10.68 |
| + ritmo | 10.83 | 10.67 |
| **+ clima (modelo completo)** | **10.84** | **10.61** |
| Mercado (total de cierre) | 10.56 | 10.34 |

Clima: el techo suma ~2 puntos, cada mph de viento por encima de 10 resta ~0.34, y el frío casi no importa. La desviación del total alrededor del modelo es de 13.6 puntos.

### Margen con números clave (`app/services/nfl_score.py`)

- `MarginDistribution`: una normal alrededor del margen previsto (σ ≈ 12.9), corregida con un peso por cada margen exacto, aprendido de los partidos jugados. El margen termina en **3 unas 2.5 veces** más de lo que diría la normal, en 7 1.7 veces, y casi nunca en empate (0.29).
  - Pesos: 0: 0.29 · 1: 0.81 · 2: 0.84 · **3: 2.54** · 4: 0.88 · 6: 1.33 · **7: 1.68** · 8: 0.92 · 10: 1.07 · 14: 1.41 · 17: 1.26.
- `probs(mu, línea)` da la probabilidad de ganar, de cubrir, de push y de que cubra el rival. `total_probs` hace lo mismo con el over/under.
- **Margen exacto** (log-loss en 2021–2026): normal 3.980 → con números clave **3.853**.
- **Push**: con línea de 3 prevé 7.0% (real 10.3%); con línea de 7, 5.1% (real 6.0%); con otras líneas enteras, 2.9% (real 2.5%). Con 3 se queda corto: se podría afinar aprendiendo los pesos según el tamaño del margen previsto.

### Hallazgo clave: contra la línea de cierre el modelo no tiene ventaja

- La probabilidad de cubrir que sale directo del modelo es **peor que decir 50% siempre**: Brier 0.2585, cuando decir siempre 50% da 0.2500. Cuando el modelo difiere de la línea, la mayoría de las veces el que acierta es el mercado.
- Calibrándola con temporadas anteriores, `P = Φ(k·z)`, el factor **k queda en 0.1–0.25 y bajando**: cada punto de diferencia con la línea vale muy poco. El Brier calibrado queda en 0.2504, prácticamente una moneda.
- Con la probabilidad calibrada solo 125 partidos de 2021–2026 pasan de 53% (acierto 54.4%), y 23 pasan de 55% (acierto 47.8%). No es una ventaja medible.

**Qué significa para la app:**
- El modelo sirve para **pronosticar**: ganador, margen, total y probabilidades realistas. Ya es mejor que el Elo y mucho mejor que el relleno.
- **No va a encontrar valor seguido** contra la línea de cierre. Lo honesto es mostrar casi todo como "sin valor".
- El backtest de la fase 5 tiene que confirmar esto con apuestas simuladas, y decidir si algún umbral sirve, por ejemplo contra líneas de apertura cuando tengamos las de ESPN.
