# OmniPicks: plan del modelo NFL

Antes, la predicción de la NFL era un relleno: el simulador usa `wind_speed` y `qb_diff`, que nadie calcula, así que siempre daba **local por 2.5 y total de 45**. Desde la fase 6 la app usa `nfl_v2`. El objetivo era cambiarlo por un modelo que use datos reales, construido igual que el de la NBA.

**Regla:** el mercado no entra al modelo. La línea de cierre solo sirve para compararse.

## Fases

| Fase | Qué | Estado |
|---|---|---|
| 0 | Datos de nflverse: jugadas con EPA desde 2010, calendario con resultados, descanso, techo, clima, QB titular y líneas de cierre | ✅ |
| 1 | Rating de cada equipo: ataque y defensa en pase y en carrera, ajustados por rival, más un rating por marcador | ✅ |
| 2 | QB: rating propio de cada QB y cambio del titular cuando no juega | ✅ |
| 3 | Contexto: localía, descanso (bye, jueves), viaje y husos horarios, división | ✅ probado, no mejora el margen |
| 4 | De ratings a marcador: margen y total, con la distribución real del margen (números clave 3 y 7) | ✅ |
| 5 | Backtest de apuestas semana a semana con momios reales de cierre | ✅ sin ventaja |
| 6 | Producción: tareas semanales, predicción congelada, liquidación, historial y front | ✅ |

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

## Fase 5: backtest de apuestas (`ml/scripts/nfl_backtest.py`)

Semana a semana, solo con lo que se sabía antes de cada partido:
- **Modelo:** margen de la fase 2 y total de la fase 4.
- **Probabilidades:** con números clave, y cada una calibrada con las temporadas anteriores: `P = Φ(k·z)`.
- **Apuestas:** 1 unidad con los **momios reales de cierre** (nflverse), cuando el valor esperado (probabilidad × momio decimal − 1) pasa un umbral.

Se reporta 2017–2020 y 2021–2026 por separado; calibrar necesita al menos 500 partidos previos.

### Calibración

| Mercado | k (1 = el modelo crudo está bien) | Brier 2021–26 calibrado | Referencia |
|---|---|---|---|
| Spread | 0.26 → 0.09 (bajando cada año) | 0.2504 | 0.2500 = decir 50% |
| Total | **0.00** | 0.2500 | 0.2500 = decir 50% |
| Ganador | ~1.1 | 0.2199 | mercado sin margen: **0.2118** |

- En el **total**, la diferencia del modelo con la línea no aporta nada: la calibración la apaga por completo.
- En el **spread**, aporta casi nada, y cada año menos.
- En el **ganador**, el modelo está bien calibrado por sí solo, pero el mercado predice mejor (0.212 contra 0.220). Por eso el "valor" que encuentra en moneyline es ficticio.

### Apuestas, prueba limpia 2021–2026

| Regla | Apuestas | Récord | Unidades | ROI (±95%) |
|---|---|---|---|---|
| Spread, valor > 0% | 515 | 240-261 | −25.3 | −4.9% (±8.4%) |
| Spread, valor > 4% | 129 | 57-67 | −10.1 | −7.9% (±16.9%) |
| Spread, valor > 6% | 66 | 32-31 | +1.3 | +1.9% (±23.9%) |
| Total, valor > 0% | 153 | 66-86 | −19.1 | −12.5% (±15.8%) |
| Ganador, valor > 0% | 1,259 | 538-716 | −83.1 | −6.6% (±6.8%) |
| Spread, el modelo difiere ≥ 3 pts (tipo NBA) | 408 | 197-203 | −21.2 | −5.2% (±9.3%) |
| Total, el modelo difiere ≥ 3 pts | 407 | 190-212 | −37.7 | −9.3% (±9.3%) |
| Referencia: siempre el local en el spread | 1,560 | 752-769 | −73.9 | −4.7% (±4.7%) |

En 2017–2020 algunas reglas salen positivas, como el spread con diferencia ≥ 1.5: +4.5% (±7.7%). Pero en 2021–2026 se dan la vuelta, y todas caen dentro del ruido.

### Conclusión

- **No hay ventaja medible en ningún mercado de la NFL** contra la línea de cierre. Casi todas las reglas pierden más o menos lo que cobra el casino (~4.5%). La única positiva (+1.9% en 66 apuestas) es ruido.
- La regla de la NBA ("el modelo difiere de la línea en 3 pts o más") en la NFL **pierde**: −5.2% en el spread y −9.3% en el total.
- El modelo sí es bueno para **pronosticar**: es mejor que el Elo, tiene probabilidades de ganar bien calibradas y números clave reales.

**Para producción (fase 6):**
- Mostrar el pronóstico y las probabilidades.
- El valor contra el casino, con la probabilidad calibrada. En la práctica casi todo saldrá "sin valor".
- No recomendar moneyline con la probabilidad del modelo.
- Guardar las líneas de apertura de ESPN para, más adelante, medir si hay ventaja antes de que la línea se mueva. Ahí es donde suele estar, si es que hay.

## Fase 6: producción

`app/services/nfl_v2.py`:
- **`train()`**: baja la temporada actual de nflverse, recorre todo el histórico semana a semana como el backtest (~15 s) y guarda en `ml/saved_models/nfl_v2.json`:
  - los coeficientes del margen y del total;
  - σ del margen y del total;
  - los pesos de los números clave;
  - la calibración `k` de spread, total y ganador, ajustada con las últimas 4 temporadas.

  Ese archivo no se versiona porque cambia cada día. Se reentrena solo cuando tiene más de 20 h.
- **`predict_games_sync()`**:
  - Empareja cada partido de la base (nombres de ESPN) con el de nflverse por abreviatura y fecha (±1 día).
  - **Vuelve a bajar el calendario en cada corrida**, para tomar los QB titulares actualizados.
  - Devuelve probabilidad, marcador, margen y total, y en `simulation_details` los QB (rating, dropbacks, ajuste), los ratings, la localía, el clima y la calibración.
- **`market_probs` / `nfl_picks`**: lo que prefiere el modelo en cada mercado, con el mismo formato que la NBA, pero con números clave y probabilidades calibradas. **El ganador nunca se recomienda.**

**Conexiones:**
- `prediction_engine`:
  - La NFL usa `nfl_v2`; si falla, queda el simulador viejo (`NFL_MODEL=v1` lo fuerza).
  - La predicción se congela antes del partido en `nba_prediction_log` con `model_name = "nfl_v2"`. La tabla es compartida; la liquidación es la misma (marcador final de ESPN).
  - Los picks de la NFL solo salen de spread o total con nivel "recomendado", nunca de moneyline. Con la calibración actual no habrá casi ninguno.
- `routes_predictions`: `model_picks`, la predicción congelada y el resultado funcionan para la NFL igual que para la NBA.
- **Scheduler:** `job_nfl` cada hora (ESPN: partidos, marcadores y líneas → predicciones). El reentrenamiento diario va dentro de la predicción.
- **Front** (tarjeta del partido):
  - "Por qué este pronóstico" con los QB titulares, su rating, el cambio contra su QB habitual, los ratings de EPA y de marcador, la localía y el clima.
  - "Rango del pronóstico" (80%).
  - Una nota de que en la NFL el modelo es para pronosticar, no para apostar.
- **Logos:** el scoreboard de ESPN trae `logo` (no `logos`); se llenaron los 32.

**Primer entrenamiento (12-nov-2026, hasta la semana 9):** 3,698 partidos fuera de muestra, MAE del margen 10.09 (mercado 9.90), acierto de ganador 65.4%. `k` spread = 0.00, `k` total = 0.00, `k` ganador = 1.05.

Con las últimas 4 temporadas, la diferencia con la línea no predijo nada, así que el spread y el total salen en 50% ("sin valor"). La app muestra hacia dónde se inclina el modelo, sin recomendarlo.

### QB titular con las lesiones de ESPN (`app/services/nfl_qb_status.py`)

nflverse publica un titular proyectado por partido, pero puede ir atrasado. El 12-nov tenía a Darnold (SEA), que ESPN da Out, y a Huntley (BAL), que está en IR. Por eso, en cada corrida:
1. Se bajan las lesiones de la NFL de ESPN (`/injuries`) y el depth chart de los QB de cada equipo (core API, `/depthcharts`).
2. Candidatos, en orden: el titular de nflverse y luego el depth chart. Se descartan Out, IR, suspendidos, PUP y NFI.
3. El primero que queda es el titular. Si está **Questionable** o **Doubtful**, las variables del QB se promedian con las de su suplente, según la probabilidad de que juegue.
   - Si no hay suplente identificado, se usa uno genérico (nivel de suplente).
   - **Probabilidad aprendida** (`learn_play_probs`, se recalcula en cada entrenamiento). Viene de los reportes oficiales de lesiones de nflverse de 2015–2025, solo con el QB que fue titular el partido anterior:

     | Estado en el reporte final | Casos | Fue titular |
     |---|---|---|
     | Questionable, entrenó completo | 37 | **88%** |
     | Questionable, entrenó limitado | 129 | **54%** |
     | Questionable, no entrenó | 22 | **48%** |
     | Questionable (sin dato de práctica) | 188 | **60%** |
     | Doubtful | 27 | **7%** |

     Antes se usaba 80% y 20%: el Questionable estaba sobreestimado.
   - **Práctica de la semana:** nflverse la publica a diario (`load_injuries` de la temporada actual), y el estado oficial en cuanto sale el reporte final. Si todavía no hay estado oficial, se usa el de ESPN.
4. Los ids de ESPN se pasan a los de nflverse con la tabla de jugadores (`load_players`, en caché diaria).
5. Si ESPN no responde (con 3 intentos), se usa el titular de nflverse.

En la app, "Por qué este pronóstico" muestra el estado de cada QB y notas como "Sam Darnold (SEA) está fuera: no se cuenta" o "J.J. McCarthy está en duda: el pronóstico supone 80% que juega y 20% Jameis Winston".

**Deadlock corregido:** las corridas de predicción de la NBA y de la NFL arrancaban a la vez y chocaban en MySQL. Un deadlock deshacía el borrado previo y dejaba predicciones duplicadas.
- Ahora van una por una (`_predict_lock` en el scheduler).
- Si el registro falla por un error de base de datos, la corrida se aborta en vez de seguir a medias.

### Clima: lluvia y pronóstico de ESPN

- **Histórico:** las jugadas de nflverse traen el clima observado en texto ("Light Snow Temp: 36° F…"). `game_weather()` marca lluvia o nieve al aire libre: 240 partidos con lluvia y 45 con nieve desde 2010.
- **Efecto** (out-of-sample, 2013–2026):
  - La **lluvia resta ~3.4 puntos** del total. En los 189 partidos con lluvia, el error baja de **10.93 a 10.44**; el mercado está en 10.36. En esos partidos la línea también se queda alta: 43.5 contra 40.9 real.
  - La **nieve** (39 partidos) empeoraba el error (13.9), así que no entra.
- **Partidos por jugar:** el resumen del partido de ESPN (`/summary?event=…`, AccuWeather) da la probabilidad de lluvia, las ráfagas y la temperatura.
  - La probabilidad de lluvia entra como "lluvia esperada": 64% → −2.2 pts.
  - Las ráfagas × 0.7 se usan como viento si nflverse todavía no lo tiene.
  - No aplica bajo techo.
- **Ejemplo (12-nov, WAS @ NYG):** con 64% de lluvia, el total del modelo bajó de 46.1 a 44.2 (línea 40.5).
- **Front:**
  - Nota "64% de probabilidad de lluvia (pronóstico ESPN): el total baja ~2.2 pts".
  - En la NFL, las fichas Ganador/Spread/Total dicen "Se inclina por", en gris y con borde punteado: no son recomendaciones.

### Más métricas (`ml/scripts/nfl_features_eval.py`)

Se agregaron a la caché las pérdidas de balón por jugada y una tabla de equipos especiales por partido: EPA neto de patadas, despejes, goles de campo y puntos extra (`load_special_teams`). Se probaron estas métricas, cada una ajustada por rival y con descuento:
- **epant:** EPA sin pérdidas de balón.
- **succ:** tasa de éxito.
- **st:** equipos especiales.
- **QB en el total.**

| Modelo | Error 2015–20 | Error 2021–26 | k contra la línea (15–20 / 21–26) |
|---|---|---|---|
| Margen actual | 9.975 | 10.082 | 0.30 / 0.00 |
| + tasa de éxito | 9.977 | 10.072 | 0.29 / 0.00 |
| + equipos especiales | 9.982 | 10.091 | 0.31 / 0.00 |
| + las tres | 9.992 | 10.080 | 0.30 / 0.01 |
| Total actual | 10.831 | 10.565 | 0.00 / 0.01 |
| **+ QB titulares** | 10.781 | 10.462 | 0.00 / 0.11 |
| **+ QB + EPA sin pérdidas (elegido)** | **10.776** | **10.430** | 0.00 / 0.18 |
| Mercado | — | margen 9.738 · total 10.338 | — |

**Qué dice:**
- El margen no mejora con nada nuevo.
- En el **total**, meter la calidad de los dos QB titulares y su cambio contra el habitual (`qb_sum`, `qb_adj_sum`), más el EPA sin pérdidas (`epant_total`), baja el error de 10.57 a **10.43**. Esto corrige justo los partidos con QB suplente: Rams–Cardinals 48.0 → 46.5 (línea 39.5) y Dolphins–Colts 45.0 → 43.1 (38.5).
- En la mezcla de QB en duda, ahora también se promedian las variables del total.

**Calibración:** antes usaba solo las últimas 4 temporadas, un corte duro que daba k = 0 y ponía todo en 50%. Ahora usa todas las temporadas fuera de muestra, con vida media de 3 temporadas (`CALIB_HALF_LIFE`):
- k spread ≈ 0.05, k total ≈ 0.02, k ganador ≈ 1.09.
- Las probabilidades contra la línea quedan en 49–51%. Es lo que dicen los datos: en 2015–2020 la diferencia con la línea sí predecía algo (k 0.30), y en 2021–2026 ya casi no.

**El lado lo deciden las métricas:** con una probabilidad de exactamente 50%, `market_picks` elegía local u over por defecto. Ahora `nfl_picks` desempata con la diferencia del modelo contra la línea y devuelve `diff_pts`: cuántos puntos ve el modelo a favor de ese lado. En la app se ve como "El modelo ve 2.5 pts a favor de este lado respecto a la línea; en el backtest reciente esa diferencia casi no predijo el resultado".

### Rendimiento en vivo en la app

- `GET /api/predictions/nfl/performance`: mismo formato que el de la NBA (`performance(db, days, sport="nfl")`):
  - récord por mercado (todos y solo recomendados), unidades, error del margen y del total;
  - comparación contra la línea, los últimos partidos con lo que prefería el modelo, y el backtest del entrenamiento.
- El historial de los picks de la NFL usa sus probabilidades calibradas (`log_picks` → `nfl_picks`).
- **Front:**
  - En Bankroll, el historial tiene pestañas NBA y NFL.
  - En Resumen, el panel "Modelo … en vivo" también.

**Pendiente:**
- Guardar las líneas de apertura para medir si hay ventaja antes de que la línea se mueva.
- Bajas por grupo de posición (además del QB).
- Semanas 17–18: equipos ya clasificados o eliminados.
