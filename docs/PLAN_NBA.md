# OmniPicks NBA v2: plan del modelo

**Fecha:** 10-nov-2026 · **Estado:** N0–N6 implementados (ver la sección 12 y [`NBA_BACKTEST.md`](NBA_BACKTEST.md)).

## Principios (acordados)
1. **El modelo no usa el mercado de apuestas.** Las líneas (ESPN/DraftKings, The Odds API) se guardan **solo para comparar**: modelo contra mercado, medición de qué tan lejos o cerca estamos, y detección de diferencias.
2. **Se toma en cuenta todo**: jugadores, bajas, cómo rinde el equipo **con y sin** cada jugador, rotaciones, descanso, viaje, altitud, calendario y momento de la temporada.
3. **Se mide antes de creer**: cada pieza entra solo si mejora el backtest.

## Datos disponibles gratis (verificados hoy)
| Dato | Fuente | Estado |
|---|---|---|
| Calendario, marcadores, récords | ESPN scoreboard | ✅ ya integrado |
| **Lesiones**: estado (Out, Doubtful, Questionable, Probable, Day-to-Day), comentario y fecha | ESPN `/nba/injuries` | ✅ probado: 93 lesionados en 29 equipos |
| Box score por jugador y por partido | `nba_api` PlayerGameLogs, BoxScoreTraditional/Advanced V3 | ✅ probado |
| **Rotaciones**: minuto exacto de entrada y salida de cada jugador | `nba_api` GameRotation | ✅ probado (77 filas por partido) |
| Play-by-play (posesiones, puntos por segmento) | `nba_api` PlayByPlayV3 | ✅ probado (~500 eventos por partido) |
| **On/Off**: rating neto del equipo con y sin cada jugador | `nba_api` TeamPlayerOnOffSummary | ✅ probado |
| Quintetos (5 jugadores) con ratings | `nba_api` LeagueDashLineups | ✅ probado (2,000 quintetos) |
| Stats avanzadas por jugador (USG%, TS%, PIE, rating neto) | `nba_api` LeagueDashPlayerStats | ✅ probado |
| Histórico de temporadas pasadas | `nba_api` LeagueGameLog | ✅ probado (2021-22: 2,460 registros) |
| Reporte oficial de lesiones de la NBA (PDF varias veces al día) | nba.com | 🔎 por validar |
| Líneas del mercado (**solo para comparar**) | ESPN (DraftKings), The Odds API | ✅ ya integrado |

---

## 1. Capa de datos (histórico propio)

Nuevas tablas en MySQL:

| Tabla | Contenido | Uso |
|---|---|---|
| `nba_player_games` | Box score de cada jugador en cada partido (min, pts, reb, ast, tov, tiros, +/-) | Ratings, minutos, uso |
| `nba_stints` | Segmentos donde los mismos 10 jugadores están en cancha, con posesiones y puntos de cada lado (de GameRotation + play-by-play) | RAPM (impacto real de cada jugador) |
| `nba_availability` | Por partido y jugador: jugó / no jugó / motivo. Hacia adelante, también cada reporte de lesiones con su hora | Bajas históricas y probabilidad de jugar |
| `nba_player_ratings` | Rating ofensivo y defensivo de cada jugador **por fecha**, con su incertidumbre | Point-in-time (backtest sin fuga de datos) |
| `nba_market_lines` | Spread, total y moneyline con hora de captura | **Solo para comparación** |

- **Backfill**: temporadas 2022-23 a 2026-27, unos 6,000 partidos con 3 llamadas cada uno y pausas para no saturar stats.nba.com. Es una corrida **única de varias horas**; después se actualiza solo lo nuevo de cada día.
- **Bajas en el histórico**: ESPN no da reportes de lesiones pasados, así que en temporadas viejas "baja" = jugador del roster con 0 minutos. A partir de ahora se guarda cada reporte, para calibrar qué tan seguido juega un "Questionable".

---

## 2. Rating de cada jugador

Es la base de todo. Si sabemos cuánto vale cada jugador, sabemos cuánto pierde un equipo sin él.

**2a. Prior por box score (tipo BPM).** Regresión que convierte las estadísticas por 100 posesiones (puntos, eficiencia, asistencias, rebotes, robos, tapones, pérdidas, uso) en impacto. Se entrena con nuestros propios datos contra el RAPM. Es estable aunque haya pocos partidos.

**2b. RAPM con prior (plus-minus ajustado y regularizado).** Regresión ridge sobre los stints: cada segmento dice "con estos 10 en cancha, el equipo A hizo X puntos en N posesiones". Separa el impacto de cada jugador del de sus compañeros y rivales. El prior de 2a evita que jugadores con pocos minutos tengan valores extremos. Se calcula **por separado en ataque y defensa**, con decaimiento temporal (lo reciente pesa más).

**2c. Actualización diaria tipo filtro de Kalman** (la idea detrás de DARKO):
- Estado = el "nivel verdadero" del jugador, con su incertidumbre.
- Cada partido es una observación ruidosa: un partido de 8 minutos mueve poco y uno de 38 mueve más.
- Ruido de proceso: el nivel cambia con el tiempo (edad, rol, regreso de lesión).
- Curva de edad: jóvenes que mejoran y veteranos que bajan, como prior entre temporadas.

**Salida:** por jugador y por fecha, `O-rating`, `D-rating` (puntos por 100 posesiones sobre el promedio) y `±` de incertidumbre.

---

## 3. Bajas y cómo rinde el equipo con y sin cada jugador ⭐

Es la pieza que más diferencia hace.

**3a. Probabilidad de jugar.**
| Estado ESPN | Prob. inicial | Se calibra con… |
|---|---|---|
| Out | 0% | — |
| Doubtful | ~15% | Historial que vamos guardando |
| Questionable | ~50% | ídem |
| Probable / Day-to-Day | ~85–90% | ídem |
| Sin reporte | ~98% | ídem |

Extras:
- **Descanso de estrellas**: probabilidad de que una estrella no juegue el segundo día de un back-to-back, aprendida de su historial.
- **Regreso de lesión**: los primeros 2–3 partidos con restricción de minutos (~70% de lo normal) y un pequeño descuento de rendimiento.

**3b. Redistribución de minutos.** Cuando falta un jugador, ¿quién juega sus minutos?
- **Aprendido del historial del propio equipo**: en los partidos en que faltó X, cuántos minutos extra recibió cada compañero.
- Respaldo cuando no hay historial: misma posición y lugar en la rotación (de GameRotation), con el tope de minutos de cada jugador.
- Minutos base de cada jugador: 60% del promedio reciente y 40% de su rol en la rotación (el método de FiveThirtyEight).

**3c. Redistribución de uso y eficiencia.** Cuando falta el que más tira, los demás tiran más, y su eficiencia baja un poco (relación uso–eficiencia). Se ajusta el rating ofensivo de cada compañero según cuánto uso extra absorbe.

**3d. Con y sin observado (validación empírica).**
- On/Off (TeamPlayerOnOffSummary) y rendimiento real del equipo en los partidos sin el jugador.
- Esas muestras son chicas y ruidosas, así que se combinan con el modelo de forma bayesiana. Con pocos partidos sin el jugador, manda el modelo (2 + 3b + 3c). Con muchos, el dato observado pesa más.
- Esto captura cosas que la suma de ratings no ve: un equipo que juega mejor sin su estrella, o un sistema que depende de un base específico.

**3e. Química de quintetos** (efecto menor). Los quintetos con muchos minutos juntos (LeagueDashLineups) aportan un ajuste pequeño y muy suavizado. Ojo: en NBA estas muestras son ruidosas y suelen sobreajustarse.

**Salida para el front:** "Sin Brunson: Knicks −3.1 pts de margen · sus minutos van a McBride (+14) y Hart (+6)".

---

## 4. Contexto del partido

| Factor | Cómo se modela |
|---|---|
| Ventaja de local | Base de liga (~2–2.5 pts), con ajuste propio por equipo **muy suavizado** |
| Descanso | 0 (back-to-back), 1, 2 o 3+ días; también 3 partidos en 4 noches y 5 en 7 |
| Viaje | Distancia entre arenas y cambio de zona horaria (coordenadas de las 30 arenas) |
| Altitud | Denver y Utah, mayor efecto en visitantes cansados |
| Momento de temporada | Inicio (más incertidumbre), final (equipos que pierden a propósito o que ya aseguraron lugar y descansan titulares), NBA Cup, playoffs (más minutos para titulares, ritmo más bajo) |
| Ritmo | Ritmo de cada equipo **según los jugadores disponibles** |

Ningún factor se fija a mano: todos se estiman con el histórico, y si alguno no mejora el backtest, se quita.

---

## 5. Simulación del partido

Monte Carlo con unas 10,000 corridas. En cada una:
1. **Se sortea quién juega**, según la probabilidad de jugar de cada uno (3a). Así, un "Questionable" juega en la mitad de las simulaciones y no en la otra mitad.
2. Se reparten minutos (3b) y uso (3c), y se calcula el rating del equipo para ese escenario: Σ minutos × rating.
3. **Posesiones** según el ritmo de los jugadores en cancha, con su variación natural.
4. **Puntos por posesión** de cada lado: ataque propio contra defensa rival, más contexto (sección 4) y ruido realista.
5. Si hay empate, tiempo extra.

**Salida:** probabilidad de victoria, distribución completa del margen y del total, marcador más probable, rango del 80%, puntos por cuarto (opcional) y, a futuro, proyección por jugador (puntos, rebotes, asistencias).

**Corrección con ML** (opcional, solo si mejora el backtest): LightGBM entrenado sobre los **errores** del simulador, con variables de contexto. **Sin variables del mercado.**

**Calibración**: regresión isotónica para que "70%" signifique de verdad 70%.

---

## 6. El mercado: solo para comparar
- Se guarda cada línea (spread, total, moneyline) con su hora.
- En el front aparece **"Modelo vs Mercado"**: margen, total, probabilidad y la diferencia.
- Se mide:
  - qué tan seguido nuestro modelo está más cerca del resultado real que el mercado;
  - si las diferencias grandes aciertan;
  - si el mercado se mueve hacia nuestro número antes del partido.
- Los picks que se muestran son **diferencias modelo–mercado, etiquetadas como tales**. No entran al modelo.

---

## 7. Evaluación (backtest walk-forward)

- **Temporadas de prueba**: 2023-24, 2024-25 y 2025-26. Cada día se predice usando solo información disponible hasta ese día.
- **Comparación contra referencias**:
  1. modelo actual `nba_possessions_v1`;
  2. Elo simple;
  3. el mercado, solo como vara de medir.
- **Métricas**: log loss, Brier, acierto de ganador, MAE del margen, MAE del total y curva de calibración.
- **Prueba de que las bajas suman**: se mide aparte en los partidos donde faltó un jugador importante. Ahí debería verse la mayor mejora.
- **Prueba por pieza** (ablación): con y sin cada módulo (2b, 2c, 3b, 3c, 3d, 3e, 4, ML), para quedarnos solo con lo que aporta.
- **Referencias aproximadas** (no promesas): los buenos modelos públicos de NBA aciertan ~66–69% de los ganadores y tienen un MAE de margen de ~10.5–11.5 pts. El mercado de cierre ronda lo mismo o un poco mejor.

---

## 8. Operación diaria
| Momento | Qué corre | Costo |
|---|---|---|
| Madrugada | Actualizar box scores, stints y ratings con los partidos de ayer | Gratis (`nba_api`) |
| Mañana | Predicción completa del día | Gratis |
| Cada 30–60 min hasta el partido | Volver a leer lesiones (ESPN y reporte oficial) y re-simular **solo** si cambió algo | Gratis |
| ~30 min antes del partido | Última predicción "congelada" para la evaluación | Gratis |

La NBA queda **100% gratis**: no gasta API-Sports y The Odds API es opcional, solo para comparar.

---

## 9. Lo que verá el usuario (front)
- Marcador proyectado, probabilidad y rango del 80%.
- **Alineación y minutos proyectados** de cada equipo.
- **Bajas con su impacto en puntos** y a quién van sus minutos.
- Factores del partido: descanso, viaje, altitud.
- Bloque aparte **"Comparación con el mercado"**.
- Historial de rendimiento del modelo (acierto y error real).

---

## 10. Fases (NBA)
| Fase | Entregable | Criterio para avanzar |
|---|---|---|
| **N0. Datos** | Tablas de la sección 1 y backfill 2022-27, con actualización diaria | Partidos y stints completos; minutos por equipo cuadran (~240 por partido) |
| **N1. Ratings de jugador** | 2a + 2b + 2c, guardados por fecha | Los ratings predicen el margen del siguiente partido mejor que los ratings de equipo |
| **N2. Bajas y "con y sin"** | 3a–3d (3e al final, si aporta) | Mejora clara en los partidos con bajas importantes |
| **N3. Simulador** | Sección 5 con sorteo de disponibilidad | Distribución del margen bien calibrada |
| **N4. Contexto + ML + calibración** | Secciones 4 y 5 (ML) | Cada factor sobrevive a la ablación |
| **N5. Backtest y reporte** | Reporte con todas las métricas contra referencias | Supera a `nba_possessions_v1` en log loss y MAE |
| **N6. Producción** | Integración al pipeline, refresco de lesiones, front | Corre solo y sin errores una semana |
| **N7. Extras** | Proyecciones por jugador (puntos, rebotes, asistencias) | Opcional |

## 11. Decisiones (10-nov-2026)
1. **Front**: solo el **impacto de las bajas** (sin alineaciones ni minutos proyectados).
2. **Refresco de lesiones**: **cada 30 minutos** el día del partido.
3. **Backfill**: **2022-23 a 2026-27**, es decir, 4 temporadas completas más la actual. Razones:
   - el RAPM necesita varias temporadas para separar bien a los jugadores;
   - el backtest usa 3 temporadas de prueba (2023-24 a 2025-26), y la primera necesita una temporada previa para calentar los ratings;
   - la redistribución de minutos aprende de muchos partidos con bajas.

   Es una corrida única de unas 3–5 horas, que se puede pausar y reanudar.
4. **Dependencias**: aprobado instalar `lightgbm`, `scikit-learn` y las que hagan falta.

## 12. Estado de la implementación (10-nov-2026)

| Fase | Estado | Dónde | Notas |
|---|---|---|---|
| N0. Datos | ✅ | `app/services/nba_history.py`, `nba_injuries.py`, `nba_scheduler.py`, `ml/scripts/nba_backfill.py` | Los puntos de los stints se cuentan por evento (la primera versión los duplicaba en ~15% de los partidos). Lesiones cada 30 min; histórico y ratings a las 04:30 ET. |
| N1. Ratings de jugador | ✅ | `app/services/nba_ratings.py` | 2a, 2b y 2c en **una sola regresión** sobre los stints: el perfil de box score (filtro bayesiano con descuento, más minutos y edad) entra como variable y el RAPM es el residuo propio de cada jugador, con ridge y decaimiento. Se recalcula cada día (~20 s). |
| N2. Bajas | ✅ | `app/services/nba_lineups.py`, `ml/scripts/nba_minutes_model.py` | 3a (probabilidad por estado); 3b con **LightGBM de minutos** que usa quién reemplaza a quién (rotaciones) y el parecido por posición y estatura: el error de minutos baja de 5.75 a 4.78. 3c/D (uso absorbido) en la corrección. Impacto de cada baja con rango, y con/sin él observado. |
| N3. Simulador | ✅ | `app/services/nba_game_model.py` | Escenarios de quién juega: con hasta 8 jugadores en duda se enumeran todos con su probabilidad exacta; con más, se sortean. De la mezcla salen 10,000 marcadores. |
| N4. Contexto + ML + calibración | ✅ | `app/services/nba_game_model.py` | Descanso, back-to-back, 3 en 4 noches, viaje, zona horaria, altitud, inicio y final de temporada, playoffs. Corrección ridge sobre los errores del estructural. |
| N5. Backtest | ✅ | `ml/scripts/nba_backtest.py` → [`NBA_BACKTEST.md`](NBA_BACKTEST.md) | Walk-forward 2023-24 a la fecha contra `nba_possessions_v1` y Elo, con ablación. |
| N6. Producción | ✅ | `app/services/nba_v2.py`, `prediction_engine.py`, front `game-card` | `nba_players_v2` reemplaza a v1 (v1 queda de respaldo si v2 falla, o con `NBA_MODEL=v1`). Se re-simula cuando cambia el reporte de lesiones. El front muestra el bloque **Bajas** con el impacto de cada una. |
| N7. Extras | ❌ | — | Proyecciones por jugador. |

**Medido y descartado** (no mejoró el backtest; el código queda, apagado):
- **3d, rendimiento observado sin el jugador:** corregir con lo que el equipo hizo sin él empeora el log loss (0.595 → 0.599–0.616 según el peso). Las muestras sin cada jugador son demasiado chicas y ruidosas.
- **3b aprendido** (a quién se fueron los minutos en los partidos que faltó): error de minutos 5.7–5.8 contra 5.5 del reparto proporcional.
- **LightGBM** para el contexto: peor que la ridge (hay pocas variables y la relación es casi lineal). En cambio, **para los minutos sí gana** y está en producción.
- **Reparto solo por reemplazos** (sin ML): un poco peor que el proporcional; sirve como variable del LightGBM.
- **Ratings con descuento por partidos jugados** (tipo Kalman, para que un lesionado no baje a nivel de suplente): sin cambio en el backtest.
- **Ajuste por racha del equipo:** no aporta.

**Decididos por el backtest** (entran solo si bajan el log loss; el script elige solo): la corrección de contexto, el ajuste por uso de los ausentes (3c) y la calibración isotónica. La elección vigente está en [`NBA_BACKTEST.md`](NBA_BACKTEST.md).

**Hecho el 11-nov (tarde):**
- Medición en vivo contra resultado y mercado (`nba_prediction_log` y `/api/predictions/nba/performance`).
- Probabilidad de jugar aprendida por estado (necesita semanas de reportes para pesar).
- Descanso en back-to-back por jugador (aplicado a la mitad; se valida en vivo).
- Ritmo según quién juega (error del total 14.80 → 14.78).
- Reporte oficial de lesiones de la NBA (PDF) como fuente principal.

**Pendiente:**
- Química de quintetos (3e), solo si aporta.
- Proyecciones por jugador (N7).
