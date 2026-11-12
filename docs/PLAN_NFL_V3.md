# OmniPicks: plan del modelo NFL v3

**Objetivo:** el pronóstico más acertado posible, con los mejores datos disponibles. No se trata de predecir "por cuánto se define", sino de dar la mejor probabilidad de ganar, el mejor margen y el mejor total.

**Cómo se mide:** la v3 reemplaza a la v2 solo si, en la prueba limpia 2021–2026, mejora:
- **ganador:** log loss y Brier;
- **margen:** error medio (hoy 10.08; mercado 9.74);
- **total:** error medio (hoy 10.43; mercado 10.34).

Cada pieza se valida con backtest antes de entrar. Lo que no mejora, no entra.

**Regla:** el mercado no entra al modelo. Solo se usa para compararse y para medir valor.

## Idea central

La v2 trabaja por **equipo**: EPA y marcador ajustados por rival, más el QB titular. La v3 trabaja por **jugador**, como la NBA v2:

    fuerza del equipo hoy = Σ (valor de cada jugador disponible × snaps esperados)

Así, cada baja (tackle izquierdo, receptor estrella, cornerback, pass rusher) se mide en puntos, y no solo la del QB.

## Datos

| Bloque | Fuente | Qué aporta |
|---|---|---|
| A. Jugada por jugada | nflverse `pbp` (1999–) | EPA, éxito, WP, CPOE, yardas en el aire / después de la recepción (y esperadas), situaciones (tercera oportunidad, zona roja, neutral), explosivas, presión, castigos, series, posición de campo, cuarta oportunidad, equipos especiales |
| B. Participación | `participation` | Los 22 jugadores de cada jugada, personal, defensores en la caja, cobertura, presión → **RAPM por jugador** |
| C. Charting | `ftn_charting` | Play action, RPO, blitz, motion, drops, pases atrapables |
| D. Next Gen Stats | `nextgen_stats` | Tiempo para lanzar, separación, yardas por tierra sobre lo esperado, agresividad |
| E. PFR avanzadas | `pfr_advstats` | Presiones generadas y permitidas, tackles fallados, cobertura por defensor, malos pases |
| F. Snaps | `snap_counts` | Cuánto juega cada jugador (base del impacto de las bajas) |
| G. Plantillas | `rosters_weekly`, `depth_charts` | Titulares y suplentes por posición |
| H. Lesiones | `injuries` + ESPN | Estado y práctica de todos los jugadores |
| I. Novatos y llegadas | `draft_picks`, `combine`, `contracts` | Valor inicial de jugadores sin historial |
| J. Contexto | `schedules` + ESPN | Descanso, viaje, horario estelar, división, altura, superficie, techo, clima pronosticado, coach y coordinadores, clasificados o eliminados |
| K. Árbitros | `officials` | Tasa de castigos por cuerpo arbitral |
| L. Mercado | nflverse + ESPN | Apertura y cierre; **solo para comparar** |

## Fases

| Fase | Qué | Validación |
|---|---|---|
| 0 | Inventario: bajar cada fuente y verificar temporadas, columnas y si trae la temporada actual | Tabla de cobertura |
| 1 | Valor de cada jugador por posición: RAPM con participación + estadísticas por posición | Predice el siguiente partido mejor que el rating de equipo |
| 2 | Fuerza del equipo hoy con los jugadores disponibles y sus snaps esperados | Error del margen, sobre todo con bajas importantes |
| 3 | Disponibilidad de todos los jugadores (lesión + práctica), aprendida del histórico | Calibración de "¿jugó?" |
| 4 | Simulación por series (series por partido × puntos por serie): marcador, total y números clave | Error del total y log loss del margen exacto |
| 5 | Contexto completo y comparación ridge contra LightGBM | Prueba limpia 2021–2026 |
| 6 | Backtest de apuestas (cierre, y apertura con lo que se vaya guardando) | Unidades y ROI con momios reales |
| 7 | Producción y app: bajas por jugador con su impacto, desglose de "cómo se arma el margen" | Revisión en el navegador |

## Expectativa honesta

- En la NFL el margen real se aleja ~13 puntos del pronóstico, y la línea de cierre falla 9.7 en promedio. Ningún modelo elimina esa incertidumbre.
- Meta realista: margen ~9.9 y total ~10.35, con mejores probabilidades de ganar.
- La ganancia más segura está en entender cada partido: el impacto de cada baja, por qué el modelo ve lo que ve.

## Fase 0: inventario (12-nov-2026, `ml/scripts/nfl_data_inventory.py`)

| Fuente | Temporadas | ¿Trae 2026? | Para qué |
|---|---|---|---|
| Jugadas (`pbp`) | 1999–2026 | ✅ hasta la semana 9 | EPA, situaciones, series, especiales (ya en caché) |
| Participación | **2016–2025** | ❌ (nflverse aún no la publica) | RAPM por jugador (quién estuvo en el campo) |
| Snaps | 2012–2026 | ✅ semana 9 | Cuánto juega cada jugador; impacto de las bajas |
| FTN charting | 2022–2026 | ✅ semana 9 | Play action, blitz, motion, drops |
| Next Gen Stats | 2016–2026 | ✅ semana 9 | Tiempo para lanzar, separación, carrera sobre lo esperado |
| PFR avanzadas (pase, carrera, recepción, defensa) | 2018–2026 | ✅ | Presiones, tackles fallados, cobertura |
| Lesiones y práctica | 2009–2026 | ✅ semana 10 | Disponibilidad de todos los jugadores |
| Plantillas semanales | 2002–2026 | ✅ semana 10 | Posición, estado, experiencia |
| Depth charts | 2001–2026 | ✅ (desde 2025 son fotos diarias) | Titulares y suplentes |
| Estadísticas por jugador | 1999–2026 | ✅ semana 9 | Producción por partido |
| Árbitros | 2015–2026 | ✅ | Cuerpo arbitral |
| Draft, combine, contratos | 1980–2026 | ✅ | Valor inicial de novatos y llegadas |

**Consecuencia para el diseño:** como la participación no tiene 2026, el valor de cada jugador no puede salir solo del RAPM. Se hace como en la NBA v2:
- **Base histórica:** RAPM con la participación de 2016–2025.
- **Ajuste con la temporada en curso:** snaps, estadísticas por jugador, PFR, NGS y FTN, que sí están al día.

Cuando nflverse publique la participación de 2026, el RAPM se actualiza solo.

## Fase 1: valor de cada jugador (`app/services/nfl_players.py`, `ml/scripts/nfl_players_eval.py`)

**RAPM jugada por jugada:** 252,133 jugadas de 2016–2025, con los 22 jugadores en el campo (sin garbage time):

    EPA de la jugada = Σ ataque[jugadores de ataque] − Σ defensa[jugadores de defensa]

- Ridge con encogimiento por grupo: QB 50 jugadas virtuales, resto 800.
- Usa las 3 temporadas anteriores, con vida media de 2 temporadas. Cada temporada se predice **solo con las anteriores**.
- Ranking 2022–2024 coherente:
  - **Ataque:** Tua, Allen, Purdy, Lamar, Hurts, Carr, Mahomes, Daniels.
  - **Defensa:** D. Williams, Sweat, Bosa, Cashman, Rousseau, Madubuike.

**Fuerza del equipo** = Σ valor × fracción de jugadas del jugador:
- **esperada:** su uso con descuento en los partidos previos de la temporada;
- **real:** quién jugó de verdad. Es el techo, como si se supieran todas las bajas.

**Resultado** (2,090 partidos, semana 2 en adelante, 2018–2025; la regresión se ajusta solo con temporadas previas desde 2018):

| Modelo | 2019–2020 | 2021–2025 (prueba limpia) | Semanas 2–6 (2021–25) |
|---|---|---|---|
| v2 (EPA, marcador, QB) | 10.162 | 10.164 | 10.260 |
| + jugadores (uso esperado) | 10.115 | 10.158 | 10.236 |
| + jugadores (quién jugó: techo) | 10.126 | 10.138 | 10.235 |
| **+ esperado + efecto de las bajas** | 10.142 | **10.135** | 10.237 |
| Solo jugadores | 10.418 | 10.761 | 10.549 |
| Mercado | 9.980 | 9.776 | 9.826 |

**Qué dice:**
- El valor por jugador **mejora poco el margen**: −0.03 puntos en la prueba limpia, y solo si se sabe bien quién juega. El rating de equipo y el QB ya capturan casi todo.
- Las bajas sí mueven partidos concretos. El efecto "quién jugó − esperado" tiene una desviación de 3.9 EPA por partido, y en el 5% de los partidos pasa de 7.8. En promedio pesan poco, pero cuando falta un grupo importante se nota.
- Para aprovecharlo en producción hace falta la **fase 3**: predecir quién juega con los reportes de lesiones de todos los jugadores. Ahí está el −0.03; sin eso, solo −0.006.
- Esto coincide con lo que se sabe de la NFL: fuera del QB, el valor de un jugador individual es chico y ruidoso.

**Pendiente de la fase 1:**
- Afinar el RAPM con estadísticas por posición: presiones (PFR), separación (NGS), snaps. Sirve sobre todo para 2026, que no tiene participación.
- Mapear los snaps de 2026 (ids de PFR) a los ids de nflverse para el uso esperado en producción.

## Fase 3: quién juega, todas las posiciones (`nfl_players.py`, `ml/scripts/nfl_availability_eval.py`)

**P(juega | estado, práctica)**, aprendida con 2016–2020: reportes finales de lesiones de nflverse cruzados con la participación. Solo cuentan jugadores con uso esperado ≥ 25%; "jugó" = estuvo en al menos un tercio de su uso esperado.

| Estado | Casos | Juega | Entrenó completo | Limitado | No entrenó |
|---|---|---|---|---|---|
| Questionable | 4,952 | 67% | 86% | 69% | 40% |
| Doubtful | 531 | 1% | — | — | — |
| Out | 2,687 | 0% | — | — | — |
| Sin estado (solo práctica) | 9,772 | 92% | 93% | 93% | 78% |

**Bajas previstas:** para cada jugador del reporte,

    pérdida = (1 − P(juega)) × su uso esperado × (su valor RAPM − el de un suplente de su posición)

El suplente típico es la media del RAPM de los jugadores de su grupo con pocas jugadas.

**Resultado** (2021–2025 es la prueba limpia):

| Modelo | 2019–2020 | 2021–2025 | 2021–25, partidos con bajas grandes |
|---|---|---|---|
| v2 (EPA, marcador, QB) | 10.162 | 10.164 | 10.084 |
| + jugadores (uso esperado, sin lesiones) | 10.115 | 10.158 | 10.110 |
| **+ jugadores con bajas previstas** | 10.116 | **10.152** | **10.010** |
| + jugadores (quién jugó: techo) | 10.142 | 10.135 | 10.028 |
| Mercado | 9.980 | 9.776 | 9.584 |

"Bajas grandes" = el 20% de partidos con más diferencia de bajas previstas entre los equipos (≥ 1.9 EPA por partido).

**Qué dice:**
- En general la mejora es chica: −0.012 puntos.
- En los partidos con bajas importantes, **el error baja 0.07 puntos**, e incluso queda mejor que el techo. Es justo donde la v2 no sabía nada fuera del QB.
- El mercado sigue adelante (9.58 en esos partidos): sabe de las bajas y además de su contexto (quién entra, esquemas).

**Para producción hace falta:**
- **Uso esperado en 2026 a partir de los snaps**, porque no hay participación de 2026. Los snaps traen el id de PFR; se pasa al de nflverse con la tabla de jugadores.
- **Estado y práctica de la semana:** de nflverse (todas las posiciones) y de ESPN.
- **Valores RAPM de 2023–2025.** Los novatos y los jugadores sin historial toman el valor de un suplente de su posición.

## Bajas por jugador en producción (12-nov-2026)

**Comparación justa antes de pasarlo.** Contra la v2 de producción, que se entrena desde 2011 y no desde 2018 como en el backtest de la fase 3, la v3 completa **no ganaba en general**: 10.160 contra 10.154. Solo ganaba en los partidos con bajas grandes: 10.058 contra 10.120. Por eso entra como **híbrido**:

    margen = v2                                                si la diferencia de bajas es chica
    margen = a + b·v2 + c·fuerza por jugadores + d·bajas         si |bajas local − visitante| ≥ p70

El umbral p70 (~1.2–1.3 EPA por partido) se elige con las temporadas de entrenamiento. El ajuste se aprende **encima de la v2**, con la v2 fuera de muestra.

| Versión | 2019–2020 | 2021–2025 (limpia) |
|---|---|---|
| v2 de producción | 10.099 | 10.154 |
| Híbrido, sin el QB en las bajas | 10.124 | 10.142 |
| **Híbrido, con el QB en las bajas (elegido)** | **10.088** | **10.144** |

Sacar al QB de las bajas parecía evitar contarlo dos veces, pero empeoraba 2019–2020. Como el ajuste se aprende encima de la v2, la regresión solo le da al QB el peso que agrega.

**P(juega)** se normaliza contra la de un jugador sin estado que entrenó completo (~93%). Así, un jugador sano no resta nada solo por estar en el reporte.

**En producción** (`nfl_v2.PlayersNow`):
- **Uso esperado:** de los **snaps de 2026** (`snaps_usage`; ids de PFR → nflverse), con la misma media con descuento.
- **Reporte:** el de la semana de nflverse (estado oficial y práctica), completado con el estado del día de ESPN.
- **Valores RAPM:** con 2023–2025 (`fit_rapm` al entrenar, guardados en `nfl_v2.json`), y suplente típico por posición.

`simulation_details.player_injuries` trae:
- por equipo: las bajas (nombre, grupo, estado, práctica, P(juega), uso, puntos que cuesta) y el total;
- si el ajuste se aplicó, y el margen v2 contra el margen con bajas.

**Front:** el bloque "Bajas" de la NFL muestra cada baja con su costo en puntos y dice si movió el pronóstico.
