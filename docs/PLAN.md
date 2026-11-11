# OmniPicks: plan de corrección y fuentes de datos

**Fecha:** 10-nov-2026
**Alcance:** backend (`omnipicks_backend`) y frontend (`Angular_proyects/omnipicks`)
**Regla:** solo fuentes **gratis**. Se puede combinar más de una por tipo de dato.

---

## 0. Estado actual de un vistazo

| Deporte | ¿Se ven partidos? | ¿El modelo usa datos reales? | Veredicto |
|---|---|---|---|
| **MLB** | Sí (temporada terminada) | Sí: ERA, bullpen, carreras, H2H | ✅ Funciona: acertó al ganador en 58.6% de 1,657 predicciones |
| **NBA** | ✅ Sí, desde hoy | ✅ Sí, desde hoy | Arreglado en la Fase 1 (ver abajo) |
| **NFL** | Sí, pero con fechas equivocadas | ❌ **No**: la predicción es casi constante | 41% de acierto (11 de 27); hay que rehacer el modelo |
| **Fútbol** | Sí (hoy no hay partidos por fecha FIFA) | A medias: muestras chicas y fórmula sin topes | ❌ Marcadores irreales (hasta 15.4 goles) |

---

## 1. ✅ Hecho: Fase 1, NBA

### Lo que estaba roto
1. **No entraba ningún partido.** API-Sports *sí* devuelve NBA; la prueba trajo 5 partidos con `errors: []`. Pero la liga viene como `"standard"` y el filtro solo aceptaba `"NBA"`, así que se descartaban todos. Además usa `teams.visitors` en vez de `teams.away`, los puntos vienen en `scores.*.points` y el estado es un número (`3`), no `"FT"`.
2. **No había algoritmo real.** El simulador usaba `adjusted_home_elo = random.uniform(1450, 1600)`, un Elo aleatorio. Los back-to-backs también eran aleatorios. Los ratings reales se calculaban, pero el simulador nunca los leía.
3. Las estadísticas pedían la temporada `'2025-26'`, que ya terminó, y hacían 4 llamadas por partido.
4. `value_calculator` calculaba over/under con Poisson para todos los deportes. Con totales de ~220 puntos lanzaba `OverflowError`.

### Lo que se cambió
| Archivo | Cambio |
|---|---|
| `app/services/espn_nba_client.py` (nuevo) | Calendario diario desde ESPN: gratis, sin cuota, con fechas en hora ET. Incluye récords y la línea de DraftKings como referencia. |
| `app/services/data_ingestion.py` | NBA se ingiere desde ESPN (`_ingest_nba_espn`) y se quitó el filtro roto. **Ahorra 2 requests de API-Sports por corrida.** |
| `app/services/nba_advanced_metrics.py` | Reescrito. `nba_api` con la temporada calculada a partir de la fecha. Trae ratings OFF/DEF, ritmo, 4 factores, últimos 10 partidos y la temporada anterior como referencia, todo **en caché**: 4 llamadas por corrida en vez de 4 por partido. Los back-to-backs salen del scoreboard de ESPN de ayer. Si falla, marca `api_error` y el front lo muestra. |
| `app/services/feature_engineering.py` | Se eliminaron todos los valores aleatorios de NBA. |
| `app/services/simulation_engine.py` | Modelo nuevo `nba_possessions_v1`. Calcula posesiones esperadas × eficiencia cruzada (mi ataque contra tu defensa) y suma 2.5 puntos de ventaja local y un ajuste por back-to-back. Suaviza al inicio de temporada: mezcla con la temporada pasada (regresada 1/3 a la media), con un peso de 10 partidos. Simula 5,000 veces el margen (σ 12.5) y el total (σ 17). |
| `app/services/value_calculator.py` | NBA y NFL usan una distribución Normal para el total. Poisson se calcula en logaritmos, así que ya no se desborda. *Esto también corrige NFL, donde Poisson subestimaba la varianza a la mitad.* |
| `app/services/prediction_engine.py` | Nuevo parámetro opcional `sports=[...]` para correr las predicciones de un solo deporte. |

### Resultado (10–11 nov)
- 16 partidos ingeridos y predichos; cuotas en 9; 15 picks.
- Totales cercanos al mercado: 219.9 contra 222.5, 214.5 contra 217.5, 235.2 contra 237.5.
- Spreads: error promedio contra el mercado de unos 2.4 puntos (antes de usar la temporada pasada era de unos 5). NY −15.7 contra −15.5; SA −16.4 contra −14.5.
- ⚠️ **Pendiente:** todavía hay picks con EV inflado en underdogs (por ejemplo, Utah +319). Se resuelve en la Fase 6, mezclando el modelo con el mercado.

---

## 2. Todos los errores encontrados

Severidad: 🔴 crítico · 🟠 alto (resultados incorrectos) · 🟡 medio · ⚪ bajo / limpieza

### Seguridad
| # | Sev. | Error | Dónde |
|---|---|---|---|
| S1 | 🔴 | Las 3 API keys (Bzzoiro, API-Sports, The Odds API) y la contraseña de MySQL siguen en el historial de git (commit `521fc0f`) | git |
| S2 | 🟡 | El front llama a `http://localhost:8000` con la URL fija en el código; no hay configuración por entorno | `services/api.service.ts:9` |

### Fútbol
| # | Sev. | Error | Dónde |
|---|---|---|---|
| F1 | 🔴 | La fórmula `league_avg × ataque × defensa_rival` no tiene topes. Feyenoord vs Como salió **1.07–15.38**; hay 29 predicciones con más de 4 goles en Champions. | `simulation_engine.py` (`_simulate_soccer`) |
| F2 | 🔴 | ESPN se acepta con `gamesPlayed > 0`: con **1 partido**, 4 goles se vuelven un promedio de 4.0 y `win_pct` llega a 100%. Tu corrección (al menos 5 partidos y que traiga goles) **no está aplicada**. | `soccer_advanced_metrics.py:474` |
| F3 | 🟠 | En Champions y Europa League el récord de ESPN solo cuenta partidos de **ese torneo** (1 a 3 en octubre). | `soccer_advanced_metrics.py:440` |
| F4 | 🟠 | API-Football está limitada a `MAX_SEASON = 2024`: los datos "históricos" son de hace dos temporadas, y los equipos ascendidos no tienen datos. | `soccer_advanced_metrics.py:40` |
| F5 | 🟠 | `game_date_str` llega vacío para fútbol: se lee `game["date"]`, pero API-Sports lo trae en `fixture.date`. La temporada siempre cae en el valor por defecto. | `feature_engineering.py` (`_compute_soccer_features`) |
| F6 | 🟠 | Elo = `1500 + (win% − 50) × 10`: con 1 partido sale 2000 contra 1000. | `feature_engineering.py` |
| F7 | 🟡 | `league_avg = 1.3` es igual para todas las ligas, aunque la Bundesliga promedia más goles que la Serie A. | `simulation_engine.py` |
| F8 | 🟡 | La ventaja de local se **suma** (+0.35 a +0.45 goles) en vez de multiplicarse; es demasiado alta (lo típico es ~+0.25, o ×1.15). | `simulation_engine.py`, `feature_engineering.py` |
| F9 | 🟡 | El piso `max(0.3, …)` convierte datos en cero en 0.3 goles, un valor artificial. | `simulation_engine.py` |
| F10 | 🟡 | Si la liga no está en el mapa, el slug de ESPN cae en `"uefa.champions"`. | `soccer_advanced_metrics.py:440` |
| F11 | ⚪ | La ingesta de fútbol gasta 1 request de API-Sports por día solo para traer el calendario (ESPN lo da gratis). | `data_ingestion.py` |

### NFL
| # | Sev. | Error | Dónde |
|---|---|---|---|
| N1 | 🔴 | **El simulador ignora los datos reales.** Lee `wind_speed` y `qb_diff`, pero nadie los calcula, así que **todos** los partidos salen con local −2.5 y total 45. Por eso hay 41% de acierto y la probabilidad media del local es 60%. | `simulation_engine.py` (`_simulate_nfl`) |
| N2 | 🟠 | Si no hay partidos ese día, se guarda **la semana completa** con la fecha de hoy. Hay 15 partidos del 6 al 10 de noviembre marcados como 11 de noviembre. | `data_ingestion.py:163` y `:200` |
| N3 | 🟡 | El "EPA" no es EPA real (sale ~3.1; el EPA real por jugada está entre −0.3 y +0.3). Es un proxy derivado de puntos y yardas, y Elo = `1500 + epa×500` da valores de ~3000. | `nfl_advanced_metrics.py`, `feature_engineering.py` |
| N4 | 🟡 | No toma en cuenta lesiones (en particular del QB), descansos ni bye weeks. | — |

### MLB
| # | Sev. | Error | Dónde |
|---|---|---|---|
| M1 | 🟡 | `home_elo` y `bullpen_fatigue` son **aleatorios**. El simulador no los usa, pero se muestran como si fueran factores reales. | `feature_engineering.py:47-53` |
| M2 | ⚪ | El factor de estadio solo cubre 4 estadios; los otros 26 quedan en 1.0. | `feature_engineering.py` |
| M3 | ⚪ | `statsapi` es síncrono y bloquea el event loop de FastAPI mientras corre. | `mlb_advanced_metrics.py` |

### Cuotas y picks de valor
| # | Sev. | Error | Dónde |
|---|---|---|---|
| V1 | 🟠 | Los picks usan la probabilidad pura del modelo contra la del casino, sin calibrar. Así aparecen EV de +50% a +98% en underdogs, que casi siempre son errores del modelo. | `value_calculator.py` |
| V2 | 🟠 | El emparejamiento con The Odds API usa los **primeros 4 caracteres** del nombre (`ilike '%New %'`): "New York" se confunde con "New Orleans", "Los " con "LA", y también "Real", "Manc"… "LA Clippers" no empata con "Los Angeles Clippers". | `data_ingestion.py` (`_find_game_by_teams_approx`) |
| V3 | 🟡 | La "mejor cuota" se elige con `min(abs(odds))`, que en cuotas positivas escoge la **peor**. | `routes_predictions.py:161` |
| V4 | 🟡 | Se pide el mercado `btts`, que la API rechaza (422) en todas las ligas: **se pierden 9 requests por corrida**. | `odds_client.py` |
| V5 | 🟡 | Solo se usa DraftKings (o el primer libro); no hay comparación real entre casas. | `data_ingestion.py` |
| V6 | 🟡 | La `confidence` está fija en el código (0.75 / 0.85); no mide nada. | `simulation_engine.py` |

### Pipeline e infraestructura
| # | Sev. | Error | Dónde |
|---|---|---|---|
| P1 | 🟠 | El pipeline corre **en cada arranque**, y con `--reload` en cada archivo guardado. Gasta cuota. | `main.py:66-72` |
| P2 | 🟠 | No hay liquidación de picks: `picks.result` y `profit_loss` existen, pero nada los llena. Sin eso no se puede medir el ROI ni hacer backtest. | — |
| P3 | 🟡 | Cada corrida inserta predicciones nuevas sin borrar las viejas: hay 3,753 de MLB para 604 partidos. | `prediction_engine.py` |
| P4 | 🟡 | La zona horaria está fija en UTC−4; en noviembre la hora ET es UTC−5. | `routes_predictions.py:75,79,244`, `data_ingestion.py:268` |
| P5 | 🟡 | `apisports_client` ignora el campo `errors` de la respuesta, así que los fallos del plan pasan sin aviso. | `apisports_client.py:47` |
| P6 | ⚪ | `APP_DEBUG` activa el eco de SQL, que sale duplicado y hace ilegible el log. | `database.py:35` |
| P7 | ⚪ | Hay piezas sin usar: `XGBoostPredictor` y `DixonColesModel` son stubs con valores fijos, y `SoccerMetaModel` se instancia pero nunca se usa. | `app/models/`, `prediction_engine.py` |

### Frontend
| # | Sev. | Error | Dónde |
|---|---|---|---|
| UI1 | 🟠 | Si la API falla, se muestran **datos falsos** (`simulateData()`: Real Madrid vs Barcelona, partidos de NBA inventados) como si fueran reales. | `picks-table.component.ts:47-56` |
| UI2 | ⚪ | Falta revisar los filtros por liga cuando haya datos de varias ligas el mismo día. | — |

---

## 3. Fuentes de datos gratis por tipo de dato

Leyenda: ✅ ya probado en este proyecto · 🔎 por validar antes de integrarlo

| Dato | Fútbol | NBA | NFL | MLB |
|---|---|---|---|---|
| **Calendario y resultados** | ESPN scoreboard por liga ✅ (sustituye a API-Sports y ahorra cuota) | ESPN ✅ | ESPN ✅ | MLB StatsAPI ✅ |
| **Estadísticas de equipo, temporada actual** | ESPN standings (PJ, GF, GC, local/visita) ✅ · football-data.org, tier gratis (10 req/min; cubre PL, LaLiga, Serie A, Bundesliga, Ligue 1, Champions; **no** cubre Europa League ni Liga MX) 🔎 | `nba_api` ✅ | ESPN team stats ✅ · **nflverse** (`nflreadpy` / `nfl_data_py`): EPA real por jugada 🔎 | MLB StatsAPI ✅ |
| **Fuerza de equipo / Elo** | **ClubElo** (`api.clubelo.com`, CSV gratis; clubes europeos, ideal para cruces de Champions) 🔎 · Liga MX: Elo propio a partir de resultados | Temporada pasada en `nba_api` ✅ | Elo propio a partir de resultados históricos de nflverse 🔎 | — |
| **xG** | **Understat** (Premier, LaLiga, Serie A, Bundesliga, Ligue 1) 🔎 | — | — | Baseball Savant (xwOBA) 🔎 |
| **Lesiones** | API-Sports `/injuries` (cuesta cuota) · ESPN | ESPN injuries endpoint 🔎 | ESPN injuries / nflverse 🔎 | MLB StatsAPI (lista de lesionados) 🔎 |
| **Alineaciones** | API-Sports `/fixtures/lineups` (ya integrado) · Bzzoiro ✅ | — | — | Lanzadores probables en StatsAPI ✅ |
| **Cuotas** | The Odds API (500 créditos al mes) ✅ · línea de ESPN en el scoreboard ✅ | Línea de ESPN (DraftKings) ✅ | Línea de ESPN ✅ | Línea de ESPN 🔎 |
| **Modelo externo para mezclar** | Bzzoiro (ya integrado, 30%) ✅ | — | — | — |

**Estrategia de cuotas:** usar la línea de ESPN como fuente principal de spread y total en los deportes de EE. UU. (gratis e ilimitada), y dejar The Odds API solo para el moneyline multi-casa y para fútbol. Así el consumo mensual baja mucho.

---

## 4. Plan por fases

### Fase 0: Seguridad (la haces tú; yo te guío) 🔴
1. Regenerar las keys de Bzzoiro, API-Sports y The Odds API, y cambiar la contraseña de MySQL.
2. Limpiar el historial con `git filter-repo` y hacer force-push. Avisar a cualquiera que tenga un clon.
3. Confirmar que `.env` está en `.gitignore` (ya lo está desde `ede5638`).

### Fase 1: NBA ✅ hecha
Siguiente paso opcional: agregar lesiones desde ESPN (restar a cada equipo el valor de los jugadores fuera) y medir el modelo con 2–3 semanas de resultados (depende de P2).

### Fase 2: Fútbol, datos de entrada (F2, F3, F4, F5, F10, F11)
1. Aplicar tu corrección de ESPN (al menos 5 partidos y que traiga goles). **Necesito que me pegues el código como texto.**
2. Correr `diag_espn_cl.py` para confirmar con datos crudos. **También pégamelo.**
3. En Champions y Europa League, usar el récord de la **liga local** de cada equipo en ESPN (8–12 partidos en octubre) en vez del récord del torneo.
4. Integrar **ClubElo** como fuerza base de cada club; resuelve los cruces entre ligas.
5. Corregir `game_date_str` (`fixture.date`) y el slug por defecto.
6. Mover el calendario de fútbol a ESPN (ahorra cuota) y dejar API-Sports para alineaciones y H2H.

### Fase 3: Fútbol, modelo (F1, F6–F9) — requiere tu visto bueno
1. Suavizado bayesiano: `ataque = (gf × pj + promedio_liga × k) / (pj + k)`, con k ≈ 5.
2. Promedio de goles **por liga**, calculado de los datos.
3. Topar la fuerza relativa entre 0.4 y 2.5 y los goles esperados en 4.0 por equipo.
4. Ventaja local multiplicativa (×1.10 a ×1.20 según la liga) y quitar el piso artificial de 0.3.
5. Elo desde ClubElo en vez de win%.
6. **Validar:** repetir los partidos de Champions de septiembre y octubre que ya tienen resultado. Meta: ningún marcador predicho arriba de 4 y un error menor que el del modelo actual.

### Fase 4: NFL (N1–N4)
1. Guardar la fecha real de cada partido (sacada de `game_time` en ET) y eliminar el respaldo de "semana completa".
2. Modelo nuevo con los datos que ya se descargan: puntos anotados y permitidos por partido, ajustados por el rival y suavizados (igual que NBA), más ventaja local de ~1.5 puntos.
3. Siguiente nivel: EPA real desde nflverse, lesiones del QB y descanso.
4. Backtest con las 9 semanas jugadas de 2026.

### Fase 5: Pipeline y cuota (P1, P3, P5, P6, V4)
1. Variable `RUN_PIPELINE_ON_STARTUP=false` y un endpoint `POST /api/admin/run-pipeline?sports=nba,soccer`. Opcionalmente, una corrida programada 2 veces al día.
2. Quitar `btts` de la llamada a The Odds API (−9 requests por corrida).
3. Reemplazar la predicción vieja en vez de insertar duplicados, y limpiar las que ya existen.
4. Registrar en el log el campo `errors` de API-Sports y las requests restantes (`x-ratelimit-requests-remaining`).
5. Apagar el eco de SQL.

### Fase 6: Picks de valor y medición (V1–V3, V5, V6, P2)
1. **Liquidación automática:** al terminar cada partido, llenar `result` y `profit_loss`. Es la base para medir todo lo demás.
2. Mezclar el modelo con el mercado: `p_final = w × p_modelo + (1 − w) × p_mercado_sin_vig`, con w ≈ 0.3–0.5 calibrado por deporte. Topar el EV y exigir un edge mínimo.
3. Emparejar equipos por nombre normalizado o tabla de alias, no por los primeros 4 caracteres.
4. Corregir la elección de "mejor cuota" y comparar varias casas de verdad.
5. `confidence` calculada a partir del tamaño de muestra y la distancia contra el mercado.
6. Un dashboard de rendimiento por deporte: acierto, ROI, Brier score y CLV.

### Fase 7: Frontend (UI1, UI2, S2)
1. Quitar `simulateData()` y mostrar un estado de error real.
2. URL de la API por entorno (`environment.ts`).
3. Revisar los filtros por liga y por deporte con datos reales.

---

## 5. Orden recomendado

1. **Fase 0**, ya.
2. **Fase 5.1 y 5.2**: dejar de gastar cuota (rápido).
3. **Fases 2 y 3**: fútbol, que es lo que más se nota.
4. **Fase 4**: NFL.
5. **Fase 6.1**: liquidación de picks, para poder medir todo.
6. Fases 6 (resto) y 7.

> Cada fase se prueba aislada (`run_predictions_for_date(..., sports=[...])`), sin relanzar el pipeline completo.
