# OmniPicks: estado del proyecto

**Actualizado:** 11-nov-2026

Este documento resume todo lo hecho, lo que está en curso y lo que falta. Los planes detallados están en la misma carpeta:

| Documento | Contenido |
|---|---|
| [`PLAN.md`](PLAN.md) | Auditoría completa (~40 errores por gravedad), fuentes de datos gratis y plan general en 7 fases |
| [`PLAN_MODELO.md`](PLAN_MODELO.md) | Arquitectura del motor de pronóstico para los 4 deportes y los mejores enfoques por deporte |
| [`PLAN_NBA.md`](PLAN_NBA.md) | Plan detallado del modelo NBA v2 (jugadores, bajas, simulación, backtest), decisiones y estado por fase |
| [`NBA_BACKTEST.md`](NBA_BACKTEST.md) | Reporte del backtest walk-forward de NBA v2 (lo genera `ml/scripts/nba_backtest.py`) |

---

## 1. Repositorios y ramas

| Repo | Ruta local | Rama | Estado |
|---|---|---|---|
| Backend (FastAPI) | `~/PycharmProjects/omnipicks_backend` | `main` | NBA v1 y NBA v2 N0 ya están en GitHub (PR #1). El commit de NBA v2 completo (N0–N6) está **solo local, sin push** |
| Frontend (Angular 18) | `~/Documents/GitHub/Angular_proyects/omnipicks` | `main` | Rediseño, paleta en negro y bloque de bajas: 3 commits **solo locales, sin push** |

Desde el 11-nov se trabaja directo en `main` (sin ramas). Las ramas viejas (`nba-v1-espn`, `nba-v2`, `ui-redesign`) ya están incluidas en `main`.

---

## 2. Lo que ya está hecho

### 2.1 NBA v1: partidos y modelo real (rama `nba-v1-espn`)
**Antes:** en la base no había ni un solo partido de NBA, y la predicción salía de un Elo **aleatorio**.

**Causas encontradas:**
- API-Sports sí trae NBA, pero llama a la liga `"standard"`, y el filtro solo aceptaba `"NBA"`.
- También usa `visitors` en vez de `away`, así que se perdía el equipo visitante.
- El simulador usaba `random.uniform(1450, 1600)` como Elo.
- Los ratings reales se calculaban pero nadie los leía, y pedían la temporada pasada (`2025-26`).

**Cambios:**
- `app/services/espn_nba_client.py` (nuevo): calendario diario desde ESPN, gratis y sin cuota.
- `data_ingestion.py`: NBA se ingiere desde ESPN. Ahorra 2 requests de API-Sports por corrida.
- `nba_advanced_metrics.py`: `nba_api` con la temporada calculada a partir de la fecha y caché por corrida. Usa la temporada pasada como prior y calcula los back-to-backs reales desde ESPN.
- `simulation_engine.py`: modelo `nba_possessions_v1`, que multiplica posesiones por eficiencia (ataque propio contra defensa rival), con ventaja de local, back-to-back y suavizado de inicio de temporada.
- `value_calculator.py`: distribución Normal para totales de NBA y NFL. Poisson se desbordaba con ~220 puntos y además subestimaba la varianza de NFL.
- `prediction_engine.py`: filtro opcional `sports=[...]` para correr un solo deporte.

**Resultado:** 16 partidos predichos. El error del spread contra el mercado bajó de ~5 a ~2.4 puntos, y los totales salen casi iguales a los del mercado.

### 2.2 NBA v2, fase N0: histórico propio (rama `nba-v2`) ✅
- **Tablas** en `app/db_models.py`: `nba_games`, `nba_player_games`, `nba_stints`, `nba_injury_reports`, `nba_player_ratings` y `nba_player_seasons` (edad, estatura y peso por temporada).
- **`app/services/nba_history.py`:**
  - partidos y box scores por temporada;
  - **stints** (tramos con los mismos 10 jugadores en cancha), armados con las rotaciones y el play-by-play, con puntos y posesiones de cada lado;
  - el **crudo de cada partido se guarda en disco** (`ml/datasets/nba_raw/`, ~10 KB por partido), así que rearmar los stints ya no requiere volver a descargar.
- **`ml/scripts/nba_backfill.py`:** carga reanudable de 2022-23 a la fecha (`--rebuild-stints` rearma todo).
- **`app/services/nba_injuries.py`:** lesiones de ESPN con historial point-in-time (solo guarda cambios de estado) y probabilidad inicial de jugar según el estado.
- **`app/services/nba_scheduler.py`** (nuevo): tareas dentro del servidor, gratis y sin cuota:
  - lesiones **cada 30 minutos**;
  - a las 04:30 ET: partidos, box scores y stints de ayer, y con eso los ratings de jugador vigentes.
  - Se apaga con `NBA_SCHEDULER_ENABLED=false`. A mano: `python ml/scripts/nba_daily_update.py`.

**⚠️ Corrección importante (10-nov):** los stints de la primera carga tenían un error. Los puntos se sacaban de la diferencia de marcador entre eventos ordenados por `actionNumber`; cuando el play-by-play trae eventos fuera de orden o marcadores con erratas, los puntos se contaban dos veces. **~15% de los partidos no cuadraban** con el marcador real (el "validados, coinciden" de la versión anterior de este documento salió de una muestra muy chica). Ahora los puntos se suman por evento (canasta o tiro libre anotado) en orden cronológico, y cada partido se compara con su marcador al cargarlo. Quedan fuera ~0.2% de partidos cuyas rotaciones vienen corruptas desde la NBA.

**Datos cargados:**
| Tabla | Filas |
|---|---|
| `nba_games` | 5,409 partidos (2022-23 a 2026-27, temporada regular y playoffs) |
| `nba_player_games` | 115,869 filas jugador-partido |
| `nba_stints` | 159,969 tramos de 5,398 partidos (20 partidos no cuadran con el marcador por rotaciones corruptas en el origen y quedan fuera del RAPM) |
| `nba_player_seasons` | 2,724 fichas jugador-temporada |
| `nba_injury_reports` | historial desde el 10-nov-2026 (~90 reportes vigentes) |
| `nba_player_ratings` | 404,172 ratings: 643 fechas (24-oct-2023 a hoy), uno por jugador activo y día |

### 2.2b NBA v2, fase N1: rating por jugador ✅
**`app/services/nba_ratings.py`**: rating ofensivo y defensivo de cada jugador **por fecha**, en puntos por 100 posesiones sobre el promedio. El rating de un día usa solo lo ocurrido antes (sirve tal cual para el backtest). Es una sola regresión sobre los stints, recalculada cada día de partidos:

- **Perfil de box score** (2a + 2c): tasas por 100 posesiones, minutos por partido y edad, con un filtro bayesiano con descuento: cada partido pesa según las posesiones jugadas, lo viejo se descuenta y con pocos datos manda el nivel de un suplente.
- **Plus-minus ajustado (RAPM)** (2b): lo que el equipo rinde con el jugador en cancha y el box score no explica, con penalización ridge y decaimiento temporal.
- Recalcular todo el histórico tarda ~20 segundos, así que no se guarda estado intermedio.

Scripts: `ml/scripts/nba_ratings_build.py` (calcula y guarda) y `ml/scripts/nba_ratings_eval.py` (evaluación, `--grid`, `--ablation`).

**Criterio de N1: ¿predicen el margen mejor que los ratings de equipo?** Sí. Caminando hacia adelante y **sin saber quién falta**:

Evaluación del 24-oct-2023 al 9-nov-2026 (4,095 partidos):

| Modelo | log loss | acierto | error medio del margen |
|---|---|---|---|
| Elo | 0.617 | 65.4% | 11.39 |
| Ridge de equipo (márgenes con decaimiento) | 0.606 | 66.8% | 11.25 |
| **Ratings de jugador, sin saber quién falta** | **0.605** | **66.8%** | **11.20** |
| Ratings de jugador, sabiendo quién falta | 0.592 | 68.3% | 11.09 |

Sin información de bajas, los ratings de jugador quedan apenas por encima de un buen modelo de equipo; en 2025-26 incluso un poco por debajo (0.606 contra 0.601). La ventaja real aparece cuando se sabe quién falta, que es justo para lo que sirven.

- **Ablación:** solo RAPM o solo box score rinden peor que juntos; las dos piezas suman (0.610 solo RAPM, 0.616 solo box score, 0.605 juntos). **La edad no aportó** (0.6048 sin ella contra 0.6050) y se quitó.
- Los hiperparámetros casi no mueven el resultado, así que se dejaron los valores por defecto (sin sobreajustar).

### 2.2c NBA v2, fase N2: bajas (primera parte) 🟡
**`app/services/nba_lineups.py`**:
- **Plantillas** al día a partir de los box scores (quién está en cada equipo, minutos habituales = 60% reciente + 40% rol, participación reciente).
- **Reparto de minutos** (3b): los minutos de quien falta se reparten entre los disponibles en proporción a lo que juegan, con tope por jugador. Quien vuelve de una ausencia larga juega al 75%.
- **Probabilidad de jugar** (3a): un "Questionable" cuenta con el 50% de sus minutos.
- **Impacto de cada baja**: fuerza del equipo con él menos sin él, en puntos de margen, y a quién van sus minutos. `injury_impacts()` ya lo calcula con el reporte de lesiones vigente (es lo que mostrará el front).

**Medición** (`ml/scripts/nba_minutes_eval.py`):

Backtest del modelo completo (ver la sección 2.2d):

| Supuesto | log loss | error medio del margen |
|---|---|---|
| Sin saber quién falta | 0.603 | 11.16 |
| **Con lo que diría el reporte de lesiones** | **0.594** | **11.02** |
| Sabiendo exactamente quién juega (techo) | 0.592 | 11.01 |

Con el reporte de lesiones se recupera casi toda la mejora posible. El error de minutos por jugador baja de 6.8 a 5.5. **Uso de los ausentes (3c):** cada 10 minutos de rotación ausentes, el equipo rinde ~0.2 pts menos de lo que suma su gente; sí mejoró el backtest y entró.

**Medido y descartado:** rendimiento observado sin el jugador (3d), reparto de minutos aprendido de los partidos en que faltó cada jugador (falla 5.7 min por jugador contra 5.5 del proporcional) y ajuste por racha del equipo. Ninguno mejoró.

Falta de N2: calibrar la probabilidad de jugar con el historial de reportes (necesita semanas de datos), descanso de estrellas en back-to-back y química de quintetos (3e, solo si aporta).

### 2.2d NBA v2, fases N3 a N6: simulador, contexto, backtest y producción ✅
- **N3, simulador** (`app/services/nba_game_model.py`): arma todos los escenarios de quién juega entre los que tienen duda (exactos hasta 8 jugadores; con más, sorteados) y saca 10,000 marcadores. Da probabilidad, margen y total esperados, rango del 80% y marcador más probable.
- **N4, contexto:** descanso, back-to-back, 3 en 4 noches, viaje, zona horaria, altitud, inicio y final de temporada y playoffs, con una regresión ridge sobre los errores. LightGBM salió peor y la calibración isotónica no mejoró; ninguno de los dos entró.
- **N5, backtest** (`ml/scripts/nba_backtest.py`, reporte en [`NBA_BACKTEST.md`](NBA_BACKTEST.md)): del 24-oct-2023 al 9-nov-2026, 4,093 partidos, sin información futura.

| Modelo | log loss | Brier | acierto | error del margen | error del total |
|---|---|---|---|---|---|
| **NBA v2 (`nba_players_v2`)** | **0.594** | **0.204** | **67.7%** | **11.02** | **14.81** |
| `nba_possessions_v1` (el anterior) | 0.607 | 0.210 | 67.0% | 11.23 | 15.03 |
| Elo | 0.617 | 0.214 | 65.4% | 11.38 | — |

  v2 tiene mejor log loss que v1 en las 4 temporadas. En partidos con una baja importante la diferencia crece: 0.596 contra 0.613. Ojo: en 2025-26 v1 acertó más ganadores (69.1% contra 68.1%), aunque con peor probabilidad. Está calibrado: cuando dice 70–75%, el favorito ganó 75%.

- **N6, producción:**
  - `app/services/nba_v2.py` predice los partidos del día. `prediction_engine.py` lo usa y deja v1 de respaldo si falla (o con `NBA_MODEL=v1`).
  - Las predicciones NBA ya no se duplican: se reemplaza la anterior.
  - Cuando cambia el reporte de lesiones, se vuelven a predecir los partidos de hoy.
  - Los parámetros entrenados están en `ml/saved_models/nba_v2.json` (versionado).
  - En el front, el análisis de cada partido muestra el bloque **Bajas**: cada lesionado, su estado, sus minutos, cuántos puntos pierde su equipo sin él y a quién van sus minutos.
  - Probado: el motor predijo los 10 partidos del 11-nov; el endpoint y el front muestran v2.


### 2.3 Frontend rediseñado (rama `ui-redesign`)
- **Tema:** claro por defecto y modo oscuro con un interruptor. La preferencia se guarda y no parpadea al cargar.
- **Paleta:** neutra en negro. El favorito siempre se resalta en negro (blanco en modo oscuro), y el color se usa solo con significado: verde, rojo y amarillo.
- **Layout:** barra lateral con íconos propios y menú desplegable en celular.
- **Cartelera:** una tarjeta por partido con probabilidad, marcador proyectado, spread, total y línea del mercado, más buscador y filtros. Al abrir **"Ver análisis"** muestra:
  - comparación local contra visita;
  - modelo contra mercado, sin el margen del casino;
  - diferencias con el mercado;
  - contexto del partido.
- **Resumen, Modelo vs Mercado y Bankroll** usan **solo datos reales**. Se quitaron:
  - el ROI falso de "+14.2%";
  - los casinos de ejemplo;
  - el "58/102 picks";
  - los partidos inventados que aparecían cuando fallaba la API.
- Estados de carga, error con botón de reintentar y vacío; fechas en español.

### 2.4 Diagnóstico
- La prueba confirmó que API-Sports no restringe la NBA por plan: con la cuota de 100 requests diarias sí devuelve partidos y no da errores.
- **MLB funciona bien:** acertó al ganador en 58.6% de 1,657 predicciones.
- **NFL acierta 41%** (11 de 27): el simulador usa `wind_speed` y `qb_diff`, que nadie calcula, así que la predicción es casi constante.
- **Fútbol da marcadores irreales** (hasta 1.07–15.38) por la fórmula sin topes y las muestras de 1 partido.

---

## 3. Operación
- **Nada corriendo en segundo plano.** El histórico ya está completo; desde ahora solo se agrega lo nuevo de cada día (lo hace el servidor a las 04:30 ET).
- **Levantar los servidores:**
  ```bash
  # Backend sin gastar cuota (no corre el pipeline completo al arrancar; las tareas NBA sí corren)
  cd ~/PycharmProjects/omnipicks_backend && RUN_PIPELINE_ON_STARTUP=false .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
  # Frontend
  cd ~/Documents/GitHub/Angular_proyects/omnipicks && npx ng serve --port 4200
  ```
  Sin `RUN_PIPELINE_ON_STARTUP=false`, el backend corre el pipeline completo al arrancar y gasta requests de API-Sports y The Odds API. No uses `--reload`.
- **A mano:**
  - `python ml/scripts/nba_daily_update.py` actualiza el histórico, los ratings y las lesiones.
  - `python ml/scripts/nba_backtest.py` reentrena y regenera el reporte (~4 min).
- **Si la Mac se duerme** durante una carga larga, la descarga se pausa y continúa al despertar.

---

## 4. Lo que falta

### 4.1 NBA v2, ~90% hecho (N0–N6 listos)
| Fase | Qué es | Tamaño | Estado |
|---|---|---|---|
| N0 | Tablas, histórico, stints y lesiones | — | ✅ hecho |
| N0 | Actualización diaria del histórico y revisión de lesiones **cada 30 min** | chico | ✅ hecho |
| **N1** | Rating por jugador: prior por box score + plus-minus ajustado (RAPM, ridge sobre stints) + actualización diaria tipo Kalman | **grande** | ✅ hecho y validado |
| **N2** | Bajas: probabilidad de jugar, redistribución de minutos y de uso, rendimiento real con y sin el jugador, química de quintetos | **grande** | ✅ hecho; 3d descartado por el backtest; falta calibrar la probabilidad de jugar con datos propios y 3e |
| N3 | Simulador: sortea quién juega y luego simula posesiones y puntos (~10,000 corridas) | mediano | ✅ hecho |
| N4 | Contexto (descanso, viaje, altitud, momento de temporada) + corrección con LightGBM + calibración isotónica | mediano | ✅ hecho (ridge; LightGBM e isotónica no mejoraron) |
| N5 | Backtest walk-forward de 3 temporadas contra `nba_possessions_v1` | mediano | ✅ hecho: v2 gana en log loss en todas las temporadas |
| N6 | Integración con el pipeline, el endpoint y el front (**solo el impacto de las bajas**) | mediano | ✅ hecho; falta verlo correr solo una semana |
| N7 | Proyecciones por jugador (puntos, rebotes, asistencias) | mediano | ❌ opcional |

**Lo que queda de NBA:**
- Dejar correr el servidor una semana y revisar que la actualización diaria y las lesiones no fallen (criterio de N6).
- Calibrar la probabilidad de jugar por estado con el historial de reportes (necesita semanas de datos).
- Comparar contra el mercado hacia adelante (no hay líneas históricas gratis de NBA).
- Ritmo según los jugadores disponibles, descanso de estrellas en back-to-back y química de quintetos (3e).

**Decisiones ya tomadas:**
- El mercado **no entra al modelo**; solo se usa para comparar.
- Se toman en cuenta las bajas y el rendimiento con y sin cada jugador.
- El front mostrará solo el impacto de las bajas.
- Las lesiones se revisan cada 30 minutos.
- Histórico de 2022-23 a la fecha.
- Se puede instalar lo que haga falta.

**Resuelto:** `lightgbm` ya carga (se instaló `libomp` con Homebrew).

### 4.2 Frontend
| Pieza | Estado |
|---|---|
| Impacto de las bajas en cada partido | ✅ hecho |
| Rendimiento del modelo e historial de resultados (hoy dicen "Pronto") | ❌ depende de la liquidación de picks |
| Logos reales de los equipos (hoy son abreviaturas) | opcional; hay que agregarlos al endpoint |
| URL de la API por entorno (hoy fija en `localhost:8000`) | ❌ |

### 4.3 Resto del sistema (detalle en `PLAN.md`)
| Prioridad | Tema | Qué hacer |
|---|---|---|
| 🔴 **urgente** | **Seguridad** | Regenerar las 3 API keys (Bzzoiro, API-Sports y The Odds API) y la contraseña de MySQL, que siguen en el historial de git (`521fc0f`), y limpiar el historial |
| 🟠 | Pipeline y cuota | ✅ ya hay flag `RUN_PIPELINE_ON_STARTUP`. Falta el endpoint para correrlo a mano, quitar `btts` (−9 requests por corrida), no duplicar predicciones en los otros deportes (NBA ya no duplica) y apagar el eco de SQL |
| 🟠 | Liquidación de picks | Llenar `result` y `profit_loss`; es la base para medir el ROI y el rendimiento |
| 🟠 | Fútbol | Aplicar la corrección de ESPN (al menos 5 partidos y que traiga goles; **falta que me pases el código**), correr `diag_espn_cl.py` (**también falta**) y usar Dixon-Coles + ClubElo |
| 🟠 | NFL | Fechas reales en lugar de "semana completa con la fecha de hoy" y un modelo que use sus datos (Elo + EPA real de nflverse) |
| 🟡 | Picks de valor | Emparejar equipos por alias (no por los primeros 4 caracteres), elegir bien la "mejor cuota", calcular una confianza real |
| 🟡 | Zona horaria | UTC−4 fijo en el código (en noviembre ET es UTC−5) |

---

## 5. Notas de operación
- **Cuota de API-Sports:** 100 requests al día. NBA ya no la usa; fútbol y MLB sí.
- **Fuentes gratis ya integradas:**
  - ESPN (calendario NBA, NFL y fútbol; lesiones NBA; récords);
  - `nba_api` (stats, rotaciones, play-by-play);
  - MLB StatsAPI.
- **Disco:** quedaban ~7 GB libres (97% usado). El histórico de NBA ocupa menos de 1 GB, pero conviene liberar espacio.
- **Push:** en los dos repos, `main` local va adelante de GitHub (backend: 1 commit; front: 3). Para subirlo:
  ```bash
  cd ~/PycharmProjects/omnipicks_backend && git push origin main
  cd ~/Documents/GitHub/Angular_proyects/omnipicks && git push origin main
  ```
  Antes de subir, regenera las API keys (ver 4.3), porque siguen en el historial.
