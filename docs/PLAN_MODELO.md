# OmniPicks: plan del motor de pronóstico (v2)

**Fecha:** 10-nov-2026
**Objetivo:** pronósticos **calibrados** de ganador, margen, total y marcador para NBA, fútbol, NFL y MLB, usando solo datos gratis. Los picks de valor se publican únicamente cuando el modelo, después de compararse con el mercado, todavía ve ventaja.

> **Meta realista.** Los modelos públicos más serios (FiveThirtyEight, nfelo y otros) quedan **cerca** de la línea de cierre de Las Vegas, no muy por encima. La línea de cierre es el mejor rival que existe. Por eso este plan mide todo contra ella, y la meta 1 es quedar a la par del mercado con probabilidades bien calibradas. La ventaja vendrá de lo que el mercado tarda en incorporar: lesiones de última hora, descansos y muestras chicas al inicio de temporada.

---

## 1. Arquitectura: la misma para todos los deportes

```
 Fuentes gratis ──► 1. Histórico en BD ──► 2. Ratings de fuerza ──► 3. Modelo del partido
 (ESPN, nba_api,     (varias temporadas,    (jugador / equipo,       (media + distribución
  nflverse, etc.)     sin fuga de datos)     con decaimiento)         de margen y total)
                                                                           │
   6. Evaluación ◄── 5. Mezcla con mercado ◄── 4. Calibración ◄── Simulación Monte Carlo
   (backtest,         (prob. sin vig,           (isotónica,         o matriz de marcadores
    CLV, ROI)          peso aprendido)           por deporte)         exacta (fútbol)
```

**Capa 1. Histórico.** Hoy el sistema solo guarda los partidos de hoy y mañana. Para entrenar y medir hacen falta **3 a 5 temporadas** por deporte: resultados, estadísticas por partido y, cuando exista, la línea de cierre. Cada dato se guarda con la fecha en que se conocía (*point-in-time*) para no "ver el futuro" en el backtest.

**Capa 2. Ratings de fuerza.** Es el corazón de cada deporte (sección 2). Siempre con tres elementos:
- **decaimiento temporal**: los partidos recientes pesan más;
- **regresión a la media** (prior bayesiano), para no exagerar con muestras chicas, que es justo lo que hoy rompe al fútbol;
- **ajuste por rival**: no es lo mismo meter 3 goles al último lugar que al líder.

**Capa 3. Modelo del partido.** Dos piezas:
- **Modelo estructural**: fórmula interpretable que convierte ratings en un margen y un total esperados.
- **Corrección con ML**: LightGBM o XGBoost entrenado sobre los **residuos** del estructural, con factores de contexto (descanso, viaje, altitud, back-to-back, lesiones, clima). Se aprende de los datos y no se adivina a mano.

La salida es una **distribución**, no un número suelto: Monte Carlo o, en fútbol, la matriz exacta de marcadores.

**Capa 4. Calibración.** Regresión isotónica ajustada en *walk-forward*. Si el modelo dice 70%, debe ganar 7 de cada 10 veces.

**Capa 5. Mezcla con el mercado.**
`p_final = w·p_modelo + (1−w)·p_mercado_sin_vig`, con `w` aprendido en el backtest por deporte (típicamente 0.2–0.5). El edge mínimo y la fracción de Kelly (¼ Kelly, con tope) se fijan con el backtest. Esto elimina los "EV +98%" falsos.

**Capa 6. Evaluación** (antes y después de cada cambio):

| Métrica | Qué mide | Aplica a |
|---|---|---|
| Log loss / Brier | Calidad de las probabilidades | Todos |
| RPS (Ranked Probability Score) | Probabilidades de 1-X-2 | Fútbol |
| MAE de margen y total | Error en puntos o goles | Todos |
| Curva de calibración | ¿El 70% gana el 70% de las veces? | Todos |
| **CLV** (contra la línea de cierre) | ¿Le ganamos al precio final del mercado? | Picks |
| ROI simulado (¼ Kelly) | Rentabilidad histórica | Picks |

Un modelo nuevo **solo entra a producción si mejora el log loss del actual** en el backtest walk-forward.

---

## 2. El mejor enfoque por deporte

### 🏀 NBA: modelo basado en jugadores
**Por qué.** FiveThirtyEight pasó a un sistema donde la fuerza del equipo sale **completamente de los jugadores** disponibles. Mantiene un rating "actual", que descuenta lesionados, y uno "a plantilla completa". Para proyectar minutos mezcla 60% del promedio reciente y 40% de la rotación esperada. Los sistemas de jugador más predictivos (DARKO, EPM) combinan inferencia bayesiana con filtros tipo Kalman que actualizan el rating de cada jugador día a día.

**Algoritmo:**
1. **Rating por jugador** (ofensivo y defensivo por 100 posesiones):
   - Base: box score de `nba_api` (fórmula tipo BPM) más plus-minus regularizado (RAPM con regresión ridge sobre stints de play-by-play).
   - Actualización bayesiana diaria, tipo filtro de Kalman: cada partido mueve el rating según su ruido, y el prior es la temporada pasada más la edad.
2. **Proyección de minutos**: 60% del promedio de los últimos partidos y 40% de la rotación esperada. A quien está fuera se le dan 0 minutos y se redistribuyen.
3. **Rating del equipo** = Σ (minutos × rating) / 48, más el ajuste de ritmo.
4. **Partido**: margen = Δrating × posesiones/100 + local (~2.5) + descanso o back-to-back + viaje y altitud (Denver, Utah).
5. **Corrección con LightGBM** sobre los residuos, y simulación del margen (σ ≈ 12.5) y del total.
6. Segunda vista: el modelo de equipo actual (`nba_possessions_v1`) queda como respaldo y como variable de entrada.

**Datos:** `nba_api` (box scores, play-by-play, stints y alineaciones por jugador) ✅, lesiones de ESPN 🔎, calendario de ESPN ✅. Históricos de cuotas: buscar un dataset gratis 🔎.

### ⚽ Fútbol: Dixon-Coles con xG y Elo de clubes
**Por qué.** Dixon-Coles es de los enfoques basados en goles más conocidos y de mejor desempeño. Corrige la poca frecuencia de los empates con marcador bajo que el Poisson simple no captura. En estudios recientes, los **pi-ratings** combinados con gradient boosting (CatBoost) lograron el mejor RPS frente a otros modelos. La librería gratis `penaltyblog` ya implementa Dixon-Coles, Poisson bivariado y otros modelos de goles.

**Algoritmo:**
1. **Dixon-Coles con decaimiento temporal**: ajustado por liga con 3 temporadas de resultados, con el partido de hace un año pesando ~30%. Da ataque y defensa por equipo **ya ajustados por rival y suavizados**, lo que elimina de raíz los 15 goles.
2. **Variante con xG**: el mismo modelo ajustado con xG (Understat) en vez de goles. Es más estable con pocos partidos. Se promedian ambos.
3. **Cruces entre ligas** (Champions, Europa League): ClubElo como puente de fuerza entre ligas, y el ajuste por liga sale de los resultados europeos.
4. **Pi-ratings** como variable de entrada adicional.
5. **Corrección con CatBoost o LightGBM** con contexto: rotación por Champions, alineaciones confirmadas (API-Sports o Bzzoiro), bajas, descanso.
6. **Salida exacta**, sin Monte Carlo: la matriz de probabilidades de marcadores 0-0…10-10, de donde salen 1X2, over/under, ambos anotan y el marcador más probable.
7. Bzzoiro sigue como opinión externa; su peso se aprende en la Capa 5.

**Datos:** ESPN (resultados y standings) ✅. **football-data.co.uk** (resultados históricos **con cuotas de cierre**, gratis, para backtest de las ligas europeas) 🔎. Understat (xG de 5 ligas) 🔎. ClubElo 🔎. Liga MX: resultados de ESPN; sin xG gratis confirmado.

### 🏈 NFL: Elo con EPA real y ajuste de QB
**Por qué.** nfelo, uno de los modelos públicos más serios, parte de Elo y le agrega ratings iniciales derivados de las líneas de victorias del mercado, un modelo histórico de ventaja local y EPA ponderado. Con eso queda a la par de la línea de cierre o ligeramente por encima. Un Elo limpio llega a ~62–65% de acierto en temporada regular.

**Algoritmo:**
1. **Elo por margen de victoria**, con regresión de 1/3 a la media entre temporadas y prior desde las líneas de victorias de pretemporada.
2. **EPA/jugada real** desde nflverse, ofensiva y defensiva, con decaimiento y peso mayor a pases y a situaciones neutrales, al estilo WEPA.
3. **Ajuste por QB**: valor del QB titular contra su suplente; si cambia el QB, el rating se mueve.
4. Contexto: descanso o semana bye, viaje, viento y lluvia (ya se descarga el clima), lesiones.
5. Margen ~ Normal (σ ≈ 13.5) y total con su propia σ, más una corrección con ML.
6. Calibración isotónica.

**Datos:** **nflverse** (`nflreadpy`): play-by-play con EPA desde 1999 y calendarios **con spread y total de cierre** para backtest, gratis 🔎. Calendario de ESPN ✅.

### ⚾ MLB: lanzador abridor, lineup y simulación
**Por qué.** El enfoque clásico combina bateador, lanzador y liga con log5. Modelos más finos de enfrentamiento (bayesianos jerárquicos) suman alrededor de 1 victoria por temporada de precisión. Los simuladores por turno al bat permiten modelar abridor, bullpen, estadio y clima. El modelo actual ya funciona (58.6%), así que aquí se mejora sin rehacer.

**Algoritmo:**
1. **Calidad del abridor** con métricas predictivas (xFIP, SIERA, xERA, K-BB%) en vez de la ERA cruda, que tiene mucho ruido. Fuente: Statcast y FanGraphs vía `pybaseball`.
2. **Ofensiva por lineup confirmado** (wOBA/xwOBA por jugador contra mano del lanzador).
3. **Bullpen real**: lanzamientos de los últimos 3 días desde StatsAPI. Hoy ese dato es aleatorio.
4. Factores de estadio completos para los 30 parques, más clima.
5. Simulación por entradas: abridor ~5–6 entradas y luego bullpen, con Poisson/negativa binomial por entrada.
6. Corrección con ML y calibración.

**Datos:** MLB StatsAPI ✅, `pybaseball` (Statcast, FanGraphs) 🔎, Retrosheet (histórico) 🔎.

---

## 3. Stack técnico (todo gratis)

| Uso | Herramienta |
|---|---|
| Modelos de goles (fútbol) | `penaltyblog` (Dixon-Coles, Poisson bivariado), `scipy` |
| ML tabular | `lightgbm` / `xgboost` / `catboost`, `scikit-learn` (calibración isotónica) |
| Bayesiano / Kalman | `numpy` propio; opcionalmente `pymc` |
| Datos | `nba_api`, `nflreadpy`, `pybaseball`, `statsapi`, ESPN, football-data.co.uk, Understat, ClubElo |
| Entrenamiento | Scripts offline en `ml/scripts/`; modelos en `ml/saved_models/` (ya existen las carpetas) |
| Producción | El pipeline diario **solo carga** los modelos entrenados y predice (rápido y sin gastar cuota) |

---

## 4. Fases

| Fase | Entregable | Criterio de éxito |
|---|---|---|
| **A. Base de medición** *(obligatoria primero)* | Liquidación automática de picks, tablas de histórico, script de backtest walk-forward y reporte de métricas | Poder medir log loss, Brier, CLV y ROI de cualquier modelo |
| **B. Fútbol v2** (Dixon-Coles + xG + ClubElo) | Reemplaza a `_simulate_soccer` | Ningún marcador > 4 goles esperados; RPS mejor que el modelo actual en el backtest de 2024-26 |
| **C. NBA v2** (jugadores + lesiones) | Reemplaza a `nba_possessions_v1`, que queda de respaldo | Log loss mejor que v1; MAE de margen ≤ 11 puntos |
| **D. NFL v2** (Elo + EPA + QB) | Reemplaza al modelo constante actual | Acierto ≥ 60% en el backtest; Brier cercano a la línea de cierre |
| **E. MLB v2** (abridor xFIP, bullpen real, lineup) | Mejora el modelo actual | Log loss mejor que el actual (58.6% de acierto) |
| **F. Calibración y mezcla con mercado** | Capa 4 y 5 para todos los deportes | Curvas calibradas; CLV ≥ 0 en picks; adiós a los EV falsos |
| **G. Panel de rendimiento** | Vista en el front: acierto, ROI, CLV por deporte y modelo | Transparencia total |

Orden sugerido: **A → B → C → D → F → E → G**. Fútbol va primero porque es lo que más se nota roto. MLB va al final porque ya funciona y su temporada terminó.

---

## 5. Riesgos y cómo se manejan

| Riesgo | Mitigación |
|---|---|
| Fuga de datos en el backtest (usar información del futuro) | Variables *point-in-time* y validación walk-forward estricta |
| Sobreajuste del ML | Pocas variables de contexto y regularización; el ML solo corrige residuos |
| Fuentes gratis que cambian o bloquean (stats.nba.com, scraping de Understat) | Caché local, respaldo de ESPN, reintentos con espera |
| Sin cuotas históricas gratis en NBA | Medir log loss y MAE sin cuotas; CLV solo hacia adelante con las cuotas que ya se guardan |
| Lesiones de último minuto | Re-correr las predicciones de NBA y NFL 1 h antes del partido (corrida ligera, sin cuota de pago) |

---

## 6. Decisiones que necesito de ti
1. **Prioridad**: ¿te parece el orden A → B (fútbol) → C (NBA)…, o empezamos por otro deporte?
2. **Objetivo principal**: ¿acertar ganador y marcador (pronóstico) o encontrar valor para apostar (picks)? Afecta cuánto peso se le da al mercado.
3. **Dependencias nuevas**: ¿está bien instalar `penaltyblog`, `lightgbm`, `nflreadpy` y `pybaseball` en el venv?
