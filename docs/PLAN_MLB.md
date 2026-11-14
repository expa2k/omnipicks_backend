# OmniPicks: plan del modelo MLB (v2)

**Hoy (modelo viejo):** una fórmula con pesos fijos:
- 55% ERA del abridor + 45% ERA del bullpen;
- 40% pitcheo, 40% carreras anotadas y permitidas, 20% historial entre los dos equipos;
- parque, ±5% por el viento, localía fija (×1.04 / ×0.96) y simulación Poisson.

Solo corre con el pipeline viejo, que está apagado. Acertó el ganador en 58.6% de 1,657 predicciones, pero nunca se validó contra la línea.

**Objetivo:** el pronóstico más acertado posible (ganador, run line ±1.5, total y primeras 5 entradas), validado con backtest. El mercado no entra al modelo.

## Qué se cambia

| Pieza | Modelo viejo | v2 | Por qué |
|---|---|---|---|
| Abridor | ERA de la temporada | xFIP / SIERA / xERA con encogimiento, y entradas esperadas | La ERA tiene mucha suerte |
| Bullpen | ERA de relevo, peso fijo de 45% | Relevistas disponibles hoy (cansancio de 3 días) y entradas que le tocan | Un bullpen agotado cambia el partido |
| Ofensiva | Carreras por partido | Alineación del día (wOBA/xwOBA) contra la mano del abridor | Estrellas y zurdo/derecho importan |
| Historial entre los dos equipos | 20% | Fuera | Casi pura suerte |
| Parque | Factor fijo | Factor por parque (carreras y jonrones) | |
| Clima | ±5% por el viento | Viento, temperatura y techo aprendidos | |
| Localía | Fija | Aprendida (incluye batear en la 9ª) | |
| Pesos | A mano | Backtest 2015–2025 | |
| Resultado | Poisson | Distribución con más dispersión → moneyline, run line, total y 1–5 entradas | |
| Validación | Ninguna | Semana a semana; contra el cierre de 2015–2021 | |
| App | Solo pronóstico | Predicción congelada, liquidación, historial, y en la tarjeta los abridores, la alineación y el bullpen | |

## Fases

| Fase | Qué | Estado |
|---|---|---|
| 0 | Fuentes y descarga del histórico | ✅ |
| 1 | Abridores, alineación, bullpen y parque en un solo modelo de carreras | ✅ |
| 2 | Bullpen cansado y ventaja de mano | ✅ (efecto chico) |
| 3 | Distribución de carreras (binomial negativa) → moneyline, run line y total | ✅ |
| 4 | Backtest contra el cierre (2015–2021) | ✅ sin ventaja |
| 5 | Producción: StatsAPI (abridores probables, alineaciones), ESPN (líneas), tareas automáticas y front | |

## Fase 0: fuentes (13-nov-2026, `app/services/mlb_history.py`)

| Fuente | Qué trae | Cobertura |
|---|---|---|
| **Retrosheet**, game logs | Cada partido: marcador, parque, día/noche, estadísticas de equipo, **los 2 abridores** y **las 2 alineaciones** (9 bateadores y su posición) | 2010–2025: **37,343 partidos** (2020 = 898) |
| **MLB StatsAPI** | Calendario, abridores probables, alineaciones del día, y game logs por pitcher (IP, ER, K, BB, HR, bateadores enfrentados) | Histórico y en vivo |
| **Baseball Savant** | Métricas esperadas por temporada (xERA, xwOBA, xBA, xSLG) en CSV | Statcast desde 2015 |
| **SBR** | Momios de apertura y cierre: moneyline, run line y total | **2010–2021** (12 temporadas); de 2022 en adelante no hay |
| **ESPN** | Líneas de DraftKings de los partidos por jugar | Solo en vivo (no guarda el histórico) |

**Consecuencias:**
- La comparación contra el mercado se hace en 2015–2021. De 2022 en adelante se mide contra los resultados y en vivo.
- Los códigos de equipo de SBR (CUB, LAD…) y los de Retrosheet (CHN, LAN…) se emparejan en la fase 4.
- La temporada 2027 empieza a finales de marzo: la meta es tener la v2 en producción antes.

Uso de Retrosheet: "The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org."

## Fases 1–3: modelo de carreras (`app/services/mlb_model.py`, `ml/scripts/mlb_eval.py`)

Por cada equipo y partido:

    log E[carreras] = μ + localía + noche + parque + ataque[equipo, temporada] − defensa[rival, temporada]
                      + (1/9) Σ bateo[alineación del día] − abridor[rival]
                      + ventaja de mano · (fracción de la alineación con ventaja contra el abridor)
                      + cansancio · (lanzadores del bullpen rival en sus 2 juegos previos)

- **Ajuste:** cuasi-Poisson con matriz rala, ξ = 0.002 por día y ~3 años de historia. Ridge: bateadores 40, **abridor 150**, equipo 6, parque 10.
- **Carreras:** binomial negativa con dispersión r ≈ 4.4.
- **Mercados:** moneyline (con extras: el local gana 52% de los empates), run line y total.
- **Mano** de cada jugador (`load_handedness`): MLB StatsAPI más el registro de Chadwick (Retrosheet ↔ MLBAM), 5,361 jugadores.

**Backtest** (cada 14 días; log loss del ganador; 2015–2019 | 2021–2025):

| Modelo | 2015–2019 | 2021–2025 |
|---|---|---|
| Base (% de local) | 0.6909 | 0.6913 |
| Solo equipos (ataque y defensa por temporada) | 0.6853 | 0.6857 |
| Solo jugadores, sin nivel de equipo | 0.6856 | 0.6877 |
| **Equipos + jugadores + contexto (elegido)** | **0.6844** | **0.6851** |
| + ponches, BB y HR del abridor (tipo xFIP) | 0.6833 | 0.6862 |

Error del total ≈ 3.56–3.60 carreras.

**Qué dice:**
- Los jugadores mejoran **encima** del nivel de cada equipo. Solos, sin el equipo, empeoran.
- El abridor necesita mucho encogimiento: las carreras de un juego tienen mucho azar.
- Mejores abridores en el último ajuste: Sale, Skubal, Crochet, Skenes, Senga, Valdez, Snell, Wheeler.
- Ponches, bases por bolas y jonrones del abridor mejoran 2015–2019, pero empeoran 2021–2025, seguramente por los cambios de reglas de 2023 (reloj de lanzamiento, sin shift). No entran.
- Ventaja de mano y cansancio del bullpen: coeficientes chicos (+0.06 y −0.001 por lanzador).

## Fase 4: contra los momios de cierre (`ml/scripts/mlb_backtest.py`)

15,402 juegos de 2015–2021 con momios de cierre de SBR (`load_odds` empareja los códigos de SBR con los de Retrosheet; en las dobles carteleras se empareja por marcador).

- **Log loss del ganador:** modelo **0.6849** contra cierre sin margen **0.6715**. El mercado está bastante mejor.
- **Apuestas** de 1 unidad con momios de cierre:

| Regla | 2015–2018 | 2019–2021 |
|---|---|---|
| Moneyline, valor > 0% | 8,282 · −4.1% | 5,075 · −2.9% |
| Moneyline, valor > 3% | 6,843 · −4.7% | 4,180 · −2.0% |
| Run line, valor > 0% | 8,090 · −5.1% | 4,852 · −4.4% |
| Total, valor > 0% | 7,346 · −2.6% | 4,447 · −0.8% |
| Moneyline, confianza ≥ 60% | 3,002 · −3.5% | 2,021 · −1.7% |
| Total, confianza ≥ 60% | 2,590 · −3.0% | 1,983 · −3.1% |
| Run line, confianza ≥ 60% | 6,092 · −4.6% | 3,302 · −2.2% |

**Igual que en NFL y fútbol: no hay ventaja contra el cierre.** Se pierde más o menos el margen del casino. El total es lo más cercano a cero (−0.8% en 2019–2021, dentro del ruido). En la app, los picks del modelo se muestran con su confianza y se miden en vivo.
