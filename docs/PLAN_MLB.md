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
| 1 | Abridores: rating propio por pitcher (K%, BB%, HR, contacto) y entradas esperadas | |
| 2 | Bullpen y alineación contra la mano del abridor | |
| 3 | Parque, clima, localía y distribución de carreras | |
| 4 | Backtest de apuestas contra el cierre (2015–2021) | |
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
