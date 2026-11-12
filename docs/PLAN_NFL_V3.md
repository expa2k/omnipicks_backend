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
