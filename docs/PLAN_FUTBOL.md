# OmniPicks: plan del modelo de fútbol (v2)

**Hoy:** la fórmula `media de la liga × ataque × defensa rival` no tiene topes y usa muestras de 1–3 partidos. Salen marcadores irreales, como Feyenoord vs Como 1.07–15.38. Además, la base casi no tiene partidos de fútbol recientes: el último es del 21-oct, porque la ingesta de fútbol depende del pipeline viejo, que está apagado.

**Objetivo:** el pronóstico más acertado posible de 1X2, marcador, goles totales y ambos anotan, construido igual que la NBA y la NFL:
- datos propios con histórico;
- modelo validado con backtest;
- el mercado solo para compararse, no entra al modelo.

**Cómo se mide** (prueba limpia en las últimas temporadas):
- log loss y Brier del 1X2;
- log loss del marcador exacto;
- error de goles totales;
- todo comparado contra el modelo actual y contra los momios de cierre.

## Datos (gratis)

| Fuente | Qué trae | Ligas |
|---|---|---|
| **football-data.co.uk** (CSV) | Resultados desde los 90, tiros, tiros a puerta, córners, tarjetas y **momios de cierre** de varias casas (1X2, más/menos 2.5, hándicap asiático) | Premier, LaLiga, Serie A, Bundesliga, Ligue 1, Eredivisie, Portugal y más; Liga MX y Brasil (resultados y momios) |
| **ClubElo** (API CSV) | Elo diario de cada club europeo | Toda Europa: sirve para Champions y Europa League |
| **Understat** | xG por partido y por tiro desde 2014 | Las 5 grandes de Europa |
| **ESPN** (ya integrado) | Calendario, resultados en vivo y momios de los partidos por jugar | Todas las que ya usamos, incluida Liga MX |
| API-Sports (ya integrado, 100 peticiones al día) | Alineaciones y lesiones | Solo para lo que ESPN no tenga |

## Fases

| Fase | Qué | Validación |
|---|---|---|
| 0 | Descargar el histórico (football-data, ClubElo, Understat) y pasar el calendario y los resultados a ESPN (deja de gastar cuota de API-Sports) | Tabla de cobertura por liga |
| 1 | **Dixon-Coles:** goles de cada equipo con Poisson, ataque y defensa ajustados por rival, localía por liga, descuento por antigüedad y corrección de marcadores bajos (0-0, 1-0, 1-1) | 1X2 y marcador exacto contra el modelo actual y contra los momios de cierre |
| 2 | **xG:** ratings con goles esperados en lugar de goles reales (en las 5 grandes). El xG predice mejor que los goles | Mejora sobre la fase 1 |
| 3 | **Copas europeas:** Champions y Europa League con ClubElo, más un ajuste por la fuerza de cada liga | Partidos entre ligas |
| 4 | **Mercados** desde la matriz de marcadores (1X2, doble oportunidad, más/menos, ambos anotan, hándicap asiático), con calibración y backtest con momios reales | Unidades y ROI con momios de cierre |
| 5 | **Producción:** predicción congelada antes del partido, liquidación con el marcador de ESPN, historial y tarjeta con el porqué del pronóstico | Revisión en el navegador |
| 6 | Opcional: alineaciones y bajas, si el backtest muestra que ayudan | Partidos con bajas importantes |

## Expectativa honesta

- En el fútbol hay mucho azar: pocos goles, y un empate o un 1-0 cambian todo. Los mejores modelos públicos quedan **cerca** de los momios de cierre, no arriba.
- Lo realista es pasar de marcadores irreales a pronósticos sólidos y bien calibrados. Si hay valor, suele estar en ligas con menos atención, como la Liga MX, o en mercados secundarios.
- Como en la NFL, lo que no mejore en el backtest no entra.

## Fase 0: inventario (13-nov-2026, `app/services/soccer_history.py`)

**football-data.co.uk:** 71,323 partidos de 12 ligas, con caché en `ml/datasets/soccer/`. Las temporadas cerradas no cambian; la actual se baja de nuevo con `refresh_current`.

| Liga | Partidos | Temporadas | Tiros / a puerta | Cierre Pinnacle | Cierre promedio y apertura | xG |
|---|---|---|---|---|---|---|
| Premier League | 6,180 | 2010–26 | ✅ | desde 2012 | desde 2019 | solo 2026 |
| Championship | 9,012 | 2010–26 | ✅ | desde 2012 | desde 2019 | solo 2026 |
| La Liga | 6,199 | 2010–26 | ✅ | desde 2012 | desde 2019 | solo 2026 |
| Serie A | 6,190 | 2010–26 | ✅ | desde 2012 | desde 2019 | solo 2026 |
| Bundesliga | 4,977 | 2010–26 | ✅ | desde 2012 | desde 2019 | solo 2026 |
| Ligue 1 | 5,847 | 2010–26 | ✅ | desde 2012 | desde 2019 | solo 2026 |
| Eredivisie | 4,930 | 2010–26 | parcial | desde 2012 | desde 2019 | solo 2026 |
| Primeira Liga | 4,730 | 2010–26 | parcial | desde 2012 | desde 2019 | solo 2026 |
| **Liga MX** | 4,798 | 2012–26 | ❌ | desde 2012 | solo cierre 1X2 | ❌ |
| Brasileirão | 5,659 | 2012–26 | ❌ | desde 2012 | solo cierre 1X2 | ❌ |
| MLS | 6,326 | 2012–26 | ❌ | desde 2012 | solo cierre 1X2 | ❌ |
| Argentina | 6,475 | 2012–26 | ❌ | desde 2012 | solo cierre 1X2 | ❌ |

Todas están al día: los últimos partidos son del 6–9 de noviembre de 2026.

**Otras fuentes:**
- **ESPN:** responde por fecha (no por rango), con resultados y momios. Ejemplo: Champions del 1-oct-2024, 9 partidos. Sirve para el calendario de todas las ligas y para el **histórico de Champions y Europa League** (cruces entre ligas), bajándolo día por día.
- **ClubElo:** caído (502) el 13-nov. Se vuelve a intentar en la fase 3; si no regresa, se arma un Elo propio con los cruces de copas de ESPN.
- **Understat:** devuelve una página de error a las consultas automáticas, así que no se usa.

**Consecuencias para el diseño:**
- **Sin xG histórico.** Los ratings de la fase 2 se arman con **tiros y tiros a puerta** (desde 2010 en las 5 grandes y la Championship), que es la aproximación estándar. El xG de football-data se usa desde 2026 a medida que se junte.
- **Comparación contra el mercado:** el cierre de Pinnacle desde 2012 en todas las ligas.
- **Liga MX y las ligas de América:** solo resultados y momios 1X2. Ahí el modelo será Dixon-Coles con goles.

## Fase 1: Dixon-Coles (`app/services/soccer_dc.py`, `ml/scripts/soccer_dc_eval.py`)

**Modelo:**

    λ_local = exp(μ + localía + ataque[local] − defensa[visitante])
    λ_visitante = exp(μ + ataque[visitante] − defensa[local])

- Poisson con la corrección de Dixon-Coles para 0-0, 1-0, 0-1 y 1-1.
- Máxima verosimilitud con peso exp(−ξ·días) y ridge en ataque y defensa.
- Una liga a la vez, con los partidos de los últimos ~3 años.
- De la matriz de marcadores salen 1X2, goles esperados, más/menos 2.5 y ambos anotan.

**Backtest:** semana a semana, cada semana se ajusta solo con los partidos anteriores. Se compara contra:
- la base: frecuencias de 1X2 de la liga en las temporadas previas;
- **Pinnacle al cierre**, sin margen.

**Parámetros** (elegidos con 2019–2021): ξ = 0.002 por día (vida media ≈ 1 año), ridge = 2. Probé ξ 0.001/0.002/0.0035 × ridge 0.5/2/6; el rango completo de log loss fue de 1.0024 a 1.0065.

**Resultado** (log loss del 1X2, menor es mejor):

| Liga | 2019–21 base | 2019–21 modelo | 2019–21 Pinnacle | 2022–26 base | **2022–26 modelo** | 2022–26 Pinnacle |
|---|---|---|---|---|---|---|
| Premier League | 1.076 | 0.982 | 0.969 | 1.068 | **0.987** | 0.950 |
| La Liga | 1.081 | 1.000 | 0.982 | 1.063 | **0.977** | 0.956 |
| Serie A | 1.086 | 0.977 | 0.955 | 1.085 | **0.982** | 0.968 |
| Bundesliga | 1.074 | 1.001 | 0.979 | 1.072 | **0.994** | 0.968 |
| Ligue 1 | 1.084 | 1.016 | 0.994 | 1.072 | **0.998** | 0.976 |
| Liga MX | 1.074 | 1.046 | 1.033 | 1.069 | **1.017** | 0.987 |
| **Todas** | 1.079 | 1.002 | 0.984 | 1.071 | **0.992** | 0.967 |

En 2022–26, el Brier del modelo es 0.592 contra 0.575 de Pinnacle; el log loss del marcador exacto, 2.93; y el error de goles totales, 1.32 goles por partido.

**Qué dice:**
- Dixon-Coles queda **muy por encima de la base** (−0.08 de log loss) y a **0.025 de Pinnacle**. Eso es lo esperable para un modelo de goles bien hecho; los modelos públicos buenos suelen quedar a 0.01–0.03 del cierre.
- El modelo actual de la app (sin topes, muestras de 1–3 partidos) no se puede ni comparar, porque daba marcadores como 1.07–15.38.
- En la Liga MX el modelo está más lejos del mercado (0.030). Allí los torneos cortos y la liguilla hacen todo más ruidoso; es candidata a ajustes propios (por ejemplo, un descuento más rápido).
- **Siguiente (fase 2):** ratings con tiros y tiros a puerta en las 5 ligas europeas, que suelen predecir mejor que los goles.

## Fase 2: ratings con tiros a puerta (`soccer_dc.fit_rates`, `ml/scripts/soccer_shots_eval.py`)

**Goles por tiros:** con 2010–2018 (antes de los años evaluados), **goles ≈ 0.289 × tiros a puerta**; los tiros desviados no aportan (coeficiente ≈ 0). La correlación con los goles del partido es 0.55.

**Modelo:** los ratings de ataque y defensa se ajustan con una mezcla

    y = w · goles + (1 − w) · 0.289 · tiros a puerta

con cuasi-verosimilitud Poisson y gradiente analítico. El rho de los marcadores bajos sale del ajuste con goles. Los partidos sin datos de tiros usan solo goles.

**Resultado** (5 ligas europeas; log loss del 1X2 y, entre paréntesis, el error de goles totales):

| Periodo | w = 1 (fase 1) | **w = 0.7** | w = 0.5 | w = 0.3 | w = 0 (solo tiros) | Pinnacle |
|---|---|---|---|---|---|---|
| 2019–2021 | 0.9946 (1.31) | **0.9933 (1.30)** | 0.9943 | 0.9969 | 1.0040 | 0.9752 |
| 2022–2026 | 0.9870 (1.31) | **0.9863 (1.31)** | 0.9874 | 0.9898 | 0.9962 | 0.9630 |

**Qué dice:**
- **70% goles y 30% tiros a puerta** es lo mejor en los dos periodos. La mejora es chica (−0.001), pero consistente. Mejora en Premier, Bundesliga, Ligue 1 y LaLiga; en la Serie A 2022–26 empeora un poco.
- Solo tiros es peor que solo goles: los tiros a puerta solos no ven la calidad del remate.
- Con xG real se esperaría más mejora. El xG de football-data se puede sumar desde 2026, cuando haya suficientes partidos.
- **Decisión:** w = 0.7 en las 5 ligas europeas. La Liga MX sigue solo con goles, porque no tiene tiros.

## Copas europeas: histórico (`app/services/soccer_cups.py`)

**4,711 partidos** de Champions y Europa League de 2016 a 2026, bajados de ESPN día por día (martes a jueves) y guardados por temporada. Son la base de la fase 3: los cruces entre clubes de distintas ligas.

## Fase 3: Champions y Europa League (`app/services/soccer_joint.py`, `ml/scripts/soccer_cups_eval.py`)

**Nombres:** `soccer_cups.match_names` empareja ESPN con football-data. Normaliza (acentos, "FC", "AC"…), usa una lista de alias (Man City, Ath Madrid, Sp Lisbon, M'gladbach…) y solo acepta coincidencias muy seguras (≥ 0.88). Los clubes de ligas que no tenemos (Celtic, Salzburgo, Shakhtar, Galatasaray…) quedan como equipos propios dentro del grupo **"otras ligas"**.

**Modelo conjunto:** un solo ajuste con las 8 ligas europeas y las copas,

    λ = exp(μ_competición + localía_competición + ataque[eq] + A[su liga] − defensa[rival] − D[liga del rival])

- La fuerza por liga (A y D) se aprende de los cruces en las copas.
- Dentro de una liga, A − D se confunde con su μ, así que solo ordena las ligas entre sí.
- Objetivo: 70% goles + 30% tiros a puerta en las ligas; solo goles en copas.
- Ridge 2 en los equipos y 5 en las ligas; ξ = 0.002; 3 años de historia; rho = −0.05.

**Resultado** (2,718 partidos de copa de 2019–2026, semana a semana; log loss del 1X2):

| Partidos | Base | Solo ligas (todas las ligas iguales) | **Conjunto** |
|---|---|---|---|
| Todos (2,718) | 1.046 | 0.986 | **0.946** |
| Champions (1,144) | 1.043 | 0.938 | **0.907** |
| Europa League (1,574) | 1.049 | 1.020 | **0.974** |
| Los dos equipos de nuestras ligas (884) | 1.047 | 1.019 | **0.978** |
| Uno de nuestras ligas y otro de fuera (1,112) | 1.037 | 0.910 | **0.872** |
| Los dos de otras ligas (722) | 1.060 | 1.060 | **1.020** |
| 2019–2021 | 1.064 | 1.004 | **0.960** |
| 2022–2026 | 1.031 | 0.969 | **0.933** |

**Fuerza por liga hoy** (A + D, en escala logarítmica de goles): Premier +0.64 · Bundesliga +0.13 · LaLiga +0.12 · Serie A +0.10 · Ligue 1 −0.02 · Championship −0.12 · Portugal −0.18 · otras ligas −0.25 · Eredivisie −0.42.

**Qué dice:**
- **Tomar en cuenta la fuerza de cada liga mejora mucho las copas**: −0.04 de log loss contra tratar a todas las ligas iguales, y −0.10 contra la base. Mejora en las dos competencias y en los dos periodos.
- Donde más se nota es en los partidos entre un club de nuestras ligas y uno de fuera (0.872).
- Los partidos entre dos clubes de "otras ligas" (las fases previas) siguen difíciles: 1.02.
- No hay momios históricos de copas para comparar contra el mercado. Se medirá en vivo con los momios de ESPN.
- El valor de "otras ligas" mezcla ligas muy distintas (Turquía con Islas Feroe). Mejora posible: separar por país con ESPN (ligas de Portugal, Bélgica, Turquía, Escocia…), o usar ClubElo cuando vuelva.
- **Decisión:** las copas usan el modelo conjunto; las ligas siguen con el Dixon-Coles de las fases 1 y 2.
