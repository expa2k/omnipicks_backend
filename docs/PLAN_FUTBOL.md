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
