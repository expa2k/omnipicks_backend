# OmniPicks: estado del proyecto

**Actualizado:** 10-nov-2026

Este documento resume todo lo hecho, lo que está en curso y lo que falta. Los planes detallados están en la misma carpeta:

| Documento | Contenido |
|---|---|
| [`PLAN.md`](PLAN.md) | Auditoría completa (~40 errores por gravedad), fuentes de datos gratis y plan general en 7 fases |
| [`PLAN_MODELO.md`](PLAN_MODELO.md) | Arquitectura del motor de pronóstico para los 4 deportes y los mejores enfoques por deporte |
| [`PLAN_NBA.md`](PLAN_NBA.md) | Plan detallado del modelo NBA v2 (jugadores, bajas, simulación, backtest) y decisiones tomadas |

---

## 1. Repositorios y ramas

| Repo | Ruta local | Rama de trabajo | Commits nuevos (solo locales, **sin push**) |
|---|---|---|---|
| Backend (FastAPI) | `~/PycharmProjects/omnipicks_backend` | `nba-v2` (sale de `nba-v1-espn`, que sale de `main`) | `4aaf08a` NBA v1 · `1a7e9cb` NBA v2 N0 · este documento |
| Frontend (Angular 18) | `~/Documents/GitHub/Angular_proyects/omnipicks` | `ui-redesign` (sale de `main`) | `a103989` rediseño · `a67e36b` paleta en negro |

`main` no se ha tocado en ninguno de los dos repos.

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

### 2.2 NBA v2, fase N0: histórico propio (rama `nba-v2`)
- **Tablas nuevas** en `app/db_models.py`: `nba_games`, `nba_player_games`, `nba_stints`, `nba_injury_reports` y `nba_player_ratings`.
- **`app/services/nba_history.py`:**
  - partidos y box scores por temporada;
  - **stints** (tramos con los mismos 10 jugadores en cancha), armados con las rotaciones y el play-by-play, con puntos y posesiones de cada lado. Validados contra marcadores reales: coinciden, con un partido que difiere por 1 punto.
- **`ml/scripts/nba_backfill.py`:** carga reanudable de 2022-23 a la fecha, con límite de velocidad y reintentos.
- **`app/services/nba_injuries.py`:** lesiones de ESPN con historial point-in-time (solo guarda cambios de estado) y probabilidad inicial de jugar según el estado.

**Datos cargados hasta ahora:**
| Tabla | Filas |
|---|---|
| `nba_games` | 5,409 partidos (2022-23 a 2026-27, temporada regular y playoffs) |
| `nba_player_games` | 115,869 filas jugador-partido |
| `nba_stints` | 73,307 tramos de **2,583 de 5,409 partidos**, sin fallos (la carga sigue corriendo) |
| `nba_injury_reports` | 93 reportes (primera captura) |

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

## 3. En curso ahora mismo
- **Carga de stints de NBA:** el proceso `ml/scripts/nba_backfill.py` corre en segundo plano (log en `logs/nba_backfill.log`). Va en 2,583 de 5,409 partidos, sin fallos. Si se detiene, se vuelve a correr el mismo comando y **continúa donde se quedó**.
- **Servidores apagados:** se detuvieron al cerrarse la sesión anterior. Para levantarlos:
  ```bash
  # Backend (sin --reload: si no, cada archivo guardado relanza el pipeline y gasta cuota)
  cd ~/PycharmProjects/omnipicks_backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
  # Frontend
  cd ~/Documents/GitHub/Angular_proyects/omnipicks && npx ng serve --port 4200
  ```
  Ojo: al arrancar, el backend corre el pipeline completo, y eso gasta requests de API-Sports y The Odds API.

---

## 4. Lo que falta

### 4.1 NBA v2 (prioridad actual), ~20% hecho
| Fase | Qué es | Tamaño | Estado |
|---|---|---|---|
| N0 | Tablas, histórico, stints y lesiones | — | ✅ hecho (stints cargando) |
| N0 | Actualización diaria del histórico y revisión de lesiones **cada 30 min** | chico | ❌ |
| **N1** | Rating por jugador: prior por box score + plus-minus ajustado (RAPM, ridge sobre stints) + actualización diaria tipo Kalman | **grande** | ❌ |
| **N2** | Bajas: probabilidad de jugar, redistribución de minutos y de uso, rendimiento real con y sin el jugador, química de quintetos | **grande** | ❌ |
| N3 | Simulador: sortea quién juega y luego simula posesiones y puntos (~10,000 corridas) | mediano | ❌ |
| N4 | Contexto (descanso, viaje, altitud, momento de temporada) + corrección con LightGBM + calibración isotónica | mediano | ❌ |
| N5 | Backtest walk-forward de 3 temporadas contra `nba_possessions_v1` | mediano | ❌ |
| N6 | Integración con el pipeline, el endpoint y el front (**solo el impacto de las bajas**) | mediano | ❌ |

**Decisiones ya tomadas:**
- El mercado **no entra al modelo**; solo se usa para comparar.
- Se toman en cuenta las bajas y el rendimiento con y sin cada jugador.
- El front mostrará solo el impacto de las bajas.
- Las lesiones se revisan cada 30 minutos.
- Histórico de 2022-23 a la fecha.
- Se puede instalar lo que haga falta.

**Pendiente técnico:** `lightgbm` está en `requirements.txt` pero no carga en Mac. Le falta `libomp` (`brew install libomp`), y solo se necesita en N4.

### 4.2 Frontend
| Pieza | Estado |
|---|---|
| Impacto de las bajas en cada partido | ❌ depende de N2 |
| Rendimiento del modelo e historial de resultados (hoy dicen "Pronto") | ❌ depende de la liquidación de picks |
| Logos reales de los equipos (hoy son abreviaturas) | opcional; hay que agregarlos al endpoint |
| URL de la API por entorno (hoy fija en `localhost:8000`) | ❌ |

### 4.3 Resto del sistema (detalle en `PLAN.md`)
| Prioridad | Tema | Qué hacer |
|---|---|---|
| 🔴 **urgente** | **Seguridad** | Regenerar las 3 API keys (Bzzoiro, API-Sports y The Odds API) y la contraseña de MySQL, que siguen en el historial de git (`521fc0f`), y limpiar el historial |
| 🟠 | Pipeline y cuota | Que no corra en cada arranque (flag y endpoint), quitar `btts` (−9 requests por corrida), no duplicar predicciones, apagar el eco de SQL |
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
- **Push:** ninguna rama se ha subido a GitHub. Para subirlas:
  ```bash
  cd ~/PycharmProjects/omnipicks_backend && git push -u origin nba-v1-espn nba-v2
  cd ~/Documents/GitHub/Angular_proyects/omnipicks && git push -u origin ui-redesign
  ```
