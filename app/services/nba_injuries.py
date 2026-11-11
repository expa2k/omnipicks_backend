"""
OmniPicks — NBA v2: lesiones desde ESPN (gratis).

Guarda un historial point-in-time: solo se inserta una fila cuando el estado de un
jugador cambia, y `is_current` marca el reporte vigente. Ese historial sirve para
calibrar qué tan seguido juega un "Day-To-Day" y para el backtest hacia adelante.
"""
import logging
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import NBAInjuryReport

logger = logging.getLogger(__name__)

ESPN_NBA_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Probabilidad inicial de jugar según el estado (se recalibra con el historial)
PLAY_PROBABILITY = {
    "Out For Season": 0.0,
    "Out": 0.0,
    "Doubtful": 0.15,
    "Questionable": 0.50,
    "Day-To-Day": 0.50,
    "Probable": 0.85,
    "Available": 1.0,       # solo en el reporte oficial de la NBA
}


def _play_probability(status: str) -> float:
    from app.services.nba_availability import play_probability
    return play_probability(status)


def norm_name(name: str) -> str:
    """Nombre normalizado para emparejar ESPN ↔ nba_api (sin acentos, sufijos ni puntuación)."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’`]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


async def fetch_espn_injuries() -> List[Dict]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(ESPN_NBA_INJURIES)
        r.raise_for_status()
        data = r.json()

    out = []
    for team in data.get("injuries", []):
        team_name = team.get("displayName", "")
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete", {})
            name = athlete.get("displayName", "")
            if not name:
                continue
            details = inj.get("details") or {}
            fantasy = (details.get("fantasyStatus") or {}).get("abbreviation")
            status = "Out For Season" if fantasy == "OFS" else inj.get("status", "Out")
            athlete_id = athlete.get("id")
            if not athlete_id:
                for link in athlete.get("links", []):
                    m = re.search(r"/id/(\d+)", link.get("href", ""))
                    if m:
                        athlete_id = m.group(1)
                        break
            out.append({
                "espn_athlete_id": str(athlete_id or norm_name(name))[:20],
                "player_name": name,
                "team_name": team_name,
                "status": status,
                "injury_type": details.get("type"),
                "comment": inj.get("shortComment"),
                "report_date": _parse_dt(inj.get("date")),
                "return_date": details.get("returnDate"),
            })
    return out


async def sync_injuries(db: AsyncSession) -> Dict[str, int]:
    """Lee ESPN y actualiza la tabla. Devuelve cuántos reportes cambiaron."""
    reports = await fetch_espn_injuries()
    current = (await db.execute(
        select(NBAInjuryReport).where(NBAInjuryReport.is_current.is_(True))
    )).scalars().all()
    current_by_key = {(c.player_name, c.team_name): c for c in current}

    seen, new, changed = set(), 0, 0
    for rep in reports:
        key = (rep["player_name"], rep["team_name"])
        seen.add(key)
        prev = current_by_key.get(key)
        if prev and prev.status == rep["status"]:
            continue
        if prev:
            prev.is_current = False
            prev.ended_at = datetime.now()          # misma hora local que captured_at (NOW() de MySQL)
            changed += 1
        else:
            new += 1
        comment = rep["comment"] or ""
        if rep.get("return_date"):
            comment = f"{comment} (regreso estimado: {rep['return_date']})".strip()
        db.add(NBAInjuryReport(
            espn_athlete_id=rep["espn_athlete_id"], player_name=rep["player_name"],
            team_name=rep["team_name"], status=rep["status"], injury_type=rep["injury_type"],
            comment=comment, report_date=rep["report_date"], is_current=True,
        ))

    # Jugadores que ya no aparecen en el reporte: están disponibles de nuevo
    cleared_ids = [c.id for key, c in current_by_key.items() if key not in seen]
    if cleared_ids:
        await db.execute(update(NBAInjuryReport).where(NBAInjuryReport.id.in_(cleared_ids))
                         .values(is_current=False, ended_at=datetime.now()))

    await db.commit()
    result = {"total": len(reports), "new": new, "changed": changed, "cleared": len(cleared_ids)}
    logger.info(f"Lesiones NBA: {result}")
    return result


async def current_injuries_by_team(db: AsyncSession) -> Dict[str, List[Dict]]:
    """Reporte vigente agrupado por nombre de equipo (como lo escribe ESPN)."""
    rows = (await db.execute(
        select(NBAInjuryReport).where(NBAInjuryReport.is_current.is_(True))
    )).scalars().all()
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r.team_name, []).append({
            "player_name": r.player_name,
            "name_key": norm_name(r.player_name),
            "status": r.status,
            "injury_type": r.injury_type,
            "comment": r.comment,
            "play_probability": _play_probability(r.status),
        })
    return out
