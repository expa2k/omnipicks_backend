"""
OmniPicks — NBA: reporte oficial de lesiones de la liga (PDF, gratis).

La NBA publica el reporte cada 15 minutos en official.nba.com. Es más completo y suele
llegar antes que ESPN: trae el estado oficial (Out, Doubtful, Questionable, Probable,
Available) y el motivo (lesión, descanso "Injury Management", G League…).

Se guarda el último reporte (is_current) y el predictor lo usa como fuente principal para
los partidos que incluye; ESPN queda de respaldo.
"""
import io
import logging
import re
from datetime import date, datetime
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

INDEX_URL = "https://official.nba.com/nba-injury-report-{season}-season/"
_URL_RE = re.compile(r"https://ak-static\.cms\.nba\.com/referee/injury/Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf")
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
# Bordes de columna (en puntos del PDF, a partir de los encabezados)
_COLS = [("game_date", 0), ("game_time", 115), ("matchup", 195), ("team", 260), ("player", 420),
         ("status", 580), ("reason", 660)]
_last_url: Dict[str, Optional[str]] = {"url": None}


def _season(d: date) -> str:
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def latest_report_url(day: date) -> Optional[str]:
    """URL del reporte más reciente publicado (se lee la página índice de la liga)."""
    r = httpx.get(INDEX_URL.format(season=_season(day)), headers=_HEADERS, timeout=20.0, follow_redirects=True)
    r.raise_for_status()
    found = []
    for m in _URL_RE.finditer(r.text):
        d, hh, mm, ap = m.groups()
        h = int(hh) % 12 + (12 if ap == "PM" else 0)
        found.append((datetime.fromisoformat(d).replace(hour=h, minute=int(mm)), m.group(0)))
    return max(found)[1] if found else None


def _report_time(url: str) -> datetime:
    d, hh, mm, ap = _URL_RE.search(url).groups()
    return datetime.fromisoformat(d).replace(hour=int(hh) % 12 + (12 if ap == "PM" else 0), minute=int(mm))


def _flip_name(name: str) -> str:
    """'Udeh, Jr., Ernest' → 'Ernest Udeh Jr.' · 'Barnes, Scottie' → 'Scottie Barnes'."""
    parts = [p.strip() for p in name.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0]
    return " ".join([parts[-1]] + parts[:-1])


def parse_pdf(content: bytes) -> List[Dict]:
    import pdfplumber

    rows: List[Dict] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        carry = {"game_date": None, "matchup": None, "team": None}
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=True, x_tolerance=1.5)
            lines: Dict[int, Dict[str, List[str]]] = {}
            for w in words:
                if w["top"] < 120:                                  # título y encabezados
                    continue
                col = [c for c, x in _COLS if w["x0"] >= x - 2][-1]
                key = round(w["top"])
                line = next((k for k in lines if abs(k - key) <= 2), key)
                lines.setdefault(line, {}).setdefault(col, []).append(w["text"])
            tops = sorted(lines)
            player_tops = [t for t in tops if " ".join(lines[t].get("status", [])).strip() in STATUSES
                           or "NOT YET SUBMITTED" in " ".join(lines[t].get("player", [])).upper()]
            for t in tops:
                cells = {c: " ".join(v).strip() for c, v in lines[t].items()}
                for c in ("game_date", "matchup", "team"):
                    if cells.get(c):
                        carry[c] = cells[c]
                if t not in player_tops:
                    continue
                if "NOT YET SUBMITTED" in cells.get("player", "").upper():
                    continue
                # El motivo puede venir partido en las líneas de arriba y abajo
                near = [u for u in tops if u not in player_tops and abs(u - t) <= 12 and lines[u].get("reason")]
                reason = " ".join([" ".join(lines[u]["reason"]) for u in near if u < t] + [cells.get("reason", "")]
                                  + [" ".join(lines[u]["reason"]) for u in near if u > t]).strip()
                try:
                    gd = datetime.strptime(carry["game_date"], "%m/%d/%Y").date()
                except (TypeError, ValueError):
                    continue
                rows.append({"game_date": gd, "matchup": carry["matchup"], "team_name": carry["team"],
                             "player_name": _flip_name(cells.get("player", "")), "status": cells["status"],
                             "reason": re.sub(r"\s+", " ", reason).strip()[:200] or None})
    return rows


def fetch_latest(day: date) -> Optional[Dict]:
    url = latest_report_url(day)
    if not url:
        return None
    r = httpx.get(url, headers=_HEADERS, timeout=30.0)
    r.raise_for_status()
    return {"url": url, "report_time": _report_time(url), "rows": parse_pdf(r.content)}


def sync_official_sync(engine, day: date) -> Dict:
    """Baja el último reporte oficial y lo deja como vigente. Bloqueante."""
    from sqlalchemy import text
    from app.db_models import NBAOfficialInjury

    rep = fetch_latest(day)
    if rep is None or not rep["rows"]:
        return {"rows": 0, "new_report": False}
    with engine.begin() as conn:
        last = conn.execute(text("SELECT MAX(report_time) FROM nba_official_injuries")).scalar()
        if last is not None and last >= rep["report_time"]:
            return {"rows": 0, "new_report": False, "report_time": str(rep["report_time"])}
        before = {(r[0], r[1], r[2]) for r in conn.execute(text(
            "SELECT team_name, player_name, status FROM nba_official_injuries WHERE is_current = 1"))}
        conn.execute(text("UPDATE nba_official_injuries SET is_current = 0 WHERE is_current = 1"))
        conn.execute(NBAOfficialInjury.__table__.insert(),
                     [{**r, "report_time": rep["report_time"], "is_current": True} for r in rep["rows"]])
    after = {(r["team_name"], r["player_name"], r["status"]) for r in rep["rows"]}
    changed = len(before ^ after)
    logger.info(f"Reporte oficial NBA {rep['report_time']:%Y-%m-%d %H:%M}: {len(rep['rows'])} jugadores, {changed} cambios")
    return {"rows": len(rep["rows"]), "new_report": True, "changed": changed, "report_time": str(rep["report_time"])}
