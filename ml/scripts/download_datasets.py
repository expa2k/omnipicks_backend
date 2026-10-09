"""
OmniPicks — ML: Download Historical Datasets
Downloads historical sports data from public sources (FiveThirtyEight)
for the last 10-15 years to train our ML models.
"""
import os
import httpx
import logging
import asyncio
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Directorio base para guardar los datasets
BASE_DIR = Path(__file__).resolve().parent.parent / "datasets"

# URLs de los datasets públicos de FiveThirtyEight (historial completo con ELO y probabilidades)
DATASETS = {
    "nba": "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv",
    "nfl": "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv",
    "mlb": "https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv",
    "soccer": "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
}

async def download_file(client: httpx.AsyncClient, sport: str, url: str):
    """Descarga un dataset y lo guarda localmente."""
    file_path = BASE_DIR / f"{sport}_history.csv"
    
    # Crear carpeta si no existe
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"⏳ Descargando dataset de {sport.upper()} desde {url}...")
    
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"✅ Dataset de {sport.upper()} guardado en: {file_path}")
    except Exception as e:
        logger.error(f"❌ Error al descargar el dataset de {sport.upper()}: {e}")

async def main():
    logger.info("🚀 Iniciando descarga masiva de datos históricos (10-15 años)...")
    async with httpx.AsyncClient() as client:
        tasks = [download_file(client, sport, url) for sport, url in DATASETS.items()]
        await asyncio.gather(*tasks)
    logger.info("🏆 Todas las descargas completadas con éxito.")

if __name__ == "__main__":
    asyncio.run(main())
