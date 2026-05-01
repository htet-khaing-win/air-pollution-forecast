"""Quick smoke test to verify Phase 1 imports and config."""
import sys
sys.path.insert(0, ".")

from config.settings import settings

print("=" * 50)
print("Config loaded successfully!")
print(f"  City: {settings.CITY_NAME}")
print(f"  Lat/Lon: {settings.CITY_LAT}, {settings.CITY_LON}")
print(f"  Raw data dir: {settings.RAW_DATA_DIR}")

key = settings.OWM_API_KEY
key_set = bool(key)
print(f"  OWM API key set: {key_set}")
if not key_set:
    print("  Add your OpenWeatherMap API key!")

print()

# Test imports
from src.ingestion.openaq import fetch_openaq_data
from src.ingestion.openweather import fetch_openweather_data
from src.ingestion.ingest import run_ingestion

print("All Phase 1 modules imported successfully!")
print("=" * 50)