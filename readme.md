# CHIRPVIEW V7

CHIRPVIEW V7 is a Python-based workflow for processing **high-resolution CHIRP seismic profiles** (example: line L35).  
It provides a reproducible, documented, and automated pipeline that reads seismic SEG data and navigation files, and exports multiple products ready for QA, mapping, and further analysis.

## Features
- Reads navigation from **CSV** (preferred) or **ODC** (fallback), or generates synthetic navigation if missing.
- Converts coordinates from **WGS84 → UTM 18S (EPSG:32718)**.
- Robust filtering of outliers and calculation of accumulated distance.
- Resampling of navigation to match SEG trace count.
- Depth calculation from `dt` and `VEL_MS` (sound velocity), with configurable `Z_MULT`.
- Profile plots (Distance vs Depth) with color scale (`turbo`) and annotations (UTM start/end).
- Interactive **Folium Map** with line and start/end markers.
- Export to multiple formats:
  - **Vector**: KML + KMZ (always), optional SHP/GeoJSON (via GeoPandas).
  - **Raster/Images**: JPG + SVG profiles.
  - **Tables**: QA CSV files (`L35_trace_distance_map.csv`, `L35_rangos_odc.csv`).
- Auto-generated **index.html** with quick links and **checksums.sha256** for reproducibility.

## Repository Structure
CHIRPVIEWV7/
├─ src/ # Python source code
│ └─ L35masterV7.py # main script
├─ run_L35masterV7.bat # Windows launcher
├─ LinesDat.L35/ # raw data (ignored by git: SEG, ODC, CSV)
├─ out/L35/ # generated outputs (ignored by git)
├─ .gitignore
└─ README.md

markdown
Copiar código

## Requirements
- **Python ≥ 3.10** (tested on 3.12)  
- Core libraries:
  - `obspy`, `numpy`, `pandas`, `matplotlib`, `pyproj`, `folium`
- Optional (for SHP/GeoJSON export):
  - `geopandas`, `shapely`, `pyogrio`

Install with:
```powershell
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
Execution
Option A – double-click the BAT file:

Copiar código
run_L35masterV7.bat
Option B – manual execution:

powershell
Copiar código
cd src
python L35masterV7.py
Outputs will be available in out/L35/.

Example Products
Profile: L35_profile_master.jpg / .svg

Map: L35_track_master.html

Vector: L35_track.kmz (+ SHP if libraries are installed)

QA CSVs: L35_trace_distance_map.csv, L35_rangos_odc.csv

License
This repository is provided for research and technical use.
See LICENSE for details (recommended: MIT License).