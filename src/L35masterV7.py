# -*- coding: utf-8 -*-
# L35master.py (con export a Shapefile y KMZ)
# IN :
#   C:\Python2025_SAC\GITHUBL58\LinesDat.L35\20190817112135.(seg|odc|CSV)
# OUT:
#   C:\Python2025_SAC\GITHUBL58\out\L35\
#
# Flujo:
# 1) Lee posiciones desde CSV (si existe) o desde ODC.
# 2) lat/lon -> metros (UTM 18S, EPSG:32718), filtra outliers, distancia acumulada.
# 3) Remuestrea distancia a Nº de trazas del SEG y genera perfil (Distance & Depth).
# 4) Genera mapa Folium.
# 5) Exporta línea de navegación a Shapefile, KMZ, GeoJSON y JSON.
# [V7] Igual que V6 + etiquetas UTM (E/N) sin caja, en negro, más grandes y
#      alineadas con los márgenes izquierdo y derecho, debajo del eje X.

import os, sys, subprocess, zipfile, json, hashlib
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from obspy import read
from pyproj import Transformer
from datetime import datetime, timedelta

# ---- INTENTAR importar geopandas/shapely para SHP (opcionalmente) ----
try:
    import geopandas as gpd
    from shapely.geometry import LineString
    _GEO_OK = True
except Exception:
    _GEO_OK = False

# ---------------- Rutas ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                    # ...\GITHUBL58\src
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))                 # ...\GITHUBL58
ARCH_DIR = os.path.join(ROOT_DIR, "LinesDat.L35")                         # ...\GITHUBL58\LinesDat.L35
OUT_DIR  = os.path.join(ROOT_DIR, "out", "L35")                           # ...\GITHUBL58\out\L35
os.makedirs(OUT_DIR, exist_ok=True)

SEG_PATH = os.path.join(ARCH_DIR, "20190817112135.seg")
ODC_PATH = os.path.join(ARCH_DIR, "20190817112135.odc")
CSV_PATH = os.path.join(ARCH_DIR, "20190817112135.CSV")                  # opcional

# ---------------- Parámetros ----------------
VEL_MS  = 1500.0     # velocidad del sonido (m/s) -> para convertir TWT a profundidad
Z_MULT  = 10.0       # factor vertical para profundidades (1.0 = sin estirar)
CMAP    = "turbo"    # paleta estilo quicklook
DPI     = 300
FIGSIZE = (18, 5.5)

# Fallback para navegación sintética (San Antonio, Chile aprox.)
FALLBACK_CENTER_LAT = -33.6000
FALLBACK_CENTER_LON = -71.6200

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16
})

def _open_file(path: str):
    try:
        if os.name == "nt": os.startfile(path)
        elif sys.platform == "darwin": subprocess.Popen(["open", path])
        else: subprocess.Popen(["xdg-open", path])
    except Exception:
        pass

def _choose_step(total_m: float) -> float:
    for step in (200, 250, 500, 1000, 2000, 5000, 10000, 20000):
        if total_m / step <= 14:
            return float(step)
    raw = max(1.0, total_m / 14.0)
    return float(int(np.ceil(raw / 100.0) * 100))

# ---------- Lectura posiciones desde CSV (si existe) ----------
def parse_csv_positions(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, engine="python", encoding_errors="ignore")
    except Exception:
        return pd.DataFrame()

    low2orig = {c.lower(): c for c in df.columns}
    def pick(keys):
        for k in keys:
            if k in low2orig:
                return low2orig[k]
        for lk, orig in low2orig.items():
            if any(k in lk for k in keys):
                return orig
        return None

    lat_col  = pick(["lat", "latitude", "latitud"])
    lon_col  = pick(["lon", "long", "longitude", "longitud"])
    date_col = pick(["date", "fecha"])
    time_col = pick(["time", "hora"])

    if not lat_col or not lon_col:
        return pd.DataFrame()

    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
    })
    if date_col: out["date"] = df[date_col].astype(str)
    if time_col: out["time"] = df[time_col].astype(str)

    m = out["lat"].between(-90, 10) & out["lon"].between(-180, -50)
    out = out[m].dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out

# ---------- Lectura posiciones desde ODC ----------
def parse_odc_positions(odc_path: str) -> pd.DataFrame:
    rows = []
    if not os.path.exists(odc_path):
        return pd.DataFrame()
    with open(odc_path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if ",151," not in s: continue
            parts = [p.strip() for p in s.split(",")]
            try:
                i_date = next(i for i,p in enumerate(parts) if "/" in p)
                i_time = i_date + 1
                lat = float(parts[i_time + 1]); lon = float(parts[i_time + 2])
                rows.append({"date": parts[i_date], "time": parts[i_time], "lat": lat, "lon": lon})
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty: return df
    m = df["lat"].between(-90, 10) & df["lon"].between(-180, -50)
    return df[m].reset_index(drop=True)

# ---------- Sintetizar navegación si no hay CSV/ODC ----------
def synthesize_navigation(n_points: int) -> pd.DataFrame:
    if n_points < 2: n_points = 2
    dlat = 2e-5
    dlon = 2e-5 * np.cos(np.deg2rad(FALLBACK_CENTER_LAT))
    lat0 = FALLBACK_CENTER_LAT - (n_points//2)*dlat
    lon0 = FALLBACK_CENTER_LON - (n_points//2)*dlon
    lat = lat0 + dlat*np.arange(n_points)
    lon = lon0 + dlon*np.arange(n_points)

    t0 = datetime(2019, 8, 17, 11, 21, 35)
    ts = [t0 + timedelta(seconds=i) for i in range(n_points)]
    df = pd.DataFrame({
        "date": [dt.strftime("%Y/%m/%d") for dt in ts],
        "time": [dt.strftime("%H:%M:%S") for dt in ts],
        "lat": lat,
        "lon": lon
    })
    print(f"[Aviso] Navegación sintetizada (n={n_points}) centrada en San Antonio, CL.")
    return df

# ---------- Lat/Lon -> distancia en metros; remuestreo a nº de trazas ----------
def latlon_to_distance(df_pos: pd.DataFrame, n_traces: int):
    if df_pos.empty:
        return None, pd.DataFrame()
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)  # UTM 18S
    x, y = tr.transform(df_pos["lon"].values, df_pos["lat"].values)
    dx = np.diff(x); dy = np.diff(y)
    step = np.hypot(dx, dy)

    med = np.median(step) if len(step) else 0.0
    thr = max(5.0, 10.0 * med) if med > 0 else 5.0
    step_clipped = np.clip(step, 0.0, thr)
    dist = np.insert(np.cumsum(step_clipped), 0, 0.0)

    t_src = np.linspace(0, 1, len(dist))
    t_tgt = np.linspace(0, 1, n_traces)
    dist_traces = np.interp(t_tgt, t_src, dist)

    df_pos = df_pos.copy()
    df_pos["dist_m"] = dist
    return dist_traces, df_pos

# ---------- Perfil desde SEG con X en metros (+ UTM en extremos) ----------
def plot_profile_with_distance(seg_path: str, dist_m: np.ndarray, out_dir: str,
                               utm_start_end=None):
    """
    utm_start_end: tuple (E_ini, N_ini, E_fin, N_fin) en EPSG:32718.
    """
    st = read(seg_path, format="SEGY", unpack_trace_headers=True)
    data = np.array([tr.data for tr in st], dtype=float)
    n_traces, n_samples = data.shape
    dt = st[0].stats.delta
    depth_max_m = n_samples * dt * VEL_MS / 2.0

    p1, p99 = np.percentile(data, [1, 99])
    p99 = p99 if p99 > p1 else (p1 + 1.0)
    data = np.clip(data, p1, p99)
    data = (data - p1) / (p99 - p1)

    total_m = float(dist_m[-1] - dist_m[0])
    extent = [float(dist_m[0]), float(dist_m[-1]), depth_max_m * Z_MULT, 0.0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(data.T, aspect="auto", cmap=CMAP, extent=extent)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Perfil L35")

    step = _choose_step(total_m)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style="plain", axis="x")
    if total_m / step > 12:
        ax.tick_params(axis="x", labelrotation=45)

    # --- [V7] Etiquetas UTM DEBAJO del eje X, alineadas a márgenes ---
    if utm_start_end is not None:
        e1, n1, e2, n2 = utm_start_end
        # Colocar en coordenadas del eje (0 = borde izq., 1 = borde der.)
        x_left_axes, x_right_axes = 0.0, 1.0
        y_frac = -0.26  # más negativo = más abajo

        txtL = f"Inicio UTM18S\nE {int(round(e1))} m\nN {int(round(n1))} m"
        txtR = f"Fin UTM18S\nE {int(round(e2))} m\nN {int(round(n2))} m"

        ax.text(x_left_axes,  y_frac, txtL, transform=ax.transAxes,
                ha="left", va="top", color="black", fontsize=13)
        ax.text(x_right_axes, y_frac, txtR, transform=ax.transAxes,
                ha="right", va="top", color="black", fontsize=13)

    # margen inferior extra para no recortar las etiquetas
    fig.subplots_adjust(bottom=0.30)
    plt.tight_layout(rect=(0, 0.10, 1, 1))

    jpg = os.path.join(out_dir, "L35_profile_master.jpg")
    svg = os.path.join(out_dir, "L35_profile_master.svg")
    plt.savefig(jpg, dpi=DPI)
    plt.savefig(svg)
    plt.close()
    print(f"✅ Perfil guardado:\n  {jpg}\n  {svg}")
    _open_file(jpg)

    qa_csv = os.path.join(out_dir, "L35_trace_distance_map.csv")
    pd.DataFrame({"trace": np.arange(n_traces), "dist_m": dist_m}).to_csv(qa_csv, index=False)
    print(f"📝 QA guardado: {qa_csv}")

# ---------- Export vectorial: Shapefile + KMZ + GeoJSON + JSON ----------
def export_track_products(df_pos: pd.DataFrame, out_dir: str, name_prefix: str = "L35_track"):
    dfp = df_pos.dropna(subset=["lat", "lon"]).copy()
    if len(dfp) < 2:
        print("[Aviso] Pocos puntos para exportar línea.")
        return

    # --- KML + KMZ ---
    kml_path = os.path.join(out_dir, f"{name_prefix}.kml")
    kmz_path = os.path.join(out_dir, f"{name_prefix}.kmz")
    coords_txt = " ".join([f"{lon:.8f},{lat:.8f},0" for lat, lon in zip(dfp["lat"], dfp["lon"])])
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{name_prefix}</name>
  <Placemark>
    <name>{name_prefix}</name>
    <Style>
      <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>
    </Style>
    <LineString>
      <tessellate>1</tessellate>
      <coordinates>
        {coords_txt}
      </coordinates>
    </LineString>
  </Placemark>
</Document>
</kml>"""
    with open(kml_path, "w", encoding="utf-8") as f:
        f.write(kml)
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, arcname=os.path.basename(kml_path))
    print(f"🗺️  KML guardado: {kml_path}")
    print(f"🗺️  KMZ guardado: {kmz_path}")

    # --- Shapefile (si hay geopandas/shapely) ---
    if _GEO_OK:
        try:
            line = LineString([(lon, lat) for lat, lon in zip(dfp["lat"], dfp["lon"])])
            gdf = gpd.GeoDataFrame({"name": [name_prefix]}, geometry=[line], crs="EPSG:4326")
            shp_path = os.path.join(out_dir, f"{name_prefix}.shp")
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            print(f"🗂️  Shapefile guardado: {shp_path}  (se crean .shx, .dbf, .prj)")

            geojson_path = os.path.join(out_dir, f"{name_prefix}.geojson")
            gdf.to_file(geojson_path, driver="GeoJSON")
            print(f"🌐  GeoJSON guardado: {geojson_path}")
        except Exception as e:
            print(f"[Aviso] No se pudo escribir Shapefile/GeoJSON ({e}). ¿Instalaste geopandas/shapely/pyogrio?")
    else:
        print("[Aviso] Geopandas/Shapely no disponibles → omito Shapefile. Instala con:")
        print("        pip install geopandas shapely pyogrio")

    # GeoJSON mínimo sin dependencias (fallback)
    try:
        geojson_path = os.path.join(out_dir, f"{name_prefix}.geojson")
        if not os.path.exists(geojson_path):
            coords_ll = [[float(lon), float(lat)] for lat, lon in zip(dfp["lat"], dfp["lon"])]
            gj = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"name": name_prefix},
                    "geometry": {"type": "LineString", "coordinates": coords_ll}
                }]
            }
            with open(geojson_path, "w", encoding="utf-8") as f:
                json.dump(gj, f, ensure_ascii=False, indent=2)
            print(f"🌐  GeoJSON guardado (modo simple): {geojson_path}")
    except Exception as e:
        print(f"[Aviso] No se pudo crear GeoJSON simple ({e}).")

    # JSON con puntos
    try:
        json_path = os.path.join(out_dir, f"{name_prefix}.json")
        cols = [c for c in ["lat", "lon", "date", "time", "dist_m"] if c in dfp.columns]
        payload = {"name": name_prefix, "crs": "EPSG:4326", "points": dfp[cols].to_dict(orient="records")}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"📄  JSON guardado: {json_path}")
    except Exception as e:
        print(f"[Aviso] No se pudo crear JSON ({e}).")

# ---------- Exportar navegación en formato ODC ----------
def export_odc_from_positions(df_pos: pd.DataFrame, out_dir: str, arch_dir: str, base_name: str = "20190817112135_v7"):
    dfp = df_pos.dropna(subset=["lat", "lon"]).copy()
    if dfp.empty:
        print("[Aviso] No hay puntos para crear ODC.")
        return
    if "date" not in dfp.columns or "time" not in dfp.columns:
        t0 = datetime(2019, 8, 17, 11, 21, 35)
        ts = [t0 + timedelta(seconds=i) for i in range(len(dfp))]
        dfp["date"] = [dt.strftime("%Y/%m/%d") for dt in ts]
        dfp["time"] = [dt.strftime("%H:%M:%S") for dt in ts]
    lines = [f"0,151,{d},{t},{la:.6f},{lo:.6f}" for d,t,la,lo in zip(dfp["date"], dfp["time"], dfp["lat"], dfp["lon"])]
    out_path  = os.path.join(out_dir,  f"{base_name}.odc")
    arch_path = os.path.join(arch_dir, f"{base_name}.odc")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"🧭 ODC generado: {out_path}")
    try:
        with open(arch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"🧭 ODC copiado a archivos: {arch_path}")
    except Exception as e:
        print(f"[Aviso] No se pudo escribir ODC en ARCH_DIR ({e}).")

# ---------- Checksums y HTML índice ----------
def write_checksums(filepaths, out_dir: str):
    chk_path = os.path.join(out_dir, "checksums.sha256")
    with open(chk_path, "w", encoding="utf-8") as fout:
        for p in filepaths:
            try:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1<<20), b""):
                        h.update(chunk)
                fout.write(f"{h.hexdigest()}  {os.path.basename(p)}\n")
            except Exception:
                pass
    print(f"🔏 Checksums: {chk_path}")

def write_index_html(out_dir: str, entries: dict):
    idx = os.path.join(out_dir, "index.html")
    def li(path):
        fn = os.path.basename(path)
        return f'<li><a href="{fn}" target="_blank">{fn}</a></li>'
    html = ["<html><head><meta charset='utf-8'><title>L35 Outputs</title></head><body>",
            "<h1>Productos L35</h1>"]
    for title, files in entries.items():
        html.append(f"<h2>{title}</h2><ul>")
        for f in files:
            if f and os.path.exists(f): html.append(li(f))
        html.append("</ul>")
    html.append("</body></html>")
    with open(idx, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"📑 Índice HTML: {idx}")

# ---------- Mapa interactivo (Folium) ----------
def make_track_map(df_pos: pd.DataFrame, out_dir: str):
    if df_pos.empty or df_pos["lat"].isna().all() or df_pos["lon"].isna().all():
        print("[Aviso] Sin puntos válidos para el mapa.")
        return
    lat = df_pos["lat"].values
    lon = df_pos["lon"].values
    center = [float(np.nanmean(lat)), float(np.nanmean(lon))]
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    coords = list(zip(lat, lon))  # (lat, lon)
    folium.PolyLine(coords, color="blue", weight=3, opacity=0.8,
                    tooltip="Línea L35 (limpia)").add_to(m)
    folium.Marker(coords[0], popup="Inicio", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], popup="Fin", icon=folium.Icon(color="red")).add_to(m)
    out_html = os.path.join(out_dir, "L35_track_master.html")
    m.save(out_html)
    print(f"🌍 Mapa interactivo:\n  {out_html}")
    _open_file(out_html)

# ---------------- Orquestador ----------------
def main():
    print("→ Preparando entorno…")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Leer SEG (necesario para n_traces)
    if not os.path.exists(SEG_PATH):
        print(f"[ERROR] No se encuentra SEG: {SEG_PATH}")
        sys.exit(1)
    st_tmp = read(SEG_PATH, format="SEGY", unpack_trace_headers=True)
    n_traces = len(st_tmp)
    print(f"   SEG ok | Trazas: {n_traces}")

    # 2) Navegación: CSV > ODC > sintetizar
    print("→ Buscando posiciones…")
    df_csv = parse_csv_positions(CSV_PATH)
    if not df_csv.empty:
        df_pos = df_csv
        print(f"   Usando CSV ({len(df_pos)} filas válidas): {CSV_PATH}")
    else:
        df_odc = parse_odc_positions(ODC_PATH)
        if not df_odc.empty:
            df_pos = df_odc
            print(f"   Usando ODC ({len(df_pos)} filas válidas): {ODC_PATH}")
        else:
            print("   No hay CSV/ODC válidos → sintetizando navegación…")
            df_pos = synthesize_navigation(n_traces)

    # 3) Distancias y limpieza
    dist_m, df_pos_clean = latlon_to_distance(df_pos, n_traces)
    if dist_m is None:
        print("[ERROR] No se pudo calcular distancia.")
        sys.exit(1)
    total = dist_m[-1] - dist_m[0]
    print(f"   Distancia total (filtrada): {total:.1f} m | Trazas: {n_traces}")

    # 4) Guardar navegación limpia
    out_csv = os.path.join(OUT_DIR, "L35_rangos_odc.csv")
    df_pos_clean.to_csv(out_csv, index=False)
    print(f"📝 L35_rangos_odc.csv guardado: {out_csv}")

    # 5) Calcular UTM de inicio/fin para anotar en el perfil
    tr_utm = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    e1, n1 = tr_utm.transform(float(df_pos_clean["lon"].iloc[0]), float(df_pos_clean["lat"].iloc[0]))
    e2, n2 = tr_utm.transform(float(df_pos_clean["lon"].iloc[-1]), float(df_pos_clean["lat"].iloc[-1]))

    # 6) Perfil + Mapa + Vectoriales
    plot_profile_with_distance(SEG_PATH, dist_m, OUT_DIR, utm_start_end=(e1, n1, e2, n2))
    try:
        make_track_map(df_pos_clean, OUT_DIR)
    except Exception as e:
        print(f"[Aviso] No se pudo generar el mapa (continuo con exportaciones): {e}")
    export_track_products(df_pos_clean, OUT_DIR, name_prefix="L35_track")

    # 7) ODC derivado (y copia a ARCH_DIR)
    export_odc_from_positions(df_pos_clean, OUT_DIR, ARCH_DIR, base_name="20190817112135_v7")

    # 8) Índice HTML y checksums
    files = [
        os.path.join(OUT_DIR, "L35_profile_master.jpg"),
        os.path.join(OUT_DIR, "L35_profile_master.svg"),
        os.path.join(OUT_DIR, "L35_trace_distance_map.csv"),
        os.path.join(OUT_DIR, "L35_rangos_odc.csv"),
        os.path.join(OUT_DIR, "L35_track.kmz"),
        os.path.join(OUT_DIR, "L35_track.kml"),
        os.path.join(OUT_DIR, "L35_track.geojson"),
        os.path.join(OUT_DIR, "L35_track.json"),
        os.path.join(OUT_DIR, "20190817112135_v7.odc"),
        os.path.join(OUT_DIR, "L35_track.shp"),
    ]
    write_checksums([p for p in files if os.path.exists(p)], OUT_DIR)
    write_index_html(OUT_DIR, {
        "Perfil": [os.path.join(OUT_DIR, "L35_profile_master.jpg"),
                   os.path.join(OUT_DIR, "L35_profile_master.svg")],
        "Navegación / QA": [os.path.join(OUT_DIR, "L35_rangos_odc.csv"),
                            os.path.join(OUT_DIR, "L35_trace_distance_map.csv"),
                            os.path.join(OUT_DIR, "20190817112135_v7.odc")],
        "Vectoriales": [os.path.join(OUT_DIR, "L35_track.kmz"),
                        os.path.join(OUT_DIR, "L35_track.kml"),
                        os.path.join(OUT_DIR, "L35_track.geojson"),
                        os.path.join(OUT_DIR, "L35_track.shp")],
        "Mapa interactivo": [os.path.join(OUT_DIR, "L35_track_master.html")],
        "Otros": [os.path.join(OUT_DIR, "checksums.sha256")]
    })

if __name__ == "__main__":
    main()
