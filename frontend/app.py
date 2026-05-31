"""
CatTracker – Streamlit Dashboard v2
Ajouts : Geofencing, Alertes WebSocket, LSTM+Attention
"""
import os, json, time, threading
import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import websocket   # websocket-client

BACKEND  = os.getenv("BACKEND_URL", "http://localhost:8000")
WS_URL   = BACKEND.replace("http://", "ws://").replace("https://", "wss://")
LAT_HOME = float(os.getenv("LAT_HOME", "48.8566"))
LON_HOME = float(os.getenv("LON_HOME", "2.3522"))

st.set_page_config(
    page_title="🐱 CatTracker v2",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.alert-exit  { background:#ff4b4b22; border-left:4px solid #ff4b4b; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
.alert-enter { background:#00d26722; border-left:4px solid #00d267; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
.stButton>button { background:linear-gradient(90deg,#e94560,#0f3460); color:white; border:none; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Alertes en session ────────────────────────────────────────────────────────
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False

# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_cats():
    try:
        r = requests.get(f"{BACKEND}/cats/", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def get_positions(chat_id, limit=1000):
    try:
        r = requests.get(f"{BACKEND}/positions/{chat_id}?limit={limit}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def get_home_range(chat_id):
    try:
        r = requests.get(f"{BACKEND}/positions/{chat_id}/home-range", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def get_prediction(chat_id):
    try:
        r = requests.get(f"{BACKEND}/predict/{chat_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def upload_csv(chat_id, file_bytes, filename):
    try:
        r = requests.post(
            f"{BACKEND}/upload/{chat_id}",
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_zones(chat_id):
    try:
        r = requests.get(f"{BACKEND}/zones/{chat_id}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def create_circle_zone(chat_id, name, lat, lon, radius, color):
    try:
        r = requests.post(
            f"{BACKEND}/zones/{chat_id}/circle",
            json={"name": name, "center_lat": lat, "center_lon": lon,
                  "radius_m": radius, "color": color},
            timeout=5,
        )
        if r.status_code == 201:
            return r.json()
    except Exception:
        pass
    return None

def delete_zone(chat_id, zone_id):
    try:
        requests.delete(f"{BACKEND}/zones/{chat_id}/{zone_id}", timeout=5)
        return True
    except Exception:
        return False

def simulate_position(chat_id, lat, lon):
    try:
        r = requests.post(
            f"{BACKEND}/alerts/simulate/{chat_id}",
            json={"latitude": lat, "longitude": lon},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        return {"error": str(e)}

# ── WebSocket background thread ───────────────────────────────────────────────

def start_ws_listener(chat_id: int):
    """Lance un thread WebSocket qui collecte les alertes dans session_state."""
    def on_message(ws_app, message):
        try:
            data = json.loads(message)
            if data.get("type") == "connected":
                st.session_state.ws_connected = True
            elif "event" in data:
                st.session_state.alerts.insert(0, data)
                if len(st.session_state.alerts) > 50:
                    st.session_state.alerts = st.session_state.alerts[:50]
        except Exception:
            pass

    def on_error(ws_app, error):
        st.session_state.ws_connected = False

    def on_close(ws_app, *args):
        st.session_state.ws_connected = False

    url = f"{WS_URL}/ws/{chat_id}"
    ws_app = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    t = threading.Thread(target=ws_app.run_forever, daemon=True)
    t.start()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🐾 CatTracker v2")

    cats = get_cats()
    if not cats:
        st.error("Backend inaccessible.")
        st.stop()

    cat_options = {c["nom"]: c for c in cats}
    selected_name = st.selectbox("Chat", list(cat_options.keys()))
    cat    = cat_options[selected_name]
    chat_id = cat["id"]

    st.markdown(f"""
    **Race :** {cat.get('race','—')}  
    **Maison :** {cat['lat_home']:.4f}, {cat['lon_home']:.4f}
    """)

    # Connexion WebSocket
    if st.button("🔌 Connecter alertes temps réel"):
        start_ws_listener(chat_id)
        st.success("WebSocket lancé !")

    ws_status = "🟢 Connecté" if st.session_state.ws_connected else "🔴 Déconnecté"
    st.caption(f"WebSocket : {ws_status}")

    st.divider()
    st.subheader("📤 Charger un CSV")
    uploaded = st.file_uploader("Fichier GPS (.csv)", type=["csv"])
    if uploaded and st.button("Importer & Entraîner"):
        with st.spinner("Insertion + entraînement LSTM+Attention…"):
            result = upload_csv(chat_id, uploaded.getvalue(), uploaded.name)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"✅ {result['inserted']} pts insérés, réentraînement en cours…")
        st.cache_data.clear()

    st.divider()
    nb_points      = st.slider("Points affichés", 100, 5000, 1000, 100)
    show_heatmap   = st.toggle("Heatmap", True)
    show_homerange = st.toggle("Domaine vital", True)
    show_pred      = st.toggle("Prédiction LSTM", True)
    show_zones     = st.toggle("Zones geofencing", True)

# ── Onglets principaux ────────────────────────────────────────────────────────

tab_map, tab_geo, tab_charts, tab_alerts = st.tabs([
    "🗺️ Carte & Trajectoire",
    "🛡️ Geofencing",
    "📊 Graphiques",
    "🔔 Alertes temps réel",
])

positions_raw = get_positions(chat_id, nb_points)
if not positions_raw:
    st.warning("Aucune position. Importez un CSV via le panneau latéral.")
    st.stop()

df = pd.DataFrame(positions_raw)
df["ts"] = pd.to_datetime(df["ts"])
df.sort_values("ts", inplace=True)
df.reset_index(drop=True, inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# Onglet 1 : Carte
# ══════════════════════════════════════════════════════════════════════════════

with tab_map:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📍 Points", len(df))
    avg_s = df["vitesse_ms"].dropna().mean()
    c2.metric("⚡ Vitesse moy.", f"{avg_s:.2f} m/s" if not np.isnan(avg_s) else "—")
    mx_d = df["distance_home_m"].dropna().max()
    c3.metric("🏠 Distance max", f"{mx_d:.0f} m" if not np.isnan(mx_d) else "—")
    c4.metric("🕐 Début", df["ts"].min().strftime("%d/%m %H:%M"))
    c5.metric("🕐 Fin",   df["ts"].max().strftime("%d/%m %H:%M"))

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                   tiles="CartoDB dark_matter")

    coords = list(zip(df["latitude"], df["longitude"]))
    folium.PolyLine(coords, color="#e94560", weight=2, opacity=0.7,
                    tooltip="Trajectoire").add_to(m)
    folium.CircleMarker(coords[0],  radius=7, color="#2ecc71", fill=True,
                        tooltip="Départ").add_to(m)
    folium.CircleMarker(coords[-1], radius=7, color="#e94560", fill=True,
                        tooltip="Arrivée").add_to(m)
    folium.Marker([cat["lat_home"], cat["lon_home"]],
                  icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                  tooltip="🏠 Maison").add_to(m)

    if show_heatmap:
        HeatMap([[r["latitude"], r["longitude"]] for _, r in df.iterrows()],
                radius=12, blur=10, min_opacity=0.3).add_to(m)

    if show_homerange:
        hr = get_home_range(chat_id)
        if hr:
            folium.GeoJson(
                hr["polygon_geojson"],
                style_function=lambda _: {"fillColor":"#f39c12","color":"#f39c12",
                                          "weight":2,"fillOpacity":0.15},
                tooltip=f"Domaine vital : {hr['area_km2']} km²",
            ).add_to(m)

    if show_pred:
        pred = get_prediction(chat_id)
        if pred:
            folium.Marker(
                [pred["predicted_latitude"], pred["predicted_longitude"]],
                icon=folium.Icon(color="purple", icon="question", prefix="fa"),
                tooltip=f"🔮 LSTM+Attention\n({pred['predicted_latitude']:.5f}, {pred['predicted_longitude']:.5f})",
            ).add_to(m)

    if show_zones:
        for zone in get_zones(chat_id):
            if zone["zone_type"] == "circle":
                folium.Circle(
                    location=[zone["center_lat"], zone["center_lon"]],
                    radius=zone["radius_m"],
                    color=zone["color"],
                    fill=True,
                    fill_opacity=0.15,
                    tooltip=f"🛡️ {zone['name']} ({zone['radius_m']}m)",
                ).add_to(m)

    st_folium(m, height=500, use_container_width=True)

    if show_homerange and (hr := get_home_range(chat_id)):
        st.info(f"📐 Domaine vital : **{hr['area_km2']} km²** | {hr['n_points']} points")
    if show_pred and (pred := get_prediction(chat_id)):
        st.info(f"🔮 Prédiction LSTM+Attention : **{pred['predicted_latitude']:.5f}**, **{pred['predicted_longitude']:.5f}** | modèle: {pred.get('model_version','—')}")

# ══════════════════════════════════════════════════════════════════════════════
# Onglet 2 : Geofencing
# ══════════════════════════════════════════════════════════════════════════════

with tab_geo:
    st.subheader("🛡️ Gestion des zones de geofencing")

    col_form, col_zones = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### ➕ Créer une zone circulaire")
        with st.form("create_zone"):
            z_name   = st.text_input("Nom de la zone", "Zone maison")
            z_lat    = st.number_input("Latitude centre", value=cat["lat_home"], format="%.6f")
            z_lon    = st.number_input("Longitude centre", value=cat["lon_home"], format="%.6f")
            z_radius = st.slider("Rayon (mètres)", 50, 2000, 200, 25)
            z_color  = st.color_picker("Couleur", "#e94560")
            submitted = st.form_submit_button("Créer la zone")
            if submitted:
                result = create_circle_zone(chat_id, z_name, z_lat, z_lon, z_radius, z_color)
                if result:
                    st.success(f"✅ Zone « {z_name} » créée (rayon={z_radius}m)")
                    st.rerun()
                else:
                    st.error("Erreur lors de la création de la zone")

        st.divider()
        st.markdown("#### 🧪 Tester une position (simulation)")
        st.caption("Simule une position pour tester si les alertes se déclenchent.")
        with st.form("simulate"):
            sim_lat = st.number_input("Latitude", value=cat["lat_home"] + 0.003, format="%.6f")
            sim_lon = st.number_input("Longitude", value=cat["lon_home"], format="%.6f")
            sim_btn = st.form_submit_button("🚀 Simuler cette position")
            if sim_btn:
                result = simulate_position(chat_id, sim_lat, sim_lon)
                if result and "alerts_fired" in result:
                    if result["alerts_fired"] > 0:
                        st.warning(f"🚨 {result['alerts_fired']} alerte(s) déclenchée(s) !")
                        for a in result.get("alerts", []):
                            alert_data = json.loads(a) if isinstance(a, str) else a
                            st.error(alert_data.get("message", str(a)))
                    else:
                        st.success("✅ Position dans toutes les zones – aucune alerte.")
                else:
                    st.error(f"Erreur : {result}")

    with col_zones:
        st.markdown("#### 📋 Zones actives")
        zones = get_zones(chat_id)
        if not zones:
            st.info("Aucune zone définie. Créez-en une à gauche.")
        else:
            for zone in zones:
                with st.container():
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        icon = "⭕" if zone["zone_type"] == "circle" else "🔷"
                        st.markdown(f"""
                        **{icon} {zone['name']}**  
                        Type : `{zone['zone_type']}` | Rayon : {zone.get('radius_m','—')} m  
                        Centre : {zone.get('center_lat',''):.4f}, {zone.get('center_lon',''):.4f}  
                        ID : `{zone['zone_id']}`
                        """)
                    with col_b:
                        if st.button("🗑️", key=f"del_{zone['zone_id']}"):
                            delete_zone(chat_id, zone["zone_id"])
                            st.rerun()
                    st.divider()

        # Mini-carte des zones
        if zones:
            st.markdown("#### 🗺️ Aperçu des zones")
            m2 = folium.Map(location=[cat["lat_home"], cat["lon_home"]],
                            zoom_start=15, tiles="CartoDB positron")
            folium.Marker([cat["lat_home"], cat["lon_home"]],
                          icon=folium.Icon(color="blue", icon="home", prefix="fa")).add_to(m2)
            for zone in zones:
                if zone["zone_type"] == "circle":
                    folium.Circle(
                        location=[zone["center_lat"], zone["center_lon"]],
                        radius=zone["radius_m"], color=zone["color"],
                        fill=True, fill_opacity=0.2,
                        tooltip=zone["name"],
                    ).add_to(m2)
            st_folium(m2, height=300, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Onglet 3 : Graphiques
# ══════════════════════════════════════════════════════════════════════════════

with tab_charts:
    st.subheader("📊 Analyses temporelles")

    col1, col2 = st.columns(2)

    with col1:
        df_s = df.dropna(subset=["vitesse_ms"])
        if not df_s.empty:
            fig = px.line(df_s, x="ts", y="vitesse_ms",
                          title="Vitesse au fil du temps",
                          color_discrete_sequence=["#e94560"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white", height=280)
            st.plotly_chart(fig, use_container_width=True)

        df["hour"] = df["ts"].dt.hour
        hourly = df.groupby("hour")["vitesse_ms"].mean().reset_index()
        fig2 = px.bar(hourly, x="hour", y="vitesse_ms",
                      title="Activité par heure (vitesse moy.)",
                      color_discrete_sequence=["#e94560"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=280)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        df_d = df.dropna(subset=["distance_home_m"])
        if not df_d.empty:
            fig3 = px.area(df_d, x="ts", y="distance_home_m",
                           title="Distance à la maison",
                           color_discrete_sequence=["#0f3460"])
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", height=280)
            st.plotly_chart(fig3, use_container_width=True)

        # Distance max par jour
        df["date"] = df["ts"].dt.date
        daily_max = df.groupby("date")["distance_home_m"].max().reset_index()
        fig4 = px.bar(daily_max, x="date", y="distance_home_m",
                      title="Distance max par jour",
                      color_discrete_sequence=["#f39c12"])
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=280)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Onglet 4 : Alertes temps réel
# ══════════════════════════════════════════════════════════════════════════════

with tab_alerts:
    st.subheader("🔔 Alertes temps réel (WebSocket)")

    col_info, col_clear = st.columns([3, 1])
    with col_info:
        if st.session_state.ws_connected:
            st.success("🟢 WebSocket connecté — les alertes arrivent en temps réel")
        else:
            st.warning("🔴 WebSocket non connecté — cliquez sur « Connecter alertes temps réel » dans le panneau latéral")
    with col_clear:
        if st.button("🧹 Vider les alertes"):
            st.session_state.alerts = []
            st.rerun()

    st.caption("Les alertes se déclenchent quand le chat entre ou quitte une zone geofencing.")
    st.divider()

    alerts = st.session_state.alerts
    if not alerts:
        st.info("Aucune alerte pour l'instant. Créez une zone et simulez une position pour tester.")
    else:
        st.markdown(f"**{len(alerts)} alerte(s) reçue(s) :**")
        for alert in alerts:
            event = alert.get("event", "")
            msg   = alert.get("message", str(alert))
            ts    = alert.get("timestamp", "")
            zone  = alert.get("zone_name", "")
            lat   = alert.get("latitude", "")
            lon   = alert.get("longitude", "")

            css_class = "alert-exit" if event == "exit" else "alert-enter"
            icon      = "🚨" if event == "exit" else "✅"
            st.markdown(f"""
            <div class="{css_class}">
            {icon} <strong>{msg}</strong><br>
            <small>Zone : {zone} | Position : {lat:.5f}, {lon:.5f} | {ts}</small>
            </div>
            """, unsafe_allow_html=True)

    # Auto-refresh toutes les 5 secondes si WebSocket connecté
    if st.session_state.ws_connected:
        time.sleep(5)
        st.rerun()
