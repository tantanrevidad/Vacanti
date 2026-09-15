"""
driver_app.py
=============
Vacanti Driver Portal — Consumer-facing Smart Parking web-app.
Features the Spatial Monolith luxury theme with:
  1. Township Selection Screen with actual photography
  2. Pop-up Notice of General Information & Verified Megaworld Rates (2-column layout)
  3. Live 3D Isometric Parking Deck with Google Maps-style Level Switcher
  4. Dynamic Availability Forecasting & Tariff Calculator
  5. Full Pipeline Grounding: Backed by parking.db, predictor.py, revenue_engine.py & real_data_pipeline.py

Run with:
    python -m streamlit run driver_app.py
"""

import copy
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import generate_data as gd
import ph_holidays
import predictor
import real_data_pipeline
import revenue_config
import revenue_engine
import simulate

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vacanti Driver Portal",
    page_icon="P",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Clean CSS to center and frame the mobile web-app ──────────────────────────
st.markdown("""
<style>
    /* Dark basalt background everywhere to match #0B0C10 */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], body {
        background-color: #0B0C10 !important;
        color: #f8fafc !important;
    }
    /* Hide Streamlit default header, footer, decoration */
    #MainMenu, footer, header, [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
    }
    /* Remove padding around the container and frame for mobile layout */
    .block-container {
        max-width: 460px !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }
    iframe {
        border: none !important;
        border-radius: 0px !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


def get_pst_now():
    utc_now = datetime.now(timezone.utc)
    pst = timezone(timedelta(hours=8))
    return utc_now.astimezone(pst).replace(tzinfo=None)


def get_synchronized_events_registry(db_path: str = "data/parking.db"):
    """
    Synchronizes and retrieves real-time commercial promotional events, sales, and
    festivals across official Facebook handles and Megaworld Contentstack Headless CMS,
    merging with the curated operational baseline.
    """
    events_registry = []
    seen = set()

    # 1. Real-time scraped events from Facebook & Contentstack Headless CMS
    try:
        import social_event_scraper
        social_event_scraper.ensure_scraped_events_table(db_path)
        telemetry = social_event_scraper.sync_all_events(force_refresh=False, db_path=db_path)
        scraped_list = telemetry.get("events", [])
        for ev in scraped_list:
            ts = ev.get("township", "")
            if "Uptown" in ts:
                skey = "uptown"
            elif "Venice" in ts or "McKinley" in ts:
                skey = "venice"
            elif "Eastwood" in ts:
                skey = "eastwood"
            else:
                skey = "all"

            raw_title = ev.get("title", "Commercial Event").split("\n")[0].strip()
            title = social_event_scraper.sanitize_text(raw_title)
            key = (skey, title[:35].lower(), ev.get("start_date"))
            if key not in seen:
                seen.add(key)
                events_registry.append({
                    "siteKey": skey,
                    "title": title[:80],
                    "start": ev.get("start_date", ""),
                    "end": ev.get("end_date", ""),
                    "factor": float(ev.get("traffic_impact_factor", 1.25)),
                    "desc": social_event_scraper.sanitize_text(ev.get("description", ""))[:160],
                    "source": ev.get("source_platform", "Official Portal"),
                    "url": ev.get("source_url", "")
                })
    except Exception:
        pass

    # 2. Curated Megaworld Events Baseline Registry
    for ev in real_data_pipeline.MEGAWORLD_EVENTS_REGISTRY:
        skey = "venice" if "McKinley" in ev["site"] else ("uptown" if "Uptown" in ev["site"] else "eastwood")
        title = ev["title"]
        key = (skey, title[:35].lower(), ev["start_date"])
        if key not in seen:
            seen.add(key)
            events_registry.append({
                "siteKey": skey,
                "title": title,
                "start": ev["start_date"],
                "end": ev["end_date"],
                "factor": float(ev["traffic_impact_factor"]),
                "desc": ev.get("description", ""),
                "source": "Megaworld Official Registry",
                "url": ""
            })

    # Sort descending by traffic impact factor so highest surge takes precedence
    events_registry.sort(key=lambda x: x["factor"], reverse=True)
    return events_registry


@st.cache_data(show_spinner=False)
def load_backend_core_payload():
    """
    Loads, trains, and caches all project backend datasets:
    1. parking.db: sites, zones, slots, plate_reads, current_state, events, holidays.
    2. predictor.py: trained HistGradientBoostingRegressor and baseline_heuristic on 25,056 rows.
    3. revenue_config.py: verified Tier A Megaworld tariffs.
    4. real_data_pipeline.py: Google Popular Times curves and Megaworld promotional calendar.
    5. ph_holidays.py: official statutory holidays calendar.
    """
    db_path = Path("data/parking.db")
    if not db_path.exists():
        gd.main()

    conn = sqlite3.connect(str(db_path))
    sites_df = pd.read_sql("SELECT * FROM sites", conn)
    zones_df = pd.read_sql("SELECT * FROM zones", conn)
    slots_df = pd.read_sql("SELECT * FROM slots", conn)
    current_state_df = pd.read_sql("SELECT * FROM current_state", conn)
    plate_reads_df = pd.read_sql("SELECT * FROM plate_reads", conn)
    events_df = pd.read_sql("SELECT * FROM events", conn)
    holidays_df = pd.read_sql("SELECT * FROM holidays", conn)
    conn.close()

    history_df = predictor.load_history()
    model, metrics, importance_df, holdout = predictor.train_model(history_df)
    baseline_lookup = predictor.baseline_heuristic(history_df)

    mall_labels = {1: "Mall Grand Wing", 2: "Mall Main Plaza", 3: "Venice Grand Canal Mall"}

    baseline_dict = {}
    for _, row in baseline_lookup.iterrows():
        zid = int(row["zone_id"])
        dow = int(row["day_of_week"])
        hr = int(row["hour"])
        rate = round(float(row["baseline_rate"]), 4)
        if zid not in baseline_dict:
            baseline_dict[zid] = {}
        if dow not in baseline_dict[zid]:
            baseline_dict[zid][dow] = {}
        baseline_dict[zid][dow][hr] = rate

    ml_predictions = {}
    rows_to_predict = []
    keys_list = []

    for zid in zones_df["zone_id"].unique():
        z_row = zones_df[zones_df["zone_id"] == zid].iloc[0]
        s_id = int(z_row["site_id"])
        mall_lbl = mall_labels[s_id]

        for dow in range(7):
            is_wknd = int(dow >= 5)
            for hr in range(24):
                if dow == 4:
                    curve = real_data_pipeline.GOOGLE_POPULAR_TIMES_DATA[mall_lbl].get("friday", real_data_pipeline.GOOGLE_POPULAR_TIMES_DATA[mall_lbl]["weekday"])
                elif dow >= 5:
                    curve = real_data_pipeline.GOOGLE_POPULAR_TIMES_DATA[mall_lbl].get("weekend", real_data_pipeline.GOOGLE_POPULAR_TIMES_DATA[mall_lbl]["weekday"])
                else:
                    curve = real_data_pipeline.GOOGLE_POPULAR_TIMES_DATA[mall_lbl]["weekday"]
                busyness = int(curve[hr])

                same_hour_hist = history_df[
                    (history_df["zone_id"] == zid) & (history_df["ts"].dt.hour == hr)
                ]["occupancy_rate"]
                rolling_avg = float(same_hour_hist.mean()) if len(same_hour_hist) else 0.3

                rows_to_predict.append({
                    "hour": hr,
                    "day_of_week": dow,
                    "is_weekend": is_wknd,
                    "is_holiday": 0,
                    "is_event": 0,
                    "google_busyness": busyness,
                    "rolling_avg_same_hour": rolling_avg,
                    "zone_id": zid
                })
                keys_list.append((int(zid), int(dow), int(hr)))

    pred_df = pd.DataFrame(rows_to_predict)
    preds = np.clip(model.predict(pred_df[predictor.FEATURE_COLS]), 0.0, 1.0)

    for i, (zid_k, dow_k, hr_k) in enumerate(keys_list):
        t_est = round(float(preds[i]), 4)
        b_est = baseline_dict.get(zid_k, {}).get(dow_k, {}).get(hr_k, t_est)
        adj_est = round(t_est + 0.05 * (1.0 - t_est), 4)

        if zid_k not in ml_predictions:
            ml_predictions[zid_k] = {}
        if dow_k not in ml_predictions[zid_k]:
            ml_predictions[zid_k][dow_k] = {}
        ml_predictions[zid_k][dow_k][hr_k] = {
            "baseline": b_est,
            "trained": t_est,
            "adjusted": adj_est
        }

    real_plates = []
    for _, r in plate_reads_df.iterrows():
        try:
            confs = json.loads(r["char_confidences"])
            mean_conf = round(float(np.mean(confs)) * 100, 1)
        except Exception:
            mean_conf = 98.4
        real_plates.append({
            "plate": r["true_plate"],
            "raw_ocr": r["raw_ocr_text"],
            "confidence": mean_conf
        })

    slots_merged = slots_df.merge(current_state_df, on="slot_id", how="left")
    slots_by_zone = {}
    for zid in zones_df["zone_id"].unique():
        z_slots = slots_merged[slots_merged["zone_id"] == zid].head(24)
        slots_list = []
        for _, s in z_slots.iterrows():
            slots_list.append({
                "slot_id": int(s["slot_id"]),
                "slot_code": s["slot_code"],
                "status": "occupied" if s["status"] in ["occupied_unpaid", "occupied_pending_match", "occupied_paid"] else "free"
            })
        slots_by_zone[int(zid)] = slots_list

    townships_meta = {
        "venice": {
            "siteId": 3,
            "siteName": "McKinley Hill",
            "name": "Venice Grand Canal Mall",
            "location": "McKinley Hill, Taguig",
            "sub": "McKinley Hill, Taguig City",
            "levels": ["4F", "B1", "B2"],
            "defaultLevel": "4F",
            "zoneIdMap": {"4F": 7, "B1": 8, "B2": 9},
            "levelMeta": {
                "4F": {"desc": "Upper Deck Open Bays", "zoneType": "mall", "zoneId": 7, "fullZoneName": "Venice Grand Canal Mall — Upper Deck 4F (McKinley Hill)"},
                "B1": {"desc": "Mall Retail & Office Bays", "zoneType": "office", "zoneId": 8, "fullZoneName": "Commerce & Industry Plaza — Level B1 (McKinley Hill)"},
                "B2": {"desc": "Lower Basement Bays", "zoneType": "residential", "zoneId": 9, "fullZoneName": "Viceroy & Morgan Deck — Level B2 (McKinley Hill)"},
            },
            "tariffs": revenue_config.PARKING_RATES["McKinley Hill"]["mall"],
            "bulletsCol1": [
                f"• <b>₱{revenue_config.PARKING_RATES['McKinley Hill']['mall']['first_3hr_flat']:.2f}</b> First 3 hrs (Base)",
                f"• <b>₱{revenue_config.PARKING_RATES['McKinley Hill']['mall']['succeeding_hr']:.2f}</b> Each succeeding hr",
                f"• <b>₱{revenue_config.PARKING_RATES['McKinley Hill']['mall']['overnight_surcharge']:.2f}</b> Overnight surcharge",
                "• <b>Cutoff:</b> 12MN in / 12NN out",
                "• <b>Validation:</b> Verified at exit barrier"
            ],
            "bulletsCol2": [
                f"• <b>{revenue_config.PARKING_RATES['McKinley Hill']['mall']['grace_period_mins']}-Min Free</b> Drop-off grace",
                "• <b>2.10m Max</b> height clearance",
                "• <b>Level 4F</b> Open-Air deck open",
                "• <b>Cashless:</b> GCash, Maya, RFID",
                "• <b>Live Weather:</b> 28°C Partly Cloudy"
            ]
        },
        "uptown": {
            "siteId": 1,
            "siteName": "Uptown Bonifacio",
            "name": "Uptown Mall",
            "location": "Uptown Bonifacio, BGC",
            "sub": "Uptown Bonifacio, BGC",
            "levels": ["B1", "B2", "B3"],
            "defaultLevel": "B1",
            "zoneIdMap": {"B1": 1, "B2": 2, "B3": 3},
            "levelMeta": {
                "B1": {"desc": "Mall Grand Wing", "zoneType": "mall", "zoneId": 1, "fullZoneName": "Mall Grand Wing — Ground & Level 1 (Uptown Bonifacio)"},
                "B2": {"desc": "Corporate Tower Alpha", "zoneType": "office", "zoneId": 2, "fullZoneName": "Office Tower Alpha — Level B2 (Uptown Bonifacio)"},
                "B3": {"desc": "Long-Stay Tenant Bays", "zoneType": "residential", "zoneId": 3, "fullZoneName": "Residential Deck — Level B3 (Uptown Bonifacio)"},
            },
            "tariffs": revenue_config.PARKING_RATES["Uptown Bonifacio"]["mall"],
            "bulletsCol1": [
                f"• <b>₱{revenue_config.PARKING_RATES['Uptown Bonifacio']['mall']['first_3hr_flat']:.2f}</b> First 3 hrs (Base)",
                f"• <b>₱{revenue_config.PARKING_RATES['Uptown Bonifacio']['mall']['4th_to_7th_hr']:.2f}/hr</b> for 4th to 7th hr",
                f"• <b>₱{revenue_config.PARKING_RATES['Uptown Bonifacio']['mall']['7th_hr_plus_am']:.2f}/hr</b> (AM entry 7h+)",
                f"• <b>₱{revenue_config.PARKING_RATES['Uptown Bonifacio']['mall']['7th_hr_plus_pm']:.2f}/hr</b> (PM entry 7h+)",
                f"• <b>₱{revenue_config.PARKING_RATES['Uptown Bonifacio']['mall']['overnight_surcharge']:.2f}</b> Overnight charge"
            ],
            "bulletsCol2": [
                f"• <b>{revenue_config.PARKING_RATES['Uptown Bonifacio']['mall']['grace_period_mins']}-Min</b> Drop-off grace period",
                "• <b>2.05m</b> Basement height limit",
                "• <b>Cinema wing</b> elevator in B1",
                "• <b>Direct access:</b> The Island BGC",
                "• <b>Live Weather:</b> 29°C Clear Sky"
            ]
        },
        "eastwood": {
            "siteId": 2,
            "siteName": "Eastwood City",
            "name": "Eastwood Mall",
            "location": "Eastwood City, Quezon City",
            "sub": "Eastwood City, Quezon City",
            "levels": ["B1", "B2", "B3"],
            "defaultLevel": "B1",
            "zoneIdMap": {"B1": 4, "B2": 5, "B3": 6},
            "levelMeta": {
                "B1": {"desc": "Mall Main Plaza", "zoneType": "mall", "zoneId": 4, "fullZoneName": "Mall Main Plaza — Citywalk & Plaza Level B1 (Eastwood City)"},
                "B2": {"desc": "Office Annex Cyberpark", "zoneType": "office", "zoneId": 5, "fullZoneName": "Office Annex Deck — Level B2 (Eastwood City)"},
                "B3": {"desc": "Cinema & Lower Deck", "zoneType": "residential", "zoneId": 6, "fullZoneName": "Residential Tower — Level B3 (Eastwood City)"},
            },
            "tariffs": revenue_config.PARKING_RATES["Eastwood City"]["mall"],
            "bulletsCol1": [
                f"• <b>₱{revenue_config.PARKING_RATES['Eastwood City']['mall']['first_3hr_flat']:.2f} Flat</b> First 3 hrs",
                f"• <b>₱{revenue_config.PARKING_RATES['Eastwood City']['mall']['succeeding_hr_weekday']:.2f}</b> Succeeding (Weekday)",
                f"• <b>₱{revenue_config.PARKING_RATES['Eastwood City']['mall']['weekend_holiday_flat']:.2f} Flat</b> All Day (Weekends)",
                f"• <b>Statutory Holidays:</b> ₱{revenue_config.PARKING_RATES['Eastwood City']['mall']['weekend_holiday_flat']:.2f} Flat",
                f"• <b>₱{revenue_config.PARKING_RATES['Eastwood City']['mall']['overnight_surcharge']:.2f}</b> Overnight fee"
            ],
            "bulletsCol2": [
                f"• <b>{revenue_config.PARKING_RATES['Eastwood City']['mall']['grace_period_mins']}-Min Free</b> Drop-off grace",
                "• <b>2.10m Max</b> height limit",
                "• <b>Direct access:</b> Citywalk plaza",
                "• <b>Automated pay</b> kiosk in B1",
                "• <b>Live Weather:</b> 28°C Fair"
            ]
        }
    }

    holidays_map = {}
    for d, (hname, _) in ph_holidays.MOVABLE_PH_HOLIDAYS.items():
        holidays_map[d] = hname
    for (m, d), (hname, _) in ph_holidays.FIXED_PH_HOLIDAYS.items():
        holidays_map[f"{m:02d}-{d:02d}"] = hname

    events_registry = get_synchronized_events_registry(str(db_path))

    return {
        "townships": townships_meta,
        "ml_predictions": ml_predictions,
        "baseline_lookup": baseline_dict,
        "real_plates": real_plates,
        "slots_by_zone": slots_by_zone,
        "holidays": holidays_map,
        "events": events_registry,
        "popular_times": real_data_pipeline.GOOGLE_POPULAR_TIMES_DATA,
        "model_metrics": metrics
    }


def get_live_payload():
    """Combines cached core ML and database payload with live Open-Meteo weather and scraped events."""
    core = load_backend_core_payload()
    payload = copy.deepcopy(core)
    payload["events"] = get_synchronized_events_registry()

    coords_map = {
        "venice": (14.5350, 121.0509),
        "uptown": (14.5562, 121.0543),
        "eastwood": (14.6094, 121.0805),
    }
    weather = {}
    for k, (lat, lon) in coords_map.items():
        try:
            w = real_data_pipeline.fetch_open_meteo_weather(lat, lon)
            temp = round(float(w.get("temperature_c", 28.5)), 1)
            cond = w.get("condition", "Partly Cloudy")
            weather[k] = {
                "temp": temp,
                "condition": cond,
                "rain_mm": float(w.get("rainfall_mm", 0.0)),
                "is_raining": bool(w.get("is_raining", False))
            }
            if k in payload["townships"]:
                payload["townships"][k]["bulletsCol2"][4] = f"• <b>Live Weather:</b> {temp}°C {cond}"
        except Exception:
            weather[k] = {"temp": 28.5, "condition": "Partly Cloudy", "rain_mm": 0.0, "is_raining": False}

    payload["weather"] = weather
    return payload


def main():
    html_file = Path(__file__).parent / "driver_portal.html"
    if not html_file.exists():
        st.error("driver_portal.html file not found.")
        return

    html_content = html_file.read_text(encoding="utf-8")

    # Fetch live project backend payload
    try:
        live_payload = get_live_payload()
        data_json = json.dumps(live_payload, ensure_ascii=False)
        backend_script = f"<script>window.BACKEND_DATA = {data_json};</script>\n"
        
        # Inject window.BACKEND_DATA immediately before the main JavaScript System Engine
        marker = "<!-- JAVASCRIPT SYSTEM ENGINE (Backed by parking.db, predictor.py, revenue_engine.py & real_data_pipeline.py) -->"
        if marker in html_content:
            html_content = html_content.replace(marker, backend_script + marker)
        else:
            # Fallback insertion before closing body
            html_content = html_content.replace("</body>", f"{backend_script}</body>")
    except Exception as e:
        st.warning(f"Backend payload notice: {e}")

    # Inject actual mall photographs as base64 data URLs
    try:
        import assets_base64
        html_content = html_content.replace("assets/thumb_venice.jpg", assets_base64.MALL_IMAGES["venice"])
        html_content = html_content.replace("assets/thumb_uptown.jpg", assets_base64.MALL_IMAGES["uptown"])
        html_content = html_content.replace("assets/thumb_eastwood.jpg", assets_base64.MALL_IMAGES["eastwood"])
        if "bg_lines" in assets_base64.MALL_IMAGES:
            html_content = html_content.replace("assets/bg_lines.jpg", assets_base64.MALL_IMAGES["bg_lines"])
    except Exception:
        pass

    # Render full interactive web-app component inside Streamlit with authentic smartphone height
    components.html(html_content, height=920, scrolling=True)


if __name__ == "__main__":
    main()
