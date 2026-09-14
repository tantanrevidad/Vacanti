# Vacanti: The smart parking management system — Proof of Concept (POC)

An enterprise-grade Streamlit application demonstrating every layer of Vacanti end-to-end, featuring predictive availability forecasting, confidence-weighted license plate matching, real-time slot state tracking, interactive direct-click SQLite database telemetry inspection, and computer vision feasibility validation across **Megaworld Townships** (Uptown Bonifacio, Eastwood City, and McKinley Hill / Venice Grand Canal Mall).

---

## Key Highlights & Current State

- **Township Deck Layout & Sequential Ordering:** Realistic township architectural scheme organized sequentially by commercial archetype (**Mall** $\rightarrow$ **Office** $\rightarrow$ **Residential**) with labeled Drive Aisles (Lane A, Lane B, Lane C) and expanded capacity across 3 Megaworld townships (**Uptown Bonifacio**, **Eastwood City**, and **McKinley Hill / Venice Grand Canal Mall**).
- **Zero-Page-Reload Direct Stall Inspection:** Every parking stall block is an interactive solid-color card powered by isolated fragment execution (`@st.fragment`). Clicking directly on any block instantly opens the **SQLite Database Row Inspector modal (`@st.dialog`)** with zero page reload or scroll jitter, displaying real-time telemetry, tickets, OCR reads, and raw SQL queries.
- **Dynamic Real-Time PST Hero Clock (Tab 1):** Live client-side Philippine Standard Time clock counting up seconds, minutes, and hours (`HH:MM:SS AM/PM`) dynamically with zero server overhead and a pulsing live monitoring badge.
- **Streamlined Date & Time Controls (Tab 2):** Dedicated future simulation interface featuring clean, centered Date and Time selectors for rapid predictive scenario analysis across any future timestamp.
- **Real-World External Data Pipeline (`real_data_pipeline.py`):** Ingests live empirical Google Places foot-traffic curves (0–100 index), real-time Metro Manila weather telemetry from Open-Meteo API (accelerated via 10-minute in-memory caching), Megaworld mall sales/events calendars, and arterial road delay estimates.
- **Real-Time Multi-Channel Social & CMS Scraper (`social_event_scraper.py`):** Automated real-time web-scraping engine monitoring the four official Facebook handles (`@MegaworldUptownMall`, `@VeniceGrandCanal`, `@eastwoodcity`, and `@megaworldlifestylemalls`) alongside Megaworld's production Contentstack Headless CMS Delivery API (`cdn.contentstack.io`). Features strict township disambiguation (rejecting non-target provincial malls and non-event corporate announcements), regex word boundary protection (eliminating substring collisions such as `'fest'` in `'lifestyle'`), and Windows charmap-safe string sanitization.
- **Philippine National Holidays Calendar (`ph_holidays.py`):** Automated statutory calendar (2024–2028) modeling zone-differentiated holiday mobility curves (commercial mall surges vs. corporate office dips).
- **Forecast Intelligence & Methodology (Tab 2):** Integrated analytical intelligence providing zone behavioral classification (Mall vs. Office vs. Residential), 4-week telemetry dataset provenance (2,688 intervals at 15-minute resolution), and detailed model specifications.
- **Stabilized Simulation State Machine:** Real-time 4-state slot tracking (`Available (Free)`, `Occupied — Unpaid`, `Occupied — Pending Match`, `Occupied — Likely Vacating Soon`) driven by simulated ticketing logs, grace periods, and stabilized 5-minute simulation window seeds.
- **Interactive Vacating Feature Simulator (`vacating_simulator.py`):** 3-way split-screen interactive studio directly inside Tab 1 demonstrating the complete departure lifecycle: (1) Mall Kiosk POS fee calculation (₱50 flat first 3h + ₱20/hr) with digital e-receipt generation, (2) 4-stage visual driver journey stepper with live mini-stall color transitions and 1-click animated demo playback, and (3) real-time SQLite database row inspection and outbound JSON signage event stream.
- **Confidence-Weighted Fuzzy Matcher:** Real OCR-to-ticket matching algorithm with optical confusable-character scoring ($0 \leftrightarrow O$, $1 \leftrightarrow I$, $8 \leftrightarrow B$, $5 \leftrightarrow S$, $2 \leftrightarrow Z$, $6 \leftrightarrow G$) and margin enforcement ensuring **0% false positive ticket matches**.
- **ML Occupancy Forecasting Engine:** `scikit-learn` `HistGradientBoostingRegressor` trained on historical readings using causal feature engineering enriched with Google busyness, weather, events, and holiday signals.
- **Dual Computer Vision ALPR Validation:**
  1. **Philippine License Plates Dataset (Roboflow Universe / LTO):** 20 authentic Philippine vehicle frames (`lpr-mgcu6/philippine-license-plates-wmxlq`) spanning LTO 2014/2018/2020 Private series (`MAT-2357`, `CAX-3200`, `LAN-3138`, `MAN-4684`, `NDU-6211`, `CBC-2080`), Legacy Rizal/Matatag series (`LHA-482`, `LEN-918`, `LGT-635`), and commercial fleet plates evaluated via multi-pass CLAHE contrast enhancement and prominence-weighted vehicle scoring.
  2. **Academic ALPR Benchmark (OpenALPR):** 14 curated international benchmark photographs with hand-verified ground truth plates.
- **5-Phase Computer Vision Space Detection Engine:** YOLOv8n vehicle detector with adaptive low-light CLAHE contrast enhancement, true perspective polygon ROI calibration (`slots_config.json`), Intersection over Area (IoA) occupancy scoring with centroid containment, and 5-frame temporal sliding-window state debouncing ($\ge 60\%$ consensus).
- **Enterprise Dark & Light Mode Theme Engine:** Dynamic theme switcher with bulletproof contrast across all widgets, BaseWeb calendars, popover dropdowns, custom unindented HTML tables (`render_styled_match_table`), custom inline code badges, and adaptive Plotly charts.

---

## What's Real vs. Simulated

| Component | Status | Implementation Details |
|---|---|---|
| **Camera Feed / Slot Occupancy** | **Simulated** | Synthetic 28-day 15-minute time series generated with zone-specific mathematical curves + Gaussian noise + event multipliers. |
| **License Plate OCR (App Live Stream)** | **Simulated** | Synthetic reads corrupted via an optical confusion matrix ($0/O, 1/I, 8/B, 5/S, 2/Z, 6/G$) with randomized confidence scores. |
| **Ticketing POS Integration** | **Simulated** | SQLite relational transaction log (`ticketing_records` in `data/parking.db`) tracking entry, payment timestamp, and ticket status. |
| **Vacating Lifecycle Simulator** | **Real** | 4-stage lifecycle state engine in `vacating_simulator.py` computing Megaworld parking fees, issuing receipts, and dispatching event bus telemetry. |
| **Plate-to-Ticket Matching Algorithm** | **Real** | Confidence-weighted Levenshtein matching with confusable character penalty discounts in `matcher.py`. |
| **Slot State Machine Engine** | **Real** | State transition rules in `state_machine.py` modeling Free, Unpaid, Pending Match, and Vacating states. |
| **ML Predictive Forecaster** | **Real** | `HistGradientBoostingRegressor` in `predictor.py` trained with time-based holdout validation, MAE metrics, and permutation feature importance. |
| **Social Media & CMS Event Ingestion** | **Real** | Multi-channel scraper in `social_event_scraper.py` ingesting 4 official Facebook pages + Contentstack Headless CMS with strict township disambiguation. |
| **Computer Vision ALPR Pipeline** | **Real** | YOLOv8n prominence vehicle detection + Sobel-X vertical edge plate localization + multi-pass CLAHE OCR + real candidate matching in `cv_demo.py`. |
| **Computer Vision Space Detector** | **Real** | 5-phase ROI & IoA vehicle detection engine with CLAHE enhancement & temporal debouncing in `parking_detector.py`. |

---

## Quickstart & Setup

### 1. Prerequisites
- **Python 3.10+** (Python 3.10–3.14 supported)
- **Tesseract OCR engine** (optional, RapidOCR CPU ONNX engine included by default)

### 2. Installation
```bash
git clone https://github.com/tantanrevidad/parking-poc.git
cd parking-poc
pip install -r requirements.txt
```

### 3. Generate Database & Launch App
```bash
# Generate data/parking.db (automatically initialized on first app launch if missing)
python generate_data.py

# Ingest and benchmark Roboflow Philippine dataset
python build_ph_roboflow_dataset.py

# Launch the Streamlit dashboard
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Inspecting SQLite Database Records

The persistent relational database is stored locally in `data/parking.db`.

### Method A: In-Dashboard Direct Click Inspection
1. Navigate to **Tab 1: Occupancy Map**.
2. Click **directly on any colored parking stall block** (`M-001`, `O-001`, `R-001`, etc.).
3. An interactive modal dialog opens instantly (with zero page reload via `@st.fragment`) displaying:
   - **Active Live Telemetry** — Consolidated snapshot of state, ticket, and OCR reading.
   - **Active Occupancy State (`current_state`)** — Current status and state change timestamp.
   - **Ticketing & Settlement (`ticketing_records`)** — Assigned vehicle plate, entry time, payment settlement time.
   - **Optical Plate Read (`plate_reads`)** — Camera OCR detected text, confidence vector, ground truth plate.
   - **Infrastructure Metadata (`slots` & `zones`)** — Deck level, archetype, township site, zone capacity.
   - **Executed SQLite Statement (Raw SQL)** — Underlying parameterized SQL statement executed against `data/parking.db`.

---

## Computer Vision Feasibility Demo

The CV demo evaluates vehicle detection and plate recognition against **real-world images with hand-verified ground truth**:

```bash
# Build & benchmark authentic Philippine Roboflow Universe dataset (20 frames)
python build_ph_roboflow_dataset.py

# Download Academic OpenALPR benchmark dataset (14 images)
python fetch_real_dataset.py

# Run the CV detection & matching pipeline on all datasets
python cv_demo.py --dataset all
```

---

## Application Modules & Tabs

1. **Occupancy Map:** Live real-time operations deck displaying Uptown Bonifacio, Eastwood City, and McKinley Hill (Venice Grand Canal Mall) zones ordered sequentially by archetype (**Mall** $\rightarrow$ **Office** $\rightarrow$ **Residential**). Displays live capacity KPIs, Drive Aisles, a dynamic counting PST clock, solid-color clickable blocks (Available, Occupied, Pending Match, Vacating) that trigger in-place SQLite modal inspection with zero page reload, and the integrated **Interactive Vacating Feature Simulator** modeling mall POS checkout, 15-minute grace windows, and automated bay release.
2. **Availability Forecast:** Dedicated future simulation & predictive planning tool allowing users to pick any zone, calendar date, and arrival time (using clean centered Date & Time pickers) to receive an ML forecast, baseline comparison, conservative safety margin, historical trend curve, and an in-depth Forecast Intelligence & Methodology analysis breakdown. Features the **Live Social Media & Event Radar** providing real-time multi-channel event scraping across 4 official Facebook accounts and Megaworld Contentstack CMS, 1-click on-demand re-syncing, telemetry badges, township quick-filter tabs, and responsive event cards with direct social links and traffic surge uplift tags.
3. **Model Performance:** Diagnostic validation dashboard detailing AI Prediction Error %, Standard Baseline Guess Error %, Accuracy Advantage %, Permutation Feature Importance with plain-English signal labels, Actual vs. Forecasted time-series tracking, and an Operational Executive Summary for non-technical stakeholders.
4. **Plate Matching:** Interactive slot inspector testing the fuzzy matcher on noisy OCR plate reads, displaying character confidence bars, candidate rankings via theme-adaptive tables, and match margin confirmation.
5. **ALPR Feasibility (CV):** Dual-dataset visual gallery allowing users to toggle between the **Philippine License Plates (Roboflow Universe)** and the **Academic OpenALPR Benchmark**, inspecting YOLOv8 vehicle boxes, localized plate crops, OCR reads, and matcher resolutions.
6. **Space Detection (CV):** Enterprise computer vision parking space occupancy detection engine across surveillance feeds in `car_dataset/`, implementing a **5-Phase Occupancy Detection Algorithm**: (1) Dual Native ROI Calibration from `slots_config.json`, (2) YOLOv8n vehicle inference with Adaptive Low-Light CLAHE Boost, (3) Intersection over Area (IoA) spatial occupancy calculation with Centroid Containment, (4) temporal sliding-window state debouncing (5-frame history, $\ge 60\%$ consensus), and (5) standardized JSON output payload generation with real-time overlays, capacity KPIs, and telemetry.
7. **Revenue Intelligence:** Executive financial engineering and commercial optimization dashboard built on a **5-Tier Data Provenance Framework** (Tier A: Actual Megaworld Tariffs, Tier B: Deterministically Derived Database Records, Tier C: Colliers Philippine Retail Benchmarks, Tier D: International Academic Research, Tier E: Explicit Modeled Diagnostics). Features (1) **Revenue Operations Deck** with daily ₱ metrics, RPBH (Revenue Per Bay Hour), peak revenue windows, and Zone × Hour Intensity Heatmaps; (2) **Dwell → Spend Elasticity** correlating empirical dwell distributions with tenant retail spend elasticity (Path Intelligence 1.3 elasticity, ₱1,000–₱3,000 baseline); (3) **Repeat-Visitor Recognition Engine** analyzing visitor retention cohorts (New, Returning, Loyal) with +67% loyalty spend premiums, operating exclusively on salted SHA-256 cryptographic hashes for full Philippine Data Privacy Act (RA 10173) compliance; and (4) an auditable Methodology transparency drawer.

---

## Zone Archetypes

- **Office Zones:** Morning surge (08:00–09:30), slight midday dip (12:00–13:00), evening drain after 17:30. Minimal weekend volume.
- **Mall Zones:** Gradual morning volume, sustained afternoon build, peak evening traffic (18:00–21:30), +35% higher traffic on weekends.
- **Residential Zones:** High overnight occupancy (85–95%), workday drop (08:00–17:00), consistent profile across weekdays and weekends.

---

## Project Structure

```
parking-poc/
├── .streamlit/
│   └── config.toml               # Custom enterprise theme configuration
├── car_dataset/                  # Multi-angle CCTV & parking row surveillance frames
├── cv-demo/                      # Local ALPR benchmark outputs (gitignored)
├── data/
│   └── parking.db                # SQLite database with relational schemas & telemetry
├── debug_output/                 # Visual regression layers & JSON telemetry outputs
├── docs/
│   ├── PROJECT_BIBLE.md               # Hyper-detailed Master Project Bible (Tab-by-Tab)
│   ├── Revised_Prototype_Plan.md      # Prototype architectural roadmap
│   ├── SYSTEM_DOCUMENTATION.md        # Comprehensive technical specification
│   ├── TAB6_SPACE_DETECTION_GUIDE.md  # Space Detection engine guide
│   ├── Smart_Parking_Implementation_Plan.pdf
│   └── Smart_Parking_POC_Technical_Implementation_Plan.pdf
├── scripts/
│   ├── calibrate_roi.py          # Interactive GUI parking slot ROI calibrator
│   └── debug_overlay.py          # 3-layer visual regression & diagnostic framework
├── app.py                        # Streamlit Enterprise Dashboard (7 Tabs)
├── build_ph_roboflow_dataset.py  # Authentic Philippine Roboflow dataset ingest & benchmark
├── calibrate.html                # Interactive browser-based polygon annotation tool
├── cv_demo.py                    # Computer Vision & ALPR evaluation pipeline
├── fetch_real_dataset.py         # OpenALPR benchmark dataset downloader
├── generate_data.py              # Synthetic database & calibrated vehicle pool generator
├── loyalty_engine.py             # Repeat-visitor cohort recognition & loyalty spend engine (RA 10173 compliant)
├── matcher.py                    # Confidence-weighted fuzzy matching algorithm
├── parking_detector.py           # 5-phase space occupancy & vehicle detection engine
├── ph_holidays.py                # Philippine National Holidays calendar & mobility modifiers
├── predictor.py                  # ML forecasting engine (HistGradientBoosting)
├── real_data_pipeline.py         # Real-world Google foot-traffic, weather, and events connector
├── requirements.txt              # Project dependencies
├── revenue_config.py             # Megaworld township rate tariffs, benchmarks, & 5 provenance tiers
├── revenue_engine.py             # RPBH, revenue operations, heatmaps, & dwell-spend elasticity
├── simulate.py                   # Real-time state simulation helper
├── slots_config.json             # Calibrated slot ROI polygon configurations
├── social_event_scraper.py       # Real-time multi-channel social media & CMS event scraping engine
├── state_machine.py              # Parking slot lifecycle state machine
├── vacating_simulator.py         # 4-stage vacating feature simulation engine
└── yolov8n.pt                    # Pre-trained YOLOv8 vehicle detector model
```