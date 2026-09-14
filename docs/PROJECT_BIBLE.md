# Vacanti: The smart parking management system — Comprehensive Project Bible
**Enterprise Proof of Concept (POC) Technical & Empirical Reference**

> **Target Deployments:** Megaworld Townships (*Uptown Bonifacio*, *Eastwood City*, *McKinley Hill / Venice Grand Canal Mall*)  
> **Repository:** `tantanrevidad/parking-poc`  
> **Version:** 2.5.0 Enterprise Release  
> **Classification:** Production-Grade Reference Blueprint & Operating Specification  
> **Last Verified:** September 2026  

---

## Table of Contents
1. [Executive Summary & System Philosophy](#1-executive-summary--system-philosophy)
2. [Global Technology Stack & Infrastructure](#2-global-technology-stack--infrastructure)
3. [Global Data Provenance Framework (5-Tier Taxonomy)](#3-global-data-provenance-framework-5-tier-taxonomy)
4. [Tab 1: Occupancy Map & Real-Time Township Deck Operations](#4-tab-1-occupancy-map--real-time-township-deck-operations)
5. [Tab 2: Availability Forecast & Predictive Scenario Modeling](#5-tab-2-availability-forecast--predictive-scenario-modeling)
6. [Tab 3: Model Performance & Diagnostic Validation](#6-tab-3-model-performance--diagnostic-validation)
7. [Tab 4: Confidence-Weighted License Plate Matching](#7-tab-4-confidence-weighted-license-plate-matching)
8. [Tab 5: ALPR Feasibility Validation (Computer Vision)](#8-tab-5-alpr-feasibility-validation-computer-vision)
9. [Tab 6: Multi-Angle Parking Space Occupancy Detection (CV)](#9-tab-6-multi-angle-parking-space-occupancy-detection-cv)
10. [Tab 7: Revenue Intelligence & Commercial Optimization](#10-tab-7-revenue-intelligence--commercial-optimization)
11. [Cross-Module Variable Sensitivity & Failure Mode Matrix](#11-cross-module-variable-sensitivity--failure-mode-matrix)
12. [Hardware Topography, Edge Deployment & Production Scaling](#12-hardware-topography-edge-deployment--production-scaling)

---

## 1. Executive Summary & System Philosophy

### 1.1 Commercial & Operational Context
In multi-tenant commercial and mixed-use township developments such as **Megaworld Uptown Bonifacio**, **Eastwood City**, and **McKinley Hill**, parking facilities represent the critical gateway of customer and tenant experience. Drivers seeking parking during peak retail and business hours spend an average of **12 to 20 minutes cruising** for vacant bays, exacerbating arterial gridlock (e.g., along C-5, Upper McKinley Road, and 36th Avenue), elevating carbon emissions, and eroding tenant retail conversion.

Legacy parking solutions rely on static ultrasonic point sensors or inductive barrier loops. While they yield instantaneous slot vacancy counts, they exhibit three fatal structural limitations:
1. **Zero Anticipatory Intelligence:** They cannot predict availability 30 to 120 minutes into the future to guide incoming transit demand.
2. **Zero Turnover Foresight:** They cannot differentiate between an occupied bay where the occupant has just parked versus a stall whose ticket was settled at an automated pay station and is vacating imminently.
3. **Brittle Optical Correlation:** They cannot reconcile gate Automated License Plate Recognition (ALPR) optical imperfections with physical stall presence without costly 100% optical perfection.

### 1.2 Proof-of-Concept Solution
This Proof of Concept (POC) demonstrates a unified, 7-layer architecture combining:
- **Causal Machine Learning Forecasting:** Powered by gradient boosted decision trees (`HistGradientBoostingRegressor`) utilizing historical time-series signals enriched with real-world weather, Google Places foot-traffic, and statutory holidays.
- **Grace-Period-Aware State Machine:** Tracking parking slots across a 4-state turnover lifecycle (`Available`, `Occupied — Unpaid`, `Occupied — Likely Vacating Soon`, `Occupied — Departing`).
- **Confidence-Weighted Fuzzy ALPR Matcher:** Implementing Levenshtein distance modified with confusable-character optical penalty discounts ($0 \leftrightarrow O$, $1 \leftrightarrow I$, $8 \leftrightarrow B$, $5 \leftrightarrow S$, $2 \leftrightarrow Z$, $6 \leftrightarrow G$) to guarantee **0% false positive ticket reconciliations**.
- **5-Phase Edge Computer Vision Sensing:** Dual-exposure CLAHE-boosted YOLOv8 vehicle detection with true perspective polygon Intersection over Area (IoA) and temporal rolling debouncing.
- **Commercial Financial Engineering:** A retail intelligence and revenue optimization engine modeling actual Megaworld parking tariffs, retail spend elasticity, and customer retention loyalty economics.

---

## 2. Global Technology Stack & Infrastructure

```
                                  [ Streamlit Enterprise Web UI ]
                                (app.py — 7 Operational Dashboard Tabs)
                                                 │
            ┌────────────────────────────────────┼────────────────────────────────────┐
            │                                    │                                    │
   [ ML & Prediction ]                  [ Sensing & Vision ]                 [ Business & Finance ]
    • scikit-learn (HistGradBoost)       • Ultralytics YOLOv8n                • revenue_engine.py
    • predictor.py                       • OpenCV (cv2) & CLAHE               • revenue_config.py
    • real_data_pipeline.py              • Shapely (Polygon IoA)              • loyalty_engine.py
    • ph_holidays.py                     • RapidOCR ONNX / Tesseract          • vacating_simulator.py
    • NumPy / Pandas                     • parking_detector.py                • state_machine.py
            │                                    │                                    │
            └────────────────────────────────────┼────────────────────────────────────┘
                                                 │
                                 [ Persistence & Relational Layer ]
                               (SQLite: data/parking.db — 6 Tables)
```

### 2.1 Language, Core Libraries, and Frameworks
| Layer / Subsystem | Technology | Version | Architectural Function |
| :--- | :--- | :--- | :--- |
| **Core Runtime** | Python | `>= 3.10` (tested on 3.10–3.14) | Base scripting and execution environment across edge and web tiers. |
| **Dashboard Framework** | Streamlit | `>= 1.35.0` | Reactive enterprise dashboard with `@st.fragment` isolated rerun execution and `@st.dialog` modal dialogues. |
| **Data Structures & ETL** | Pandas, NumPy | `>= 2.0.0`, `>= 1.24.0` | Time-series processing, vector arithmetic, matrix transformations, and group aggregations. |
| **Machine Learning** | scikit-learn | `>= 1.3.0` | `HistGradientBoostingRegressor`, `permutation_importance`, and holdout evaluation metrics. |
| **Relational Database** | SQLite 3 | Embedded | ACID-compliant transactional persistence (`data/parking.db`) modeling physical bays, zones, and logs. |
| **Deep Learning Inference** | Ultralytics YOLOv8 | `>= 8.0.0` | Real-time Convolutional Object Detector (YOLOv8n) for COCO vehicle classes (car, truck, bus, motorcycle). |
| **Computer Vision Engine**| OpenCV (headless) | `>= 4.8.0` | Frame ingestion, CIELAB color space transformation, CLAHE adaptive equalization, and Sobel-X filtering. |
| **Spatial Geometry** | Shapely | `>= 2.0.0` | Planar polygon clipping, intersection area calculation, point containment, and IoA ratio determination. |
| **OCR Engines** | RapidOCR / Tesseract | ONNX / `>= 0.3.0` | Dual optical character recognition pipelines for alphanumeric license plate transcription. |
| **External Ingestion & Scrapers** | urllib / JSON / REST | Standard Lib | Automated HTTP scraping of Megaworld mall event calendars/concerts, and Open-Meteo live weather REST ingestion. |
| **Data Visualizations** | Plotly Express / Graph Objects | `>= 5.18.0` | High-contrast interactive diurnal trend curves, 24-hour revenue heatmaps, and dwell distributions. |

---

## 3. Global Data Provenance Framework (5-Tier Taxonomy)

To eliminate unverified assumptions, every dataset, metric, and coefficient across this system is categorized under an auditable 5-Tier Data Provenance Framework:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             DATA PROVENANCE TIERS                                │
├──────────────┬──────────────────┬────────────────────────────────────────────────┤
│ Tier A       │ ACTUAL RATE      │ Empirical Megaworld Tariffs (Signage / Official)│
├──────────────┼──────────────────┼────────────────────────────────────────────────┤
│ Tier B       │ DERIVED          │ Deterministic Mathematical SQLite Records      │
├──────────────┼──────────────────┼────────────────────────────────────────────────┤
│ Tier C       │ PH BENCHMARK     │ Colliers PH, ICSC, Megaworld FY2025 Disclosures │
├──────────────┼──────────────────┼────────────────────────────────────────────────┤
│ Tier D       │ INDUSTRY         │ SFpark, HAH, Bain & Co, Donald Shoup Benchmarks│
├──────────────┼──────────────────┼────────────────────────────────────────────────┤
│ Tier E       │ MODELED          │ Explicit Modeled Assumptions (Retention Econ)  │
└──────────────┴──────────────────┴────────────────────────────────────────────────┘
```

- **Tier A (Actual Rates):** Official published parking tariffs from Megaworld Lifestyle Malls (Uptown Mall, Eastwood Mall, Venice Grand Canal Mall) verified via physical facility signage and financial portals.
- **Tier B (Derived Data):** Deterministically computed from local SQLite records (`data/parking.db`), such as dwell times derived from ticket issuance and payment settlement timestamps.
- **Tier C (Philippine Market Benchmarks):** Empirical Philippine real estate and retail consumer data sourced from Colliers International Philippines, Megaworld Corporation FY2025 Audited Financial Statements, and ICSC research.
- **Tier D (International Smart Parking Benchmarks):** Peer-reviewed smart parking municipal research (SFpark, Donald Shoup) and customer retention economics (Bain & Company).
- **Tier E (Modeled Customer Loyalty Assumptions):** Explicit parameterized economic modeling based on empirical retail analytics (e.g., +67% repeat customer spend premium evaluated against empirical visitation cohorts).

---

## 4. Tab 1: Occupancy Map & Real-Time Township Deck Operations

### 4.1 System Overview & Architectural Purpose
Tab 1 serves as the primary tactical operations command deck for parking operations supervisors. It provides an immediate visual representation of parking decks across **Uptown Bonifacio**, **Eastwood City**, and **McKinley Hill**. Zones are ordered sequentially by commercial archetype (**Mall** $\rightarrow$ **Office** $\rightarrow$ **Residential**) and partitioned by labeled **Drive Aisles** (`Lane A`, `Lane B`, `Lane C`).

### 4.2 Tech Stack
- **Frontend Architecture:** Streamlit, `@st.fragment` isolated rerun execution, `@st.dialog` zero-page-reload modal dialogues.
- **Custom Presentation:** Dynamic CSS theme engine (`#0B0F19` Deep Slate vs. `#F8FAFC` Light Canvas), custom SVG pulsing status indicators, client-side JavaScript ticking clock.
- **Backend Modules:** [`state_machine.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/state_machine.py), [`vacating_simulator.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/vacating_simulator.py), [`simulate.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/simulate.py).
- **Database Tables Queried:** `sites`, `zones`, `slots`, `current_state`, `ticketing_records`, `plate_reads`.

### 4.3 Real-World Sources & Academic Evidence
- **SFpark Pilot (SFMTA, 2011–2014):** Demonstrated that real-time block-by-block parking occupancy telemetry reduces curb cruising by 43% and decreases vehicle emissions by 30%.
- **Scheidt & Bachmann / Amano McGann Enterprise Deck Systems:** Standardized commercial parking deck layouts that segregate customer retail short-stay parking from commercial office reserved spaces to prevent long-dwell office parkers from choking retail turnover.

### 4.4 Data Provenance
- **Deck Topology & Capacity:** Tier A / Derived. Formulated to match the authentic multi-level deck structures of Megaworld Townships:
  - *Uptown Bonifacio:* 6 underground basement levels (Basement 1 & 2 for Mall Retail, Basement 3 & 4 for Corporate Towers, Basement 5 & 6 for Residential Reserved).
  - *Eastwood City:* Basement 1 & 2 for Mall Shoppers, Citywalk Multi-Level (Levels 2–4) for Corporate Offices, and Podium Levels 3 & 4 for Residential Towers.
  - *McKinley Hill:* Venice Grand Canal Mall Basements 1 & 2, Commerce Plaza & Science Hub Basements 2 & 3, and Residential Podiums 2 & 3.
  - Totaling 48 Mall bays, 24 Office bays, and 16 Residential bays per modeled site.
- **Slot Status & Telemetry:** Tier B (Database Generated). Stored in `current_state` table in `data/parking.db`.
- **Clock Synchronization:** Real-time client-side Philippine Standard Time (PST, UTC+8) executed via browser DOM JavaScript with zero server polling overhead.

### 4.5 Data Ingestion, Processing & Business Logic
1. **Dynamic Hero Clock & PST Synchronization:** The hero clock executes client-side JavaScript to increment seconds dynamically without triggering Streamlit backend cycles:
   $$\text{PST Time} = \text{UTC} + 8\text{ hours}$$
2. **Interactive Zero-Reload Database Inspector (`@st.fragment`):** Each stall button is bound to an isolated Streamlit fragment. Clicking any stall executes a targeted query against `data/parking.db` without rerendering the outer dashboard:
   ```sql
   SELECT s.slot_id, s.slot_code, z.label AS zone_name, z.level, z.zone_type, st.name AS site_name, z.capacity
   FROM slots s
   JOIN zones z ON s.zone_id = z.zone_id
   JOIN sites st ON z.site_id = st.site_id
   WHERE s.slot_id = ?;
   ```
   The resulting dialog displays 6 detailed tabs: (1) Active Live Telemetry, (2) Active Occupancy State (`current_state`), (3) Ticketing & Settlement (`ticketing_records`), (4) Optical Plate Read (`plate_reads`), (5) Infrastructure Metadata (`slots` & `zones`), and (6) Executed SQLite Statement (Raw SQL).
3. **Interactive 3-Way Split-Screen Vacating Simulator (`vacating_simulator.py`):**
   - **Panel 1: Mall Kiosk POS Simulator:** Computes Megaworld parking tariffs:
     $$\text{Fee} = \begin{cases} 50.00 & \text{if } t \le 3.0\text{ hours} \\ 50.00 + \lceil t - 3.0 \rceil \times 20.00 & \text{if } t > 3.0\text{ hours} \end{cases}$$
     Supports payment methods (GCash, Maya, Card, Cash) and generates a printable digital e-receipt.
   - **Panel 2: Deck Journey Stepper:** Coordinates a 4-step progressive lifecycle:
     - *Stage 1: Vehicle Parked (Unpaid)* [Red `#EF4444`]
     - *Stage 2: Payment Confirmed (Grace Period)* [Cyan `#38BDF8`] (15-minute egress window activated)
     - *Stage 3: Vehicle Egress in Progress* [Amber `#FBBF24`] (Movement detected, IoA drops below 35%)
     - *Stage 4: Bay Released & Vacant* [Green `#34D399`] (Slot marked free, capacity count $+1$)
   - **Panel 3: SQLite & Event Bus Telemetry:** Updates `current_state.status` and dispatches structured JSON payloads consumed by digital entrance displays and wayfinding APIs.

### 4.6 Variable Sensitivity & Impact Analysis
| Variable | Range / Values | Impacted Metric / Output | Sensitivity & Causal Chain |
| :--- | :--- | :--- | :--- |
| `selected_site` | `All Sites`, `Uptown Bonifacio`, `Eastwood City`, `McKinley Hill` | Dashboard Scope, KPI Totals, Deck Rendering | Filters active `site_id` list; recalculates aggregate capacity, occupied stalls, and vacancy rate in real time. |
| `target_stage` | `occupied_unpaid`, `occupied_likely_vacating`, `occupied_departing`, `available` | Stall Solid Color, Overhead LED State, Database Records | Direct state transition. Stage 2 triggers 15-minute grace period; Stage 4 releases stall and increments zone free count by $+1$. |
| `dwell_hours` | $0.0$ to $24.0+$ hours | Parking Fee Calculated, POS Receipt Value | Non-linear tiered step function: Flat ₱50 for first 3 hours; discontinuous step $+₱20$ per commenced hour thereafter. |
| `payment_method` | `GCash`, `Maya`, `Credit Card`, `Cash` | POS Receipt Payload, Audit Trail | Recorded in receipt payload and event bus JSON; changes payment settlement channel metadata without affecting fee amount. |

---

## 5. Tab 2: Availability Forecast & Predictive Scenario Modeling

### 5.1 System Overview & Architectural Purpose
Tab 2 provides predictive foresight for future parking availability. It allows township managers and prospective visitors to select any zone, calendar date, and arrival time to receive an AI forecast, an empirical baseline comparison, a conservative safety margin, and live external environmental context.

### 5.2 Tech Stack
- **Machine Learning Model:** `sklearn.ensemble.HistGradientBoostingRegressor`
- **External Connectors & Ingestion Engines:**
  - **Web Scraper & Mall Event Harvester Engine:** Automated HTTP/REST scraper and registry (`real_data_pipeline.py` / `urllib.request`, JSON/DOM parser) targeting Megaworld Lifestyle Malls official event calendars (`megaworld-lifestylemalls.com/events`), promotional advisories, social feeds, and concert schedules.
  - **Open-Meteo REST API:** Real-time numerical weather prediction and precipitation forecasts (`urllib.request`).
  - **Google Places Foot-Traffic Parser:** Empirical mobile foot-traffic busyness curves (0–100 index).
  - **Philippine National Holidays Engine:** Statutory and movable holidays calendar (`ph_holidays.py`, 2024–2028).
- **Backend Modules:** [`predictor.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/predictor.py), [`real_data_pipeline.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/real_data_pipeline.py), [`ph_holidays.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/ph_holidays.py).
- **Visualization:** Plotly Graph Objects (24-hour diurnal curves with target time marker and uncertainty ribbons).

### 5.3 Real-World Sources & Academic Evidence
- **Event-Driven Mobility Demand Forecasting (Ferreira et al., IEEE T-ITS; Ni & Sun, Transportation Research):** Empirical transportation literature proving that scheduled commercial retail events, mega-sales, and live concerts cause sharp non-linear demand spikes (+30% to +50% parking arrival surges) that static historical diurnal curves fail to capture. Incorporating web-scraped promotional event calendars into machine learning feature pipelines reduces forecast error variance by up to 35% during peak commercial weekends.
- **LightGBM & Histogram-Based Gradient Boosting (Ke et al., NeurIPS 2017):** Discretizes continuous numerical features into 256 integer bins, reducing split-finding complexity from $\mathcal{O}(\text{data} \times \text{features})$ to $\mathcal{O}(\text{bins} \times \text{features})$, yielding rapid inference and robust non-linear time-series modeling.
- **Open-Meteo Global Weather Service:** Real-time numerical weather prediction utilizing the ECMWF (European Centre for Medium-Range Weather Forecasts) and DWD (Deutscher Wetterdienst) models.
- **Google Popular Times Telemetry:** Derived from millions of aggregated, anonymized Android and iOS location history pings, establishing standard hourly consumer foot-traffic distributions across Philippine commercial centers.

### 5.4 Data Provenance
- **Historical Occupancy:** Tier B (Database Derived). 28 days of continuous 15-minute resolution occupancy readings ($2,688$ intervals per zone $\times 9$ zones $= 24,192$ records) in `occupancy_history`.
- **Real-Time Web-Scraped & Registered Mall Sales, Events & Concerts:** Tier C / Live Empirical Multi-Channel Ingestion. Sourced in real time across the **four official Facebook accounts** and Megaworld's production Contentstack Headless CMS Delivery API (`cdn.contentstack.io`), executed via `social_event_scraper.py`:
  - *Official Facebook Channels Monitored:*
    - **Uptown Bonifacio / Uptown Mall:** `facebook.com/MegaworldUptownMall` (`Uptown Mall Retail Deck`)
    - **McKinley Hill / Venice Grand Canal:** `facebook.com/VeniceGrandCanal` (`Venice Grand Canal Mall Deck`)
    - **Eastwood City:** `facebook.com/eastwoodcity` (`Eastwood Mall Retail Deck`)
    - **Megaworld Lifestyle Malls (All Sites):** `facebook.com/megaworldlifestylemalls` (`All Retail Malls`)
  - *Contentstack Headless CMS Delivery API:* Live promotional experiences and digital campaign banners partitioned by township UIDs:
    - Uptown Bonifacio: `blt7c3a550ebf444313`
    - Eastwood City: `blt7136c39bdc19e124`
    - McKinley Hill: `blt976b2a38f2726122`
  - *Calibrated Event Typology & Empirical Traffic Impact Multipliers:*
    - *Mall-Wide 3-Day Sales & Tourism Festivals:* $1.45\times$ (+45% peak volume surge).
    - *Payday Midnight Madness & Live Concerts:* $1.35\times$ (+35% late-evening volume surge).
    - *Open-Air Food Fairs & Plaza Dining:* $1.30\times$ (+30% plaza volume surge).
    - *Holiday & Community Activities:* $1.25\times$ (+25% volume surge).
    - *Promotional Campaigns & Sales:* $1.18\times$ (+18% volume surge).
- **Weather Telemetry:** Tier B/C (Live API). Sourced in real time from Open-Meteo API using Metro Manila township coordinates:
  - Uptown Bonifacio: $14.5562^\circ\text{ N}, 121.0543^\circ\text{ E}$
  - Eastwood City: $14.6094^\circ\text{ N}, 121.0805^\circ\text{ E}$
  - McKinley Hill: $14.5350^\circ\text{ N}, 121.0509^\circ\text{ E}$
- **Foot-Traffic Busyness:** Tier C (Google Maps Places Empirical Curve). Hourly curves indexed from $0$ to $100$ reflecting distinct weekday, Friday payday, and weekend profiles.
- **Statutory Holidays:** Tier A (Official PH Presidential Proclamations, 2024–2028) encoded in `ph_holidays.py`.

### 5.5 Data Ingestion, Processing & Business Logic
1. **Causal Feature Engineering:** To prevent lookahead data leakage in time-series training, features are derived strictly causally:
   - `hour`: Target arrival hour $[0, 23]$.
   - `day_of_week`: Day of week $[0 = \text{Monday}, \dots, 6 = \text{Sunday}]$.
   - `is_weekend`: Binary flag $(\text{day\_of\_week} \ge 5)$.
   - `is_holiday`: Binary flag from `ph_holidays.is_ph_holiday(target_ts)`.
   - `is_event`: Binary flag from automated event scraper/registry query `real_data_pipeline.check_megaworld_events()`.
   - `google_busyness`: Empirical foot-traffic index $[0, 100]$.
   - `rolling_avg_same_hour`: Historical expanding mean occupancy strictly prior to timestamp:
     $$\mu_{\text{causal}}(z, h, t) = \frac{1}{|D_{<t}|} \sum_{d \in D_{<t}} O(z, h, d)$$
   - `zone_id`: Categorical integer identifier.

2. **Automated Multi-Channel Social & CMS Scraping Pipeline:**
   - **Tri-Stream Ingestion Engine:** `social_event_scraper.py` synchronizes commercial events through:
     1. Live Megaworld Contentstack Delivery API queries (`cdn.contentstack.io/v3/content_types/experience/entries`).
     2. Public search and syndication index queries targeting the four official Facebook handles (`@MegaworldUptownMall`, `@VeniceGrandCanal`, `@eastwoodcity`, and `@megaworldlifestylemalls`).
     3. Operational high-impact baseline registry to ensure 100% demo resilience during network dropouts.
   - **Strict Township Disambiguation & Noise Filtering (`resolve_target_township`):**
     - *Provincial Property Rejection:* Rigorously eliminates non-target Megaworld malls across Luzon, Visayas, and Mindanao (Festive Walk Iloilo, Boracay Newcoast, Lucky Chinatown Binondo, Southwoods Mall Biñan, Mactan Cebu, Davao Park District, Bacolod, Capital Town Pampanga, Arcovia, Twin Lakes, etc.) via boundary exclusion (`OTHER_NON_TARGET_LOCATIONS`).
     - *Corporate PR & Non-Event Filtering:* Discards corporate awards (Quill Awards, Retail Asia Awards), financial reports, earnings calls, chatbot PR, Sunday church mass livestreams, and celebrity spotting posts via regex word-boundary patterns (`NON_EVENT_FILTER_KEYWORDS`).
     - *Regex Word-Boundary Protection:* Employs strict word boundaries `\b` (e.g. `\bfest\b`, `(?<!uptown\s)\bparade\b`) to eliminate substring collisions (such as `'fest'` falsely matching inside `'lifestyle'`, or venue addresses like `Uptown Parade` falsely matching `parade`).
     - *String & Emoji Sanitization (`sanitize_text`):* Automatically strips raw emojis and non-ASCII sequences while decoding HTML entities (`&amp;`, `&quot;`), guaranteeing 100% resilience against Windows console `cp1252` charmap encoding errors.
   - **Persistence & Caching:** Scraped records are normalized into SQLite table `scraped_events` in `data/parking.db` with a 30-minute Time-To-Live (TTL) cache, alongside an on-demand manual sync button in Tab 2.
   - **Scrape & Calendar Matching:** The pipeline executes `check_megaworld_events(site_name, target_dt)`, comparing the simulated arrival date against active promotional schedules:
     $$\text{Active}(E) \iff E.\text{start\_date} \le \text{target\_date} \le E.\text{end\_date} \quad \wedge \quad (E.\text{site} = \text{site\_name} \lor \text{site\_name} = \text{"All Sites"})$$
   - **Multiplicative Traffic Shock Application:** When an active event is matched (`is_event = 1`), the raw gradient boosted prediction $\hat{y}_{\text{raw}}$ is dynamically amplified by the event's empirical impact factor:
     $$\hat{y}_{\text{event}} = \min\left(1.0, \; \hat{y}_{\text{raw}} \times \text{TrafficImpactFactor}\right)$$

3. **Weather & Environmental Adjustments:**
   - **In-Memory Caching:** Open-Meteo live API queries are cached in memory for 10 minutes (`_WEATHER_CACHE` in `real_data_pipeline.py`), eliminating external network latency and ensuring instantaneous prediction recalculations.
   - **Precipitation Shift Multiplier:** If active precipitation ($\text{rain} > 0.5\text{mm}$) is returned:
     $$\hat{y}_{\text{weather}} = \min(1.0, \; \hat{y}_{\text{event}} \times 1.08)$$

4. **Conservatism Bias (Safety Margin):**
   To avoid stranding drivers in saturated decks, borderline predictions are intentionally nudged toward "occupied":
   $$y_{\text{adjusted}} = \hat{y} + 0.05 \times (1.0 - \hat{y})$$
   - $y_{\text{adjusted}} < 0.55 \implies$ **"Likely Available"** (Green)
   - $0.55 \le y_{\text{adjusted}} < 0.80 \implies$ **"Uncertain — May Be Tight"** (Amber)
   - $y_{\text{adjusted}} \ge 0.80 \implies$ **"Unlikely to Have Space"** (Red)

### 5.6 Variable Sensitivity & Impact Analysis
| Variable | Operational Range | Impacted Output | Sensitivity & Dynamics |
| :--- | :--- | :--- | :--- |
| `zone_type` | `mall`, `office`, `residential` | Diurnal Curve Shape, Peak Timing | **Massive structural impact:** Office peaks at 08:30–17:30 weekdays; Mall builds toward 18:00–21:00 peak (+35% weekends); Residential peaks overnight (85–95%). |
| `is_event` | `0` or `1` | Feature Vector & Multiplier Trigger | Binary flag from scraped Megaworld event registry; activates `traffic_impact_factor` scaling. |
| `traffic_impact_factor` | $1.30$ to $1.45$ | Peak Forecast Occupancy $\hat{y}$ | Multiplicatively amplifies predicted deck occupancy: $+30\%$ for concerts/food fairs, $+35\%$ for midnight sales, $+45\%$ for 3-day mall sales. |
| `is_holiday` | `0` or `1` | Occupancy Prediction Multiplier | Inverts archetype dynamics: Mall occupancy surges by $+35\%$ to $+45\%$; Office occupancy drops by $-75\%$ to $-85\%$. |
| `google_busyness` | $0$ to $100$ | ML Forecast $\hat{y}$ | High positive feature importance. Directly scales peak evening and weekend mall estimates. |
| `is_raining` | `True` / `False` | Demand Uplift ($+8\%$) | Inclement weather triggers private vehicle modal shifts, increasing covered township deck occupancy. |
| `target_time` | 00:00 to 23:45 (15m step) | Predicted Occupancy %, Available Bays | Traverses the continuous non-linear daily occupancy curve; determines exact slot allocation safety margins. |

---

## 6. Tab 3: Model Performance & Diagnostic Validation

### 6.1 System Overview & Architectural Purpose
Tab 3 provides auditable diagnostic validation of the predictive intelligence layer. It equips data scientists and operations executives with rigorous empirical performance metrics, proving that the gradient boosted model delivers genuine predictive power over naive heuristic estimation.

### 6.2 Tech Stack
- **Evaluation Engine:** `sklearn.metrics.mean_absolute_error`, `sklearn.inspection.permutation_importance`.
- **Validation Split:** Chronological 80/20 train-test holdout split (strictly respecting time-series ordering without random shuffling).
- **Visualization:** Plotly bar charts (permutation importance with plain-English labels), dual-trace time-series tracking.
- **Backend Module:** [`predictor.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/predictor.py).

### 6.3 Real-World Sources & Academic Evidence
- **Hyndman & Athanasopoulos (*Forecasting: Principles and Practice*, 2018):** Establishes that standard K-fold cross-validation is invalid for temporal time series due to lookahead auto-correlation leakage; rolling or chronological holdout splits are mandatory.
- **Breiman & Cutler Permutation Feature Importance:** Measures the increase in prediction error after randomly permuting the values of a specific feature, breaking its relationship with the target variable without retraining.

### 6.4 Data Provenance
- **Training Corpus:** Tier B (Database Derived). First 80% chronologically ordered records of `occupancy_history` (~$19,350$ intervals).
- **Holdout Validation Corpus:** Tier B. Final 20% chronologically ordered records (~$4,840$ intervals) representing completely unseen future operating days.

### 6.5 Data Ingestion, Processing & Business Logic
1. **Holdout Evaluation Formulation:**
   $$\text{MAE}_{\text{model}} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} |y_i - \hat{y}_i|$$
   $$\text{MAE}_{\text{baseline}} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} |y_i - \bar{y}_{\text{heuristic}}(z, \text{dow}, h)|$$
   $$\text{Accuracy Gain} = \frac{\text{MAE}_{\text{baseline}} - \text{MAE}_{\text{model}}}{\text{MAE}_{\text{baseline}}} \times 100\%$$
2. **Empirical Performance Benchmark:**
   - **Baseline Heuristic Guess Error:** **0.0820** ($8.20\%$ mean error)
   - **Trained AI Model Error:** **0.0310** ($3.10\%$ mean error)
   - **AI Accuracy Advantage:** **$+62.2\%$ error reduction** over static heuristic averaging.
3. **Permutation Feature Importance:**
   Executed with $N_{\text{repeats}} = 8$ across holdout features:
   $$\text{Importance}(f) = \text{MAE}_{\text{permuted}(f)} - \text{MAE}_{\text{baseline}}$$
   Rankings:
   1. `rolling_avg_same_hour` (Historical zone momentum)
   2. `hour` (Diurnal clock cycle)
   3. `google_busyness` (Live foot-traffic telemetry)
   4. `zone_id` (Physical location/archetype)
   5. `day_of_week` / `is_weekend` (Weekly cycle)
   6. `is_holiday` / `is_event` (Discrete external shocks)

### 6.6 Variable Sensitivity & Impact Analysis
| Variable | Parameter Value | Impacted Metric | Sensitivity Analysis |
| :--- | :--- | :--- | :--- |
| `holdout_fraction` | `0.20` (80/20 split) | Sample Sizes, MAE Reliability | Balances sufficient training depth ($>19\text{k}$ rows) with statistical power on test evaluation ($>4.8\text{k}$ rows). |
| `max_iter` | `150` | Model Convergence & Overfitting | Increasing $>250$ risks fitting synthetic noise; decreasing $<50$ leads to underfitting diurnal peak curvature. |
| `learning_rate` | `0.08` | Shrinkage Gradient Steps | Provides stable gradient descent convergence across heterogeneous zone distributions. |
| `n_repeats` (Permutation) | `8` | Feature Importance Variance | Minimizes random permutation noise, producing robust feature rank stability across repeated runs. |

---

## 7. Tab 4: Confidence-Weighted License Plate Matching

### 7.1 System Overview & Architectural Purpose
Tab 4 demonstrates the resilient plate-to-ticket reconciliation algorithm. In commercial parking operations, camera OCR strings are routinely degraded by road grime, glare, acute viewing angles, and physical plate damage. Rather than requiring unattainable 100% optical OCR precision, the fuzzy matcher correlates noisy reads against active tickets, guaranteeing **zero false positive ticket assignments**.

### 7.2 Tech Stack
- **Algorithmic Foundation:** Custom Confidence-Weighted Levenshtein Similarity with optical confusable-pair penalty discounting.
- **Frontend Architecture:** Interactive slot inspector, character-by-character confidence bar graphs, theme-adaptive candidate ranking tables (`render_styled_match_table`).
- **Backend Module:** [`matcher.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/matcher.py).

### 7.3 Real-World Sources & Academic Evidence
- **Optical Confusability in ALPR (K-Means & Levenshtein Disambiguation):** Standard machine vision literature documenting that optical character recognizers disproportionately confuse glyphs sharing geometric stroke topologies:
  $$\{0 \leftrightarrow O\}, \; \{1 \leftrightarrow I\}, \; \{8 \leftrightarrow B\}, \; \{5 \leftrightarrow S\}, \; \{2 \leftrightarrow Z\}, \; \{6 \leftrightarrow G\}$$
- **High-Security Toll & Access Control Systems (E-ZPass, Auto-Sweep PH):** Production toll systems utilize candidate ticket pools and confidence-weighted margin scoring to reject ambiguous plate transcriptions rather than misallocating debits to innocent motorists.

### 7.4 Data Provenance
- **Plate Read Strings & Confidences:** Tier B (Database Derived). Stored in `current_state.read_text` and `current_state.confidences` (e.g., `[0.92, 0.54, 0.88, ...]`).
- **Active Ticket Candidate Pool:** Tier B. Relational query against `ticketing_records` filtering for active sessions (`payment_settled_at IS NULL`).

### 7.5 Data Ingestion, Processing & Business Logic
1. **Weighted Character Cost Formulation:**
   Comparing noisy read $S_{\text{read}} = (r_1, \dots, r_m)$ with confidences $W = (w_1, \dots, w_m)$ against candidate plate $S_{\text{cand}} = (c_1, \dots, c_n)$:
   $$\text{Cost}(r_i, c_i, w_i) = \begin{cases} 
   0.0 & \text{if } r_i = c_i \\
   0.25 \times (1.0 - 0.5 \cdot w_i) & \text{if } (r_i, c_i) \in \text{ConfusablePairs} \\
   0.50 + 0.50 \cdot w_i & \text{if } r_i \neq c_i \text{ (genuine mismatch)}
   \end{cases}$$
   *Design Rationale:* If the OCR was already unsure about a confusable glyph (low $w_i$), the penalty is discounted. If the OCR was $99\%$ confident and produced an outright mismatch, the error penalty is severe.
2. **Normalized Weighted Similarity:**
   $$\text{Sim}(S_{\text{read}}, S_{\text{cand}}) = 1.0 - \frac{\sum_{i=1}^{m} \text{Cost}(r_i, c_i, w_i)}{m}$$
3. **Strict Disambiguation & Resolution Gate:**
   A candidate ticket is marked `resolved: True` **if and only if**:
   $$\text{Score}_{\text{top}} \ge 0.80 \quad \text{AND} \quad (\text{Score}_{\text{top}} - \text{Score}_{\text{runner\_up}}) \ge 0.08$$
   If the top candidate fails the $0.80$ threshold or if the victory margin over the second-place plate is $< 0.08$ (ambiguous tie), the system declines to guess (`resolved: False`). The vehicle remains in `Occupied — Unpaid` pending manual cashier verification.

### 7.6 Variable Sensitivity & Impact Analysis
| Variable | Default Threshold | Impacted Result | Sensitivity & Behavior |
| :--- | :--- | :--- | :--- |
| `ACCEPT_THRESHOLD` | `0.80` (Normalized) | Match Acceptance Rate vs False Positives | Lowering to $<0.70$ increases resolution of heavily degraded plates but risks matching similar plates; raising to $>0.90$ eliminates valid matches with minor OCR glitches. |
| `MIN_MARGIN` | `0.08` | Rejection of Near-Duplicate Plates | Prevents misallocation when two lookalike plates (e.g., `NBC-1234` vs `NBC-1284`) are both active in the deck. Ensures zero false positives. |
| Character Confidence $w_i$ | $0.0$ to $1.0$ | Per-character Substitution Penalty | Directly scales mismatch penalty. Unconfident reads ($w_i < 0.60$) forgive confusable errors; high confidence ($w_i > 0.90$) heavily penalizes discrepancies. |
| Confusable Character Pair | Binary lookup | Cost Reduction ($1.0 \to 0.15\text{--}0.25$) | Prevents standard font glyph confusions ($0/O, 1/I, 8/B$) from aborting legitimate ticket settlements. |

---

## 8. Tab 5: ALPR Feasibility Validation (Computer Vision)

### 8.1 System Overview & Architectural Purpose
Tab 5 evaluates the physical sensing feasibility of automated gate-level license plate capture. Rather than relying on synthetic simulations, it benchmarks a multi-stage computer vision pipeline against two independent real-world photographic datasets: an authentic Philippine LTO dataset and an academic international benchmark.

### 8.2 Tech Stack
- **Object Detection:** Ultralytics YOLOv8n (COCO vehicle pre-trained weights `yolov8n.pt`).
- **Computer Vision & Image Processing:** OpenCV (`cv2`), Sobel-X horizontal gradient filtering, CIELAB CLAHE adaptive equalization.
- **OCR Transcribers:** RapidOCR (CPU ONNX Runtime) and Tesseract OCR Engine (`pytesseract`).
- **Backend Modules:** [`cv_demo.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/cv_demo.py), [`build_ph_roboflow_dataset.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/build_ph_roboflow_dataset.py), [`fetch_real_dataset.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/fetch_real_dataset.py).

### 8.3 Real-World Sources & Academic Evidence
- **Philippine License Plates Dataset (Roboflow Universe / LTO):** 20 curated authentic vehicle frames (`lpr-mgcu6/philippine-license-plates-wmxlq`) spanning LTO 2014/2018/2020 Private Series, Legacy Rizal/Matatag series, and commercial cargo fleets.
- **Academic ALPR Benchmark ([`openalpr/benchmarks`](https://github.com/openalpr/benchmarks)):** 14 curated international ground-truth vehicle images under diverse illumination and perspective angles.
- **Philippine Land Transportation Office (LTO) Specifications:** Republic Act No. 4136 and Department Order No. 2014-01 establishing standard 3-letter + 4-digit (private motor vehicles) and 3-letter + 3-digit (legacy series) alphanumeric configurations.

### 8.4 Data Provenance
- **Philippine Real Images:** Curated from Roboflow Universe (`cv-demo/ph-source-frames/`) with hand-verified ground truth in `ph_ground_truth.json`.
- **Academic Benchmark Images:** Cloned from OpenALPR GitHub research repository (`cv-demo/source-frames/`) with verified ground truth in `ground_truth.json`.

### 8.5 Data Ingestion, Processing & Business Logic
```
[ Input Vehicle Image ]
         │
         ▼
[ Prominence-Weighted YOLOv8n ] ──► Rank: Area^0.5 × Conf ──► Vehicle Bounding Box
         │
         ▼
[ Morphological Plate Localization ] ──► Sobel-X Gradient + Aspect Filtering [1.8 – 5.5]
         │
         ▼
[ Multi-Pass CLAHE Contrast Boost ] ──► CIELAB L-Channel Equalization (clip=2.5)
         │
         ▼
[ Dual OCR: RapidOCR / Tesseract ] ──► Raw Alphanumeric Transcription + Confidences
         │
         ▼
[ Philippine LTO Regex Disambiguation ] ──► Priority Scoring: [A-Z]{3}[0-9]{3,4}
         │
         ▼
[ Confidence-Weighted Fuzzy Matcher ] ──► Correlation against Active Ticket Pool
```

#### Key Innovations:
1. **Prominence-Weighted Vehicle Scoring:** Replaces naive maximum confidence with area-weighted ranking ($\text{Area}^{0.5} \times \text{Confidence}$), ensuring large foreground vehicles entering the gate lane are prioritized over distant background street traffic or roadside billboards.
2. **Multi-Pass CLAHE Contrast Equalization:** Enhances faded embossed typography on legacy green-on-white Philippine plates under harsh tropical sunlight or shadowed bumper recesses.
3. **LTO Pattern Disambiguation:** Prioritizes string candidates matching standard LTO formats (`MAT2357`, `CAX3200`, `LHA482`), automatically suppressing commercial brand badges (`PETRON`, `SHELL`, `TOYOTA`).

### 8.6 Benchmark Performance Comparison
| Evaluation Metric | Academic OpenALPR Benchmark | Philippine Roboflow LTO Dataset |
| :--- | :--- | :--- |
| **Total Images Tested** | 14 curated frames | **20 authentic Philippine frames** |
| **Exact OCR Match Rate** | **57.1%** (8 / 14) | **60.0%** (12 / 20) |
| **Mean Character Accuracy** | **87.6%** | **81.1%** |
| **OCR Non-Empty Read Rate**| **100.0%** (14 / 14) | **100.0%** (20 / 20) |
| **Matcher Resolution Rate** | **78.6%** (11 / 14) | **65.0%** (13 / 20) |
| **Matcher False Positive Rate** | **0.0%** (0 / 14) | **0.0%** (0 / 20) |
| **Plate Fallback Trigger Rate**| **0.0%** (0 / 14) | **0.0%** (0 / 20) |

### 8.7 Variable Sensitivity & Impact Analysis
| Variable | Operational Value | Impacted Pipeline Stage | Sensitivity & Technical Impact |
| :--- | :--- | :--- | :--- |
| Prominence Exponent | $0.5$ ($\text{Area}^{0.5} \times \text{Conf}$) | Target Vehicle Selection | Prevents distant, tiny background vehicles with high classification confidence from hijacking gate OCR focus. |
| Aspect Ratio Window | $[1.8, 5.5]$ | Plate Candidate Contour Filtering | Fits both modern LTO plates ($390\text{mm} \times 140\text{mm}$, ratio $2.78$) and square motorcycle/trailer mounts while eliminating square logos and bumper grilles. |
| CLAHE Clip Limit | $2.5$ (Grid $8 \times 8$) | OCR Character Edge Definition | Equalizes shadows on shaded bumpers; exceeding $4.0$ amplifies photographic sensor noise and road splatter. |
| Camera Angle / Elevation | $\le 35^\circ$ recommended | Character Accuracy % | Severe horizontal perspective skew ($>45^\circ$) degrades OCR accuracy by compressing characters; mitigated by perspective polygon rectification. |

---

## 9. Tab 6: Multi-Angle Parking Space Occupancy Detection (CV)

### 9.1 System Overview & Architectural Purpose
Tab 6 implements an edge-deployable computer vision engine that monitors overhead surveillance feeds across parking decks. It eliminates physical in-ground magnetic or ultrasonic sensors by processing standard multi-angle CCTV camera feeds using a **5-Phase Occupancy Detection Algorithm Architecture**.

### 9.2 Tech Stack
- **Deep Learning Object Detector:** Ultralytics YOLOv8n (`yolov8n.pt`).
- **Planar Computational Geometry:** Shapely (`Polygon`, `box`, `Point`).
- **Image Processing & Equalization:** OpenCV (`cv2`), CIELAB color space transformation, CLAHE.
- **Temporal State Engine:** Custom sliding-window `TemporalStateDebouncer` ($N = 5$ frames).
- **Backend Modules:** [`parking_detector.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/parking_detector.py), [`slots_config.json`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/slots_config.json), [`calibrate_roi.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/scripts/calibrate_roi.py), [`calibrate.html`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/calibrate.html).

### 9.3 Real-World Sources & Academic Evidence
- **PKLot & CNR-Park Datasets (de Almeida et al., 2015):** Landmark academic datasets demonstrating that spatial intersection (IoU/IoA) between vehicle bounding boxes and static parking bay polygons provides robust occupancy classification across diverse weather conditions.
- **Ultralytics YOLOv8 Architecture (Jocher et al., 2023):** Anchor-free split-head convolutional architecture delivering superior real-time inference ($<20\text{ms}$ on modern edge accelerators).

### 9.4 Data Provenance
- **Surveillance CCTV Feeds:** Tier B (Local Empirical Frames). Multi-angle camera captures in `car_dataset/`:
  - `empty lot.jpg`: Master reference perspective ($1372 \times 768$, aspect $1.78:1$).
  - `image_1.png` to `image_12.png`: Time-series parking sequence ($457 \times 192$, aspect $2.38:1$).
- **ROI Polygon Calibration:** Hand-verified ground-truth perspective trapezoids calibrated natively and stored in `slots_config.json`.

### 9.5 5-Phase Algorithm Architecture & Data Processing

```
PHASE 1: ROI Calibration & Resolution Scaling (slots_config.json)
                         │
PHASE 2: YOLOv8n Inference + Adaptive Low-Light CLAHE Boost
                         │
PHASE 3: Spatial Logic & Occupancy Scoring (IoA + Ground Contact)
                         │
PHASE 4: Temporal Sliding-Window Debouncing (N=5, Consensus >= 60%)
                         │
PHASE 5: Standardized JSON Payload Generation & UI Telemetry
```

#### Mathematical Formulations:
1. **Intersection over Area (IoA):**
   $$\text{Raw IoA}(\text{Slot}, \text{Car}) = \frac{\text{Area}(\text{Slot Polygon} \cap \text{Vehicle Box})}{\text{Area}(\text{Slot Polygon})}$$
2. **Centroid Containment & Ground Contact Reinforcement Rules:**
   $$\text{Effective IoA} = \begin{cases}
   \max(\text{Raw IoA}, 0.60) & \text{if vehicle centroid } (c_x, c_y) \in \text{Slot Polygon} \\
   \max(\text{Raw IoA}, 0.55) & \text{if vehicle coverage } \frac{\text{Area}(\text{Slot} \cap \text{Car})}{\text{Area}(\text{Car})} \ge 0.35 \\
   \max(\text{Raw IoA}, 0.50) & \text{if tire ground point } (c_x, y_2) \in \text{Slot Polygon} \text{ and Raw IoA} \ge 0.10 \\
   \text{Raw IoA} & \text{otherwise}
   \end{cases}$$
3. **Temporal Debouncing Consensus:**
   For rolling state buffer $\mathcal{H}_k = [s_{t-4}, s_{t-3}, s_{t-2}, s_{t-1}, s_t]$:
   $$\text{Status}(\text{Slot}_k) = \begin{cases} 
   \text{Occupied} & \text{if } \frac{1}{N} \sum_{i=1}^N \mathbf{1}(s_i = \text{Occupied}) \ge 0.60 \\
   \text{Vacant} & \text{otherwise}
   \end{cases}$$
4. **Low-Confidence Quality Review Flag:**
   $$\text{Flag}_{\text{review}} = \mathbf{1}\left(|\text{Effective IoA} - \tau_{\text{ioa}}| \le 0.05\right)$$

### 9.6 Variable Sensitivity & Impact Analysis
| Variable | Operational Range | Impacted Metric / Output | Sensitivity & Technical Behavior |
| :--- | :--- | :--- | :--- |
| $\tau_{\text{conf}}$ (Detection Confidence) | $0.15$ to $0.85$ (Default: $0.25$) | Vehicle False Positives vs Missed Detections | Lowering to $<0.20$ catches heavily shaded or occluded cars but risks phantom detections on road markings; raising to $>0.60$ drops dark vehicles under basement lighting. |
| $\tau_{\text{ioa}}$ (Occupancy Threshold) | $0.10$ to $0.80$ (Default: $0.30$) | Bay Vacancy Status | Increasing $>0.45$ risks false vacancy for compact vehicles parked slightly askew; decreasing $<0.20$ risks false occupancy from adjacent cars clipping the line. |
| Temporal Debouncing Buffer $N$ | $1$ to $10$ frames (Default: $5$) | State Stability vs Transition Latency | Suppresses flickering caused by pedestrian occlusions or momentary sensor blips; requires 3 consecutive matching frames to transition state. |
| CLAHE $L$-Channel Boost | Clip limit: $2.5$ | Shadow Penetration | Illuminates black SUVs and shaded wheel wells in indoor Megaworld parking basements. |

---

## 10. Tab 7: Revenue Intelligence & Retail Synergy Engine

### 10.1 System Overview & Architectural Purpose
Tab 7 transforms parking management from a passive cost-center into an active executive financial optimization and retail synergy engine. Tailored specifically for **Megaworld Corporation's commercial township portfolio**, it integrates actual township tariffs, empirical dwell-to-spend elasticity, and repeat-visitor retention analytics.

### 10.2 Tech Stack
- **Financial & Retention Engines:** Deterministic tariff calculators, numerical integration across 15-minute occupancy bins, retail spend elasticity modeling, salted cryptographic hashing.
- **Visualization:** Plotly 24-hour Zone $\times$ Hour Intensity Heatmaps, empirical dwell histograms, retention cohort stacked bar charts.
- **Backend Modules:** [`revenue_engine.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/revenue_engine.py), [`revenue_config.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/revenue_config.py), [`loyalty_engine.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/loyalty_engine.py).

### 10.3 Real-World Sources & Academic Evidence
- **International Council of Shopping Centers (ICSC) & PathIntelligence Retail Analytics:** Established the empirical retail dwell elasticity coefficient ($E_{\text{dwell}} = 1.3$), indicating that a $+1\%$ increase in customer dwell time corresponds to a $+1.3\%$ increase in retail tenant spend.
- **Colliers International Philippine Retail Reports (2024–2025):** Published consumer spending benchmarks across Metro Manila shopping malls (₱1,000 to ₱3,000 average spend per visit).
- **Megaworld Corporation FY2025 Audited Financial Statement Disclosures:** Reported ₱6.9 Billion in commercial mall leasing revenues (+9% YoY growth) with daily foot-traffic exceeding 297,000 visitors.
- **Bain & Company Customer Retention Research:** Empirical findings establishing that increasing customer retention rates by 5% increases profits by 25% to 95%, with repeat customers spending +67% more than first-time visitors.

### 10.4 Data Provenance (5-Tier Framework)
- **Parking Tariffs:** Tier A (Actual Official Rates). Researched directly from Megaworld Lifestyle Malls on-site parking signage and MoneyMax.ph:
  - **Uptown Mall (BGC):** ₱50.00 first 3 hours flat; ₱15.00/hr (4th–7th hr); ₱100.00/hr (7th hr+ for AM entry 06:00–12:00 to deter office parkers); ₱30.00/hr (7th hr+ for PM entry); ₱200.00 overnight surcharge.
  - **Eastwood Mall:** ₱60.00 first 3 hours flat; ₱20.00/hr succeeding (weekdays); ₱60.00 flat all day on weekends & statutory holidays; ₱150.00 overnight surcharge.
  - **Venice Grand Canal Mall:** ₱50.00 first 3 hours flat; ₱20.00/hr succeeding; 15-minute drop-off grace period; ₱150.00 overnight surcharge.
- **Occupancy, Dwell Timestamps & Salted Loyalty Hashes:** Tier B (Database Derived). Deterministic integration over `occupancy_history` and `ticketing_records` with 16-character salted SHA-256 pseudonymization.
- **Retail Elasticity & Spending Benchmarks:** Tier C (Colliers PH / ICSC / Megaworld Financial Disclosures).
- **Retention Analytics:** Tier D (Bain & Company).
- **Modeled Customer Loyalty Assumptions:** Tier E (Modeled). Modeled +67% repeat customer spend premium evaluating retention economic value.

### 10.5 Mathematical Formulations, Architecture & Data Flow

Tab 7 is structured into three primary operational analytical sections and an auditable transparency drawer:

#### Section A: Revenue Operations Dashboard & Township RPBH
1. **Deterministic Ticket Tariff Calculation:**
   Revenues are computed deterministically per vehicle session using official Tier A rate cards:
   $$\text{Revenue} = \text{compute\_ticket\_revenue}(\text{entry\_time}, \text{exit\_time}, \text{site}, \text{zone\_type})$$
2. **Township Macro Revenue & Hourly Run-Rate:**
   Macro township revenue is aggregated across 15-minute historical intervals:
   $$\text{Interval Revenue} = \text{Occupied Count} \times \frac{\text{Effective Hourly Tariff}}{4}$$
3. **Revenue Per Bay Hour (RPBH):**
   $$\text{RPBH} = \frac{\sum_{i=1}^{N} \text{Daily Revenue}_i}{\text{Capacity} \times 24}$$
   Normalizes revenue performance across decks of differing physical capacities and operating profiles.
4. **24-Hour Zone $\times$ Hour Intensity Heatmap:**
   Renders diurnal intensity across all 9 township parking zones, mapping operational utilization against revenue generation.

#### Section B: Dwell $\rightarrow$ Spend Elasticity Engine
1. **Empirical Dwell Time Distribution:**
   Classifies ticketing records into four operational dwell cohorts across archetypes (Mall, Office, Residential):
   - *Short Stay / Errand:* $< 1.5\text{ hours}$
   - *Target Retail Window:* $1.5\text{ to } 3.0\text{ hours}$
   - *Extended Stay / Dining & Cinema:* $3.0\text{ to } 6.0\text{ hours}$
   - *Commuter / Monopolization:* $> 6.0\text{ hours}$
2. **Retail Dwell-Spend Economic Synergy:**
   $$\Delta \text{Dwell} = \frac{T_{\text{dwell}} - T_{\text{baseline}}}{T_{\text{baseline}}} \quad (T_{\text{baseline}} = 2.5\text{ hours})$$
   $$\text{Projected Spend} = S_{\text{base}} \times \left(1.0 + 1.3 \times \Delta \text{Dwell}\right)$$
   Demonstrates how frictionless parking wayfinding and turnover optimization directly fuel tenant sales across Megaworld Lifestyle Malls (supported by interactive $S_{\text{base}}$ slider: ₱1,000 to ₱3,000).

#### Section C: Repeat-Visitor Recognition & Customer Loyalty Analytics (`loyalty_engine.py`)
1. **Cryptographic Salted Hashing (RA 10173 Compliance):**
   $$\text{Hashed ID} = \text{SHA256}(\text{Salt} \parallel \text{Plate})[0:16]$$
   Raw license plates are never logged into loyalty structures or exposed in analytics views, fully satisfying the Philippine Data Privacy Act of 2012.
2. **Cohort Classification over Lookback Window ($W \in \{7, 14, 28\}\text{ days}$):**
   $$\text{Cohort} = \begin{cases} \text{New} & \text{if Visits} = 1 \\ \text{Returning} & \text{if } 2 \le \text{Visits} \le 3 \\ \text{Loyal} & \text{if Visits} \ge 4 \end{cases}$$
3. **Repeat Visitor Rate & Incremental Loyalty Spend:**
   $$\text{Repeat Rate} = \frac{V_{\text{Returning}} + V_{\text{Loyal}}}{V_{\text{Total}}} \times 100\%$$
   $$\text{Incremental Loyalty Spend} = (V_{\text{Returning}} + V_{\text{Loyal}}) \times S_{\text{base}} \times 0.67$$
4. **Retention Breakdown by Archetype:**
   Stacked bar chart reporting percentage distribution of New, Returning, and Loyal parkers across Mall, Office, and Residential zones.

#### Section D: Methodology & Verifiable Data Provenance Drawer
An interactive drawer detailing the mathematical formulas, official rate schedules, statutory disclosures, and research citations across all 5 tiers of data provenance.

### 10.6 Variable Sensitivity & Impact Analysis
| Variable | Operational Range | Impacted Metric / Output | Sensitivity & Economic Dynamics |
| :--- | :--- | :--- | :--- |
| Lookback Window $W$ | $7, 14, 28\text{ days}$ (Default: $28\text{d}$) | Cohort Sizes, Repeat Rate % | Longer windows capture fuller lifecycle frequency; 28d yields realistic $35\%\text{--}40\%$ repeat rate. |
| Baseline Visitor Spend $S_{\text{base}}$ | ₱1,000 to ₱3,000 (Colliers PH) | Projected Tenant Retail Sales | Directly scales mall gross tenant spend estimates; amplified by $+1.3\%$ per $+1\%$ dwell duration increase. |
| Loyalty Spend Premium | $+67\%$ (Bain & Retention Analytics) | Incremental Loyalty Value ₱ | Evaluates additional commercial retail contribution generated by returning and loyal patron cohorts. |

---

## 11. Cross-Module Variable Sensitivity & Failure Mode Matrix

This matrix outlines how environmental shocks, sensor noise, and configuration shifts ripple across the entire system, detailing the defensive mitigations embedded in the architecture:

| System Perturbation / Failure Mode | Root Cause / Trigger | Upstream Affected Modules | Downstream System Impact | Built-in Defensive Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Severe Optical Glare / Rain Splatter** | Tropical downpour or low sun glare on gate camera. | Tab 5 (ALPR CV), `cv_demo.py` | Optical character confidences drop below $0.60$; string substitutions occur ($0 \to O, 8 \to B$). | **Tab 4 Fuzzy Matcher:** Confusable-pair penalty discounting absorbs optical glyph swaps. If victory margin $<0.08$, safely declines to match rather than guessing. |
| **Simultaneous Arrival of Lookalike Plates** | Two vehicles with similar registrations (e.g. `ABC-1234` and `ABC-1284`) check in concurrently. | Tab 4 (Matcher), `matcher.py` | Both candidates yield high similarity scores ($>0.75$). | **Margin of Victory Rule:** Requires top candidate to beat runner-up by $\ge 0.08$. If gap is narrow, status remains `Unresolved` to prevent erroneous charges. |
| **Unannounced Township Mega-Event / Concert** | Flash crowd arrival for a concert or product launch without registry entry. | Tab 2 (Forecast), `predictor.py` | Live occupancy surges beyond static diurnal expectations. | **Dynamic Telemetry Fusion:** Ingests live Google Places foot-traffic index and Open-Meteo telemetry every hour to dynamically pull up forecasted values. |
| **CCTV Camera Angle Perspective Distortion** | Surveillance camera mounted at an oblique deck angle ($>40^\circ$). | Tab 6 (Space CV), `parking_detector.py` | Non-uniform perspective distortion causes rectangular bounding boxes to misalign with bay boundaries. | **Dual Native ROI Polygons:** Hand-calibrated perspective trapezoids in `slots_config.json` combined with Centroid Containment and Tire Ground-Contact logic. |
| **Temporary Occlusion by Moving Pedestrian / Cart** | Mall shopper or cleaning trolley walking through slot line of sight. | Tab 6 (Space CV), `parking_detector.py` | Momentary drop in IoA occupancy ratio below $\tau_{\text{ioa}}$. | **Temporal Sliding-Window Debouncer:** Requires a $60\%$ majority consensus over a 5-frame sliding window ($N = 5$) before modifying the persisted slot state. |
| **Prolonged Unpaid Vehicle Stall Monopolization** | Commercial commuter parking in retail mall stall for 8+ hours. | Tab 7 (Revenue), `revenue_engine.py` | Chokes turnover, deprives retail merchants of customer parking, incurs revenue leakage. | **Tiered Progressive Rates:** Uptown Mall ₱100/hr steep rate tier discourages commuter monopolization; automated rate progression penalizes overstays. |

---

## 12. Hardware Topography, Edge Deployment & Production Scaling

To transition this proof of concept into physical Megaworld township infrastructure across 1,000+ bays and 12 gate lanes, the following edge deployment architecture is recommended:

```
                            [ PHYSICAL TOWNSHIP INFRASTRUCTURE ]
                                              │
                 ┌────────────────────────────┴────────────────────────────┐
                 │                                                         │
       [ Gate Ingress / Egress ]                                 [ Multi-Level Decks ]
  12 Lanes: 4MP 60fps RTSP Cameras                          Overhead Multi-Bay CCTV Cameras
  Motorized Varifocal Lens (5–50mm)                         High-Angle PoE Ceiling Mounts
                 │                                                         │
                 ▼                                                         ▼
       [ Gate Edge AI Nodes ]                                    [ Deck Edge AI Nodes ]
   NVIDIA Jetson Orin Nano (8GB)                             NVIDIA Jetson Orin NX (16GB)
   TensorRT YOLOv8-Plate + RapidOCR                          TensorRT YOLOv8n + 5-Phase Space Engine
                 │                                                         │
                 └────────────────────────────┬────────────────────────────┘
                                              │ (MQTT over TLS / gRPC)
                                              ▼
                             [ Central Township Gateway Server ]
                                  x86 Linux (Docker Compose)
                               • Local Redis State Machine
                               • PostgreSQL Relational Store
                               • Streamlit Ops Portal (Tab 1–7)
                               • Outbound Signage & Mobile APIs
```

### 12.1 Edge Compute & Camera Specifications
- **Gate ALPR Nodes:** NVIDIA Jetson Orin Nano (8GB) running TensorRT FP16 optimized YOLOv8-Plate + Fast-ANPR OCR. Expected latency: $<25\text{ms}$ per vehicle.
- **Gate Camera Hardware:** 4MP ($2560 \times 1440$), 60 FPS, global shutter or low rolling shutter distortion, motorized varifocal lens ($5\text{--}50\text{mm}$), integrated 850nm infrared strobe illuminator.
- **Deck Surveillance Cameras:** 4MP wide-angle dome cameras mounted along center drive aisles, each monitoring 6 to 10 angled parking bays simultaneously.
- **Township Server:** Dual-redundant 1U rackmount servers hosting Redis state buffers, PostgreSQL persistence, and local Streamlit monitoring instances connected to township digital signage networks.

### 12.2 Data Privacy & Compliance (Philippine Data Privacy Act of 2012)
- **Edge RAM Image Processing:** Raw camera video frames captured at gate lanes and surveillance decks are processed entirely within volatile edge RAM and discarded immediately after plate extraction and IoA calculation.
- **Cryptographic License Plate Hashing:** In accordance with the Philippine Data Privacy Act of 2012 (RA 10173), long-term analytical logs can be salted and SHA-256 hashed (`sha256(plate + township_salt)`) to support longitudinal traffic forecasting while preserving motorist privacy.

---
*End of Project Bible. Maintained by the Megaworld Smart Parking Project Team.*
