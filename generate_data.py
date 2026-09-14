"""
generate_data.py
-----------------
Builds a fully synthetic (but internally consistent) dataset for the Smart
Parking POC: sites, zones, slots, weeks of historical occupancy, ticketing
records, plate reads with injected OCR noise, and a small holidays/events
calendar.

Everything here is fake data — but the SHAPE of the data (schema, noise
characteristics, event effects) mirrors what a real deployment would produce,
so the downstream matching/ML/UI code is exercising real logic.

Two mock townships (Uptown Bonifacio, Eastwood City) with zone types
(office, mall, residential), each with a genuinely different daily
occupancy curve — office fills mornings & drains evenings, mall peaks
evenings & weekends, residential stays high overnight.

Run directly to (re)build data/parking.db:
    python generate_data.py
"""

import os
import sqlite3
import random
import string
import math
from datetime import datetime, timedelta

import numpy as np

random.seed(42)
np.random.seed(42)

DB_PATH = "data/parking.db"

SITES = [
    {"name": "Uptown Bonifacio"},
    {"name": "Eastwood City"},
    {"name": "McKinley Hill"},
]

ZONES = [
    # Uptown Bonifacio (site_idx=0) — 6 Underground Basement Levels (B1 to B6)
    {"site_idx": 0, "level": "Basement 1 & 2 (B1-B2)",            "label": "Uptown Mall Retail Deck",             "zone_type": "mall",        "capacity": 48},
    {"site_idx": 0, "level": "Basement 3 & 4 (B3-B4)",            "label": "Alliance Global & Corporate Towers",  "zone_type": "office",      "capacity": 24},
    {"site_idx": 0, "level": "Basement 5 & 6 (B5-B6)",            "label": "One Uptown & Parksuites Residences",   "zone_type": "residential", "capacity": 16},
    # Eastwood City (site_idx=1) — Mall Basements (B1-B2), Citywalk Multi-Level, and Tower Podiums
    {"site_idx": 1, "level": "Basement 1 & 2 (B1-B2)",            "label": "Eastwood Mall Retail Deck",           "zone_type": "mall",        "capacity": 48},
    {"site_idx": 1, "level": "Citywalk Multi-Level (Levels 2-4)", "label": "Citywalk & CyberPark Office Deck",   "zone_type": "office",      "capacity": 24},
    {"site_idx": 1, "level": "Podium Levels 3 & 4 (P3-P4)",       "label": "Grand Eastwood Palazzo & Residences", "zone_type": "residential", "capacity": 16},
    # McKinley Hill (site_idx=2) — Venice Mall Basements (B1-B3) and Residential Podiums
    {"site_idx": 2, "level": "Basement 1 & 2 (B1-B2)",            "label": "Venice Grand Canal Mall Deck",        "zone_type": "mall",        "capacity": 48},
    {"site_idx": 2, "level": "Basement 2 & 3 (B2-B3)",            "label": "Commerce Plaza & Science Hub Deck",   "zone_type": "office",      "capacity": 24},
    {"site_idx": 2, "level": "Podium Levels 2 & 3 (P2-P3)",       "label": "Venice Luxury Residences & Viceroy",   "zone_type": "residential", "capacity": 16},
]

HISTORY_DAYS = 28          # weeks of synthetic historical data
INTERVAL_MINUTES = 15      # resolution of the historical occupancy series

CONFUSABLE_PAIRS = {
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "5": "S", "S": "5",
    "2": "Z", "Z": "2",
}


# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------

def random_plate():
    letters = "".join(random.choices(string.ascii_uppercase, k=3))
    digits = "".join(random.choices(string.digits, k=4))
    return f"{letters}{digits}"


def build_vehicle_pool(n_regulars=350, zipf_exponent=0.55, seed=42):
    """
    Builds a calibrated vehicle pool with regular returning parkers (Zipf weighted)
    and transient parkers to achieve an authentic 20-35% repeat-visitor cohort rate.
    """
    rng = random.Random(seed)
    plates_set = set()
    while len(plates_set) < n_regulars:
        letters = "".join(rng.choices(string.ascii_uppercase, k=3))
        digits = "".join(rng.choices(string.digits, k=4))
        plates_set.add(f"{letters}{digits}")
    reg_plates = list(plates_set)
    ranks = np.arange(1, n_regulars + 1)
    weights = 1.0 / np.power(ranks, zipf_exponent)
    weights = weights / weights.sum()
    rng.shuffle(reg_plates)
    return reg_plates, weights


class VehiclePoolDrawer:
    def __init__(self, n_regulars=350, repeat_prob=0.18, seed=42):
        self.reg_plates, self.reg_weights = build_vehicle_pool(n_regulars, seed=seed)
        self.repeat_prob = repeat_prob
        self.transient_pool = []
        self.rng = random.Random(seed + 7)

    def draw(self):
        if np.random.random() < self.repeat_prob:
            return str(np.random.choice(self.reg_plates, p=self.reg_weights))
        if self.transient_pool and np.random.random() < 0.12:
            return self.rng.choice(self.transient_pool)
        fresh = random_plate()
        if len(self.transient_pool) < 2500:
            self.transient_pool.append(fresh)
        return fresh


def corrupt_plate(plate, noise_level=0.25):
    """Simulate a low/variable-confidence OCR read of a real plate.
    Returns (read_text, per_char_confidence_list)."""
    chars = list(plate)
    confidences = []
    for i, ch in enumerate(chars):
        conf = float(np.clip(np.random.normal(0.85, 0.12), 0.25, 0.99))
        confidences.append(round(conf, 2))
        if random.random() < noise_level and conf < 0.75:
            # low-confidence character: sometimes swap for a confusable char
            if ch in CONFUSABLE_PAIRS and random.random() < 0.7:
                chars[i] = CONFUSABLE_PAIRS[ch]
            elif random.random() < 0.3:
                chars[i] = random.choice(string.ascii_uppercase + string.digits)
    return "".join(chars), confidences


def daily_occupancy_curve(hour_frac, is_weekend, zone_type="mall"):
    """Returns a base occupancy rate in [0,1] for a given fractional hour,
    differentiated by zone type.

    Zone types:
      - office:      fills ~8–9am, dips at lunch, drains after 6pm,
                     quiet on weekends.
      - mall:        quiet mornings, builds through the afternoon,
                     peaks in the evening, busier on weekends.
      - residential: high overnight, dips during work hours,
                     flat across weekdays/weekends.
    """
    if zone_type == "office":
        if not is_weekend:
            # Weekday office: morning ramp → lunch dip → afternoon → evening drain
            base = (
                0.05
                + 0.65 * math.exp(-((hour_frac - 10.0) ** 2) / (2 * 1.8 ** 2))   # morning fill
                + 0.55 * math.exp(-((hour_frac - 14.5) ** 2) / (2 * 2.0 ** 2))   # afternoon
                - 0.15 * math.exp(-((hour_frac - 12.5) ** 2) / (2 * 0.8 ** 2))   # lunch dip
            )
        else:
            # Weekend office: nearly empty
            base = (
                0.05
                + 0.12 * math.exp(-((hour_frac - 12.0) ** 2) / (2 * 3.0 ** 2))
            )

    elif zone_type == "residential":
        if not is_weekend:
            # Weekday residential: high overnight, dips during work hours
            base = (
                0.82
                - 0.40 * math.exp(-((hour_frac - 12.0) ** 2) / (2 * 3.5 ** 2))   # daytime dip
                + 0.10 * math.exp(-((hour_frac - 20.0) ** 2) / (2 * 2.0 ** 2))   # evening fill
            )
        else:
            # Weekend residential: stays high all day, slight midday dip
            base = (
                0.82
                - 0.18 * math.exp(-((hour_frac - 13.0) ** 2) / (2 * 3.0 ** 2))
            )

    else:  # "mall" (default / legacy)
        if not is_weekend:
            # Weekday mall: quiet morning, afternoon build, evening peak
            base = (
                0.08
                + 0.55 * math.exp(-((hour_frac - 12.5) ** 2) / (2 * 3.0 ** 2))   # lunch
                + 0.75 * math.exp(-((hour_frac - 18.5) ** 2) / (2 * 2.5 ** 2))   # evening peak
                + 0.30 * math.exp(-((hour_frac - 9.5) ** 2) / (2 * 1.5 ** 2))    # morning trickle
            )
        else:
            # Weekend mall: broader, busier midday through evening
            base = (
                0.10
                + 0.80 * math.exp(-((hour_frac - 15.0) ** 2) / (2 * 4.0 ** 2))
                + 0.25 * math.exp(-((hour_frac - 11.0) ** 2) / (2 * 2.0 ** 2))   # late-morning shoppers
            )

    return float(np.clip(base, 0.03, 0.97))


import ph_holidays

def build_calendar(start_date, num_days, sites):
    """Builds a real Philippine holidays calendar and synthetic township events."""
    holidays = []
    events = []

    # Populate all official Philippine holidays within the simulation date window (plus future buffer)
    for day_offset in range(-7, num_days + 60):
        cur_d = (start_date + timedelta(days=day_offset)).date()
        h_info = ph_holidays.get_ph_holiday_info(cur_d)
        if h_info:
            h_name, h_type = h_info
            holidays.append({
                "date": cur_d.isoformat(),
                "name": h_name,
                "type": h_type,
            })

    # Spread realistic township events across sites
    for i, site in enumerate(sites):
        event_day_offset = int(num_days * (0.35 + 0.2 * i))
        event_start = start_date + timedelta(days=event_day_offset, hours=18)
        events.append({
            "site": site["name"], "name": f"Megaworld Live Concert Series — {site['name']}",
            "starts_at": event_start.isoformat(),
            "ends_at": (event_start + timedelta(hours=4)).isoformat(),
            "impact": "high",
        })
        bazaar_day = int(num_days * (0.75 + 0.1 * i))
        bazaar_start = start_date + timedelta(days=bazaar_day, hours=11)
        events.append({
            "site": site["name"], "name": f"Township Weekend Fair & Bazaar — {site['name']}",
            "starts_at": bazaar_start.isoformat(),
            "ends_at": (bazaar_start + timedelta(hours=8)).isoformat(),
            "impact": "medium",
        })
    return holidays, events


def event_multiplier(ts, events, site_name=None):
    mult = 1.0
    for e in events:
        if site_name and e["site"] != site_name:
            continue
        s = datetime.fromisoformat(e["starts_at"])
        en = datetime.fromisoformat(e["ends_at"])
        if s <= ts <= en:
            mult *= 1.5 if e["impact"] == "high" else 1.25
    return mult


def holiday_multiplier(ts, holidays, zone_type="mall"):
    """Applies zone-differentiated Philippine holiday mobility multipliers."""
    return ph_holidays.get_holiday_occupancy_factor(ts, zone_type)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def main():
    if os.path.dirname(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS sites;
    DROP TABLE IF EXISTS zones;
    DROP TABLE IF EXISTS slots;
    DROP TABLE IF EXISTS occupancy_history;
    DROP TABLE IF EXISTS ticketing_records;
    DROP TABLE IF EXISTS plate_reads;
    DROP TABLE IF EXISTS holidays;
    DROP TABLE IF EXISTS events;
    DROP TABLE IF EXISTS current_state;

    CREATE TABLE sites (site_id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE zones (
        zone_id INTEGER PRIMARY KEY, site_id INTEGER, level TEXT, label TEXT,
        zone_type TEXT, capacity INTEGER
    );
    CREATE TABLE slots (
        slot_id INTEGER PRIMARY KEY, zone_id INTEGER, slot_code TEXT
    );
    CREATE TABLE occupancy_history (
        ts TEXT, zone_id INTEGER, occupied_count INTEGER, capacity INTEGER,
        is_holiday INTEGER, is_event INTEGER, occupancy_rate REAL
    );
    CREATE TABLE ticketing_records (
        ticket_id TEXT PRIMARY KEY, plate TEXT, entry_time TEXT,
        payment_settled_at TEXT, slot_id INTEGER
    );
    CREATE TABLE plate_reads (
        read_id INTEGER PRIMARY KEY AUTOINCREMENT, slot_id INTEGER,
        raw_ocr_text TEXT, char_confidences TEXT, true_plate TEXT
    );
    CREATE TABLE holidays (date TEXT, name TEXT);
    CREATE TABLE events (site TEXT, name TEXT, starts_at TEXT, ends_at TEXT, impact TEXT);
    CREATE TABLE current_state (
        slot_id INTEGER PRIMARY KEY, status TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS scraped_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_platform TEXT,
        township TEXT,
        mall_deck TEXT,
        title TEXT,
        event_type TEXT,
        description TEXT,
        start_date TEXT,
        end_date TEXT,
        traffic_impact_factor REAL,
        source_url TEXT,
        image_url TEXT,
        created_at TEXT,
        is_active INTEGER DEFAULT 1
    );
    """)

    # Sites
    for si, site in enumerate(SITES, start=1):
        cur.execute("INSERT INTO sites VALUES (?, ?)", (si, site["name"]))

    # Zones / slots
    zone_rows = []
    slot_rows = []
    slot_id = 1
    for zi, z in enumerate(ZONES, start=1):
        site_id = z["site_idx"] + 1  # 1-indexed
        zone_rows.append((zi, site_id, z["level"], z["label"], z["zone_type"], z["capacity"]))
        prefix = z["zone_type"][0].upper()  # M for Mall, O for Office, R for Residential
        for s in range(1, z["capacity"] + 1):
            code = f"{prefix}-{s:03d}"
            slot_rows.append((slot_id, zi, code))
            slot_id += 1
    cur.executemany("INSERT INTO zones VALUES (?,?,?,?,?,?)", zone_rows)
    cur.executemany("INSERT INTO slots VALUES (?,?,?)", slot_rows)

    # Calendar (aligned to start of day through end of today for complete 24h operational profiles)
    start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    holidays, events = build_calendar(start_date, HISTORY_DAYS + 1, SITES)
    cur.executemany("INSERT INTO holidays VALUES (?,?)", [(h["date"], h["name"]) for h in holidays])
    cur.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?)",
        [(e["site"], e["name"], e["starts_at"], e["ends_at"], e["impact"]) for e in events],
    )

    # Historical occupancy per zone, at INTERVAL_MINUTES resolution (full 24h for all days)
    hist_rows = []
    n_steps = (HISTORY_DAYS + 1) * 24 * (60 // INTERVAL_MINUTES)
    for step in range(n_steps):
        ts = start_date + timedelta(minutes=step * INTERVAL_MINUTES)
        hour_frac = ts.hour + ts.minute / 60.0
        is_weekend = ts.weekday() >= 5
        for zi, z in enumerate(ZONES, start=1):
            site_name = SITES[z["site_idx"]]["name"]
            base_rate = daily_occupancy_curve(hour_frac, is_weekend, z["zone_type"])
            mult = event_multiplier(ts, events, site_name) * holiday_multiplier(ts, holidays, z["zone_type"])
            rate = float(np.clip(base_rate * mult + np.random.normal(0, 0.04), 0.0, 1.0))
            occ_count = int(round(rate * z["capacity"]))
            is_hol = 1 if ph_holidays.is_ph_holiday(ts) else 0
            is_evt = 1 if event_multiplier(ts, events, site_name) > 1.0 else 0
            hist_rows.append((ts.isoformat(), zi, occ_count, z["capacity"], is_hol, is_evt, rate))
    cur.executemany(
        "INSERT INTO occupancy_history VALUES (?,?,?,?,?,?,?)", hist_rows
    )

    # "Right now" simulated live state
    now = datetime.now().replace(hour=18, minute=30, second=0, microsecond=0)
    ticket_rows = []
    plate_read_rows = []
    state_rows = []
    ticket_counter = 1

    hour_frac_now = now.hour + now.minute / 60.0
    is_weekend_now = now.weekday() >= 5

    # Pre-build vehicle pool drawer for realistic repeat visitor dynamics
    pool_drawer = VehiclePoolDrawer(n_regulars=350, repeat_prob=0.22, seed=42)

    for zi, z in enumerate(ZONES, start=1):
        site_name = SITES[z["site_idx"]]["name"]
        live_mult = event_multiplier(now, events, site_name) * holiday_multiplier(now, holidays, z["zone_type"])
        base_rate = float(np.clip(daily_occupancy_curve(hour_frac_now, is_weekend_now, z["zone_type"]) * live_mult, 0.0, 1.0))
        n_occupied = int(round(base_rate * z["capacity"]))
        zone_slot_ids = [row[0] for row in slot_rows if row[1] == zi]
        occupied_slots = random.sample(zone_slot_ids, min(n_occupied, len(zone_slot_ids)))

        for sid in zone_slot_ids:
            if sid not in occupied_slots:
                state_rows.append((sid, "free", now.isoformat()))
                continue

            plate = pool_drawer.draw()
            # ~45% of currently-occupied slots have a settled payment on file
            has_paid_ticket = random.random() < 0.45
            ticket_id = f"TCK{ticket_counter:05d}"
            ticket_counter += 1

            # Archetype-specific dwell time distribution
            ztype = z["zone_type"]
            if ztype == "mall":
                # Lognormal centered around ~2.5 hours (150 mins)
                dwell_mins = int(np.clip(np.random.lognormal(mean=np.log(140), sigma=0.55), 20, 420))
            elif ztype == "office":
                # Bimodal: 75% full workday (480-540 mins) + 25% meetings/visitors (60-150 mins)
                dwell_mins = int(random.gauss(510, 35)) if random.random() < 0.75 else int(random.gauss(100, 25))
            else:
                # Residential: 70% overnight/long (600-960 mins) + 30% day guests (120-240 mins)
                dwell_mins = int(random.gauss(780, 70)) if random.random() < 0.70 else int(random.gauss(180, 40))

            entry_time = now - timedelta(minutes=dwell_mins)
            settled_at = (entry_time + timedelta(minutes=int(dwell_mins * 0.9))).isoformat() if has_paid_ticket else None

            ticket_rows.append((ticket_id, plate, entry_time.isoformat(), settled_at, sid))

            # Always generate an OCR read for the slot (whether or not it'll match)
            read_text, confidences = corrupt_plate(plate)
            plate_read_rows.append((sid, read_text, str(confidences), plate))

            status = "occupied_unpaid" if not has_paid_ticket else "occupied_pending_match"
            state_rows.append((sid, status, now.isoformat()))

    # Generate historical completed ticket sessions across the full 28-day window
    # This enables realistic repeat-visitor recognition and empirical dwell distributions
    for past_day in range(1, HISTORY_DAYS + 1):
        day_date = now - timedelta(days=past_day)
        is_wknd = day_date.weekday() >= 5
        for zi, z in enumerate(ZONES, start=1):
            ztype = z["zone_type"]
            # Sessions proportional to zone capacity
            n_sessions = int(z["capacity"] * (1.8 if ztype == "mall" and is_wknd else (1.4 if ztype != "residential" else 0.8)))
            zone_slot_ids = [row[0] for row in slot_rows if row[1] == zi]
            
            for _ in range(n_sessions):
                plate = pool_drawer.draw()
                ticket_id = f"TCK{ticket_counter:05d}"
                ticket_counter += 1
                sid = random.choice(zone_slot_ids)

                # Peak-weighted arrival hour
                if ztype == "office":
                    arr_hour = int(random.gauss(9, 1.5))
                elif ztype == "mall":
                    arr_hour = int(random.gauss(16, 3.0))
                else:
                    arr_hour = int(random.gauss(19, 2.5))
                arr_hour = max(6, min(22, arr_hour))
                arr_min = random.randint(0, 59)
                t_entry = day_date.replace(hour=arr_hour, minute=arr_min, second=0)

                if ztype == "mall":
                    d_mins = int(np.clip(np.random.lognormal(mean=np.log(145), sigma=0.55), 20, 360))
                elif ztype == "office":
                    d_mins = int(random.gauss(500, 40)) if random.random() < 0.75 else int(random.gauss(95, 20))
                else:
                    d_mins = int(random.gauss(760, 60)) if random.random() < 0.70 else int(random.gauss(170, 35))

                t_settle = t_entry + timedelta(minutes=d_mins)
                ticket_rows.append((ticket_id, plate, t_entry.isoformat(), t_settle.isoformat(), sid))

    cur.executemany("INSERT INTO ticketing_records VALUES (?,?,?,?,?)", ticket_rows)
    cur.executemany(
        "INSERT INTO plate_reads (slot_id, raw_ocr_text, char_confidences, true_plate) VALUES (?,?,?,?)",
        plate_read_rows,
    )
    cur.executemany("INSERT INTO current_state VALUES (?,?,?)", state_rows)

    conn.commit()
    conn.close()

    n_sites = len(SITES)
    n_zones = len(ZONES)
    n_slots = len(slot_rows)
    print(f"Built {DB_PATH}")
    print(f"  Sites: {n_sites} | Zones: {n_zones} | Slots: {n_slots}")
    print(f"  Zone types: office, mall, residential")
    print(f"  Historical rows: {len(hist_rows)} ({HISTORY_DAYS} days @ {INTERVAL_MINUTES}min)")
    print(f"  Total ticket records (live + 7d past): {len(ticket_rows)} | Plate reads: {len(plate_read_rows)}")
    print(f"  Holidays: {len(holidays)} | Events: {len(events)}")


if __name__ == "__main__":
    main()