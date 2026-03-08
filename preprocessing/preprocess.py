

import pandas as pd
import numpy as np
import math
import os

RAW_PATH  = os.path.join(os.path.dirname(__file__), "orders.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "orders_clean.csv")

TRAFFIC_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Very_High"}
WEATHER_MAP = {0: "Clear", 1: "Cloudy", 2: "Light_Rain",
               3: "Heavy_Rain", 4: "Fog", 5: "Storm", 6: "Snow"}

# ── Haversine (for unit-square coords we treat 1 unit ≈ 111 km) ──────────────
def haversine_unit(lat1, lon1, lat2, lon2):
    """Approximate distance (km) for normalised [0,1] coordinates."""
    R = 111.0  # km per degree (approx)
    dlat = (lat2 - lat1) * R
    dlon = (lon2 - lon1) * R
    return math.sqrt(dlat**2 + dlon**2)


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Build an N×N distance matrix for an array of (lat, lon) pairs
    where coords are in normalised [0,1] space.
    """
    n = len(coords)
    mat = np.zeros((n, n), dtype=np.float32)
    R = 111.0
    for i in range(n):
        for j in range(i + 1, n):
            dlat = (coords[j, 0] - coords[i, 0]) * R
            dlon = (coords[j, 1] - coords[i, 1]) * R
            d = math.sqrt(dlat**2 + dlon**2)
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ── IQR-based winsorisation ───────────────────────────────────────────────────
def iqr_cap(series: pd.Series) -> pd.Series:
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return series.clip(lower=lo, upper=hi)


def preprocess(raw_path=RAW_PATH, out_path=OUT_PATH, verbose=True) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    n0 = len(df)
    if verbose:
        print(f"[1] Loaded  {n0:,} rows")

    # ── 1. Drop exact duplicates ───────────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    if verbose:
        print(f"[2] After dedup: {len(df):,} rows  (removed {n0-len(df)})")

    # ── 2. Fix timing inversions (pickup_time < order_time) ───────────────────
    bad_mask = df["pickup_time_min"] < df["order_time_min"]
    # Swap the two columns for inverted rows
    df.loc[bad_mask, ["order_time_min", "pickup_time_min"]] = (
        df.loc[bad_mask, ["pickup_time_min", "order_time_min"]].values
    )
    if verbose:
        print(f"[3] Fixed {bad_mask.sum()} timing inversions (swapped order/pickup times)")

    # ── 3. Remove extreme wait-time outliers (>120 min or <0 after fix) ───────
    df["wait_time_min"] = df["pickup_time_min"] - df["order_time_min"]
    before = len(df)
    df = df[(df["wait_time_min"] >= 0) & (df["wait_time_min"] <= 120)].copy()
    if verbose:
        print(f"[4] Removed {before-len(df)} extreme wait-time rows  →  {len(df):,} remain")

    # ── 4. Winsorise distance_km and Time_taken ────────────────────────────────
    df["distance_km"]       = iqr_cap(df["distance_km"])
    df["Time_taken (min)"]  = iqr_cap(df["Time_taken (min)"])
    if verbose:
        print(f"[5] Winsorised distance_km and Time_taken (min)")

    # ── 5. Re-derive est_time from distance (speed ≈ 25 km/h baseline) ─────────
    SPEED_BASELINE = 25.0  # km/h
    TRAFFIC_FACTOR = {0: 1.0, 1: 1.2, 2: 1.5, 3: 1.8}
    df["speed_factor"] = df["Road_traffic_density"].map(TRAFFIC_FACTOR)
    df["est_time_derived"] = (df["distance_km"] / SPEED_BASELINE) * 60 * df["speed_factor"]
    if verbose:
        print(f"[6] Added est_time_derived (traffic-adjusted ETA in minutes)")

    # ── 6. Label columns ───────────────────────────────────────────────────────
    df["traffic_label"] = df["Road_traffic_density"].map(TRAFFIC_MAP)
    df["weather_label"] = df["Weather_conditions"].map(WEATHER_MAP)

    # ── 7. Fuel estimate (L) — simple linear model: 0.12 L/km ────────────────
    df["fuel_L"] = (df["distance_km"] * 0.12).round(3)

    # ── 8. Virtual depot (centroid of all pickups) ────────────────────────────
    depot_lat = df["pickup_lat"].mean()
    depot_lon = df["pickup_lon"].mean()
    df["depot_lat"] = depot_lat
    df["depot_lon"] = depot_lon

    # ── 9. Distance from depot to pickup ─────────────────────────────────────
    R = 111.0
    df["depot_to_pickup_km"] = np.sqrt(
        ((df["pickup_lat"] - depot_lat) * R) ** 2 +
        ((df["pickup_lon"] - depot_lon) * R) ** 2
    )

    # ── 10. Normalised time-of-day feature (0–1) ──────────────────────────────
    MAX_MINS = 1440  # minutes in a day
    df["order_time_norm"]  = df["order_time_min"]  / MAX_MINS
    df["pickup_time_norm"] = df["pickup_time_min"] / MAX_MINS

    # ── 11. Order ID ──────────────────────────────────────────────────────────
    df.insert(0, "order_id", range(1, len(df) + 1))

    # ── 12. Final column cleanup ───────────────────────────────────────────────
    df.drop(columns=["speed_factor"], inplace=True)
    df.rename(columns={"Time_taken (min)": "time_taken_min"}, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\n✅ Saved cleaned dataset → {out_path}")
        print(f"   Final shape : {df.shape}")
        print(f"   Depot       : ({depot_lat:.4f}, {depot_lon:.4f})")
        print(f"\nColumn summary:\n{df.dtypes}")
    return df


if __name__ == "__main__":
    preprocess()
