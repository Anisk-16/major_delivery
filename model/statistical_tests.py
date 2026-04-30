"""
statistical_tests.py
====================
Runs statistical significance tests comparing:
  1. PPO-only (no penalty)  — baseline
  2. PPO + unserved penalty — intermediate
  3. Hybrid (PPO + OR-Tools) — proposed system

Tests performed:
  - Mann-Whitney U (non-parametric, no normality assumption)
  - Paired Wilcoxon signed-rank test
  - Cohen's d effect size
  - 95% Confidence Intervals (bootstrap)
  - Descriptive statistics

Usage:
    python statistical_tests.py
"""

import os
import sys
import math
import json
import random
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon

# ── path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_DIR = os.path.join(BASE_DIR, "..", "preprocessing")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PREP_DIR)

N_EPISODES   = 30      # number of evaluation episodes per configuration
BATCH_SIZE   = 20      # orders per episode
SEEDS        = list(range(42, 42 + N_EPISODES))
RESULTS_FILE = os.path.join(BASE_DIR, "statistical_results.json")


# ── helpers ────────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    """Cohen's d effect size between two samples."""
    n1, n2 = len(a), len(b)
    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = math.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
        / (n1 + n2 - 2)
    )
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def bootstrap_ci(data, n_boot=2000, ci=0.95):
    """Bootstrap 95% confidence interval for the mean."""
    data = np.array(data)
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return round(lo, 4), round(hi, 4)


def interpret_d(d):
    d = abs(d)
    if d < 0.2:   return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else:         return "large"


def interpret_p(p):
    if p < 0.001: return "p < 0.001 (***)"
    elif p < 0.01: return "p < 0.01 (**)"
    elif p < 0.05: return "p < 0.05 (*)"
    else: return f"p = {p:.4f} (not significant)"


# ── Configuration A: PPO-only evaluation ──────────────────────────────────────
def run_rl_episodes(df, model, vecnorm_path, n_episodes, batch_size, seeds):
    """Run n_episodes of the PPO policy and collect per-episode metrics."""
    from delivery_env import DeliveryEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    results = []
    for i, seed in enumerate(seeds[:n_episodes]):
        random.seed(seed)
        np.random.seed(seed)
        sample = df.sample(n=min(batch_size, len(df)), random_state=seed)

        try:
            env = DummyVecEnv([lambda s=seed: DeliveryEnv(
                sample.reset_index(drop=True), max_orders=batch_size, seed=s)])
            if vecnorm_path and os.path.exists(vecnorm_path):
                env = VecNormalize.load(vecnorm_path, env)
                env.training = False
                env.norm_reward = False

            obs = env.reset()
            ep_reward = 0.0
            served = 0
            late = 0

            for _ in range(batch_size):
                remaining = list(env.envs[0].remaining_idx)
                if not remaining:
                    break
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward[0])
                served += 1
                if bool(done[0]):
                    late = int(info[0].get("late_count", 0))
                    break

            on_time_pct = 100.0 * max(0, served - late) / max(served, 1)
            results.append({
                "episode": i,
                "seed": seed,
                "orders_served": served,
                "on_time_pct": round(on_time_pct, 1),
                "reward": round(ep_reward, 3),
                "late_count": late,
            })
            env.close()
            print(f"  Episode {i+1:2d}: served={served}, on_time={on_time_pct:.1f}%, reward={ep_reward:.1f}")

        except Exception as e:
            print(f"  Episode {i+1} failed: {e}")

    return results


# ── Configuration B: OR-Tools cold-start (no RL warm-start) ───────────────────
def run_ortools_episodes(df, depot_lat, depot_lon, n_episodes, batch_size, seeds):
    """Run OR-Tools solver on random batches and collect metrics."""
    from ortools_solver import solve_vrp

    results = []
    for i, seed in enumerate(seeds[:n_episodes]):
        sample = df.sample(n=min(batch_size, len(df)), random_state=seed)
        orders = sample.to_dict(orient="records")

        try:
            result = solve_vrp(
                orders=orders,
                depot_lat=depot_lat,
                depot_lon=depot_lon,
                n_vehicles=3,
                capacity=10,
                time_limit=10,
                warm_start=None,
                alpha=0.50, beta=0.20, gamma=0.20, delta=0.10,
            )
            results.append({
                "episode": i,
                "seed": seed,
                "solve_time_s": round(result["solve_time_s"], 4),
                "total_dist_km": round(result.get("total_dist_km", 0), 3),
                "total_fuel_L":  round(result.get("total_fuel_L", 0), 3),
                "status": result.get("status", ""),
            })
            print(f"  Episode {i+1:2d}: dist={result.get('total_dist_km',0):.2f} km, "
                  f"time={result['solve_time_s']:.3f}s, {result.get('status','')}")
        except Exception as e:
            print(f"  Episode {i+1} failed: {e}")

    return results


# ── Statistical comparison ─────────────────────────────────────────────────────
def compare(name_a, values_a, name_b, values_b, metric, higher_is_better=True):
    """Run full statistical comparison between two sample arrays."""
    a, b = np.array(values_a, dtype=float), np.array(values_b, dtype=float)

    # Mann-Whitney U (non-parametric)
    alt = "greater" if higher_is_better else "less"
    u_stat, p_mw = mannwhitneyu(a, b, alternative=alt)

    # Wilcoxon (paired, same n)
    p_wx = None
    if len(a) == len(b):
        try:
            _, p_wx = wilcoxon(a, b)
        except Exception:
            pass

    d = cohens_d(a, b)
    ci_a = bootstrap_ci(a)
    ci_b = bootstrap_ci(b)

    return {
        "metric": metric,
        "comparison": f"{name_a} vs {name_b}",
        f"mean_{name_a}": round(float(np.mean(a)), 4),
        f"std_{name_a}":  round(float(np.std(a, ddof=1)), 4),
        f"ci95_{name_a}": ci_a,
        f"mean_{name_b}": round(float(np.mean(b)), 4),
        f"std_{name_b}":  round(float(np.std(b, ddof=1)), 4),
        f"ci95_{name_b}": ci_b,
        "mann_whitney_U": round(float(u_stat), 2),
        "p_value_mw": round(p_mw, 6),
        "p_wilcoxon": round(p_wx, 6) if p_wx is not None else None,
        "cohens_d": round(d, 4),
        "effect_size": interpret_d(d),
        "significance": interpret_p(p_mw),
        "favors": name_a if (d > 0) == higher_is_better else name_b,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Statistical Significance Testing — Delivery Route Optimizer")
    print("=" * 65)

    # Load dataset
    data_path = os.path.join(PREP_DIR, "orders_clean.csv")
    df = pd.read_csv(data_path)
    depot_lat = float(df["depot_lat"].iloc[0])
    depot_lon = float(df["depot_lon"].iloc[0])
    print(f"\nDataset: {len(df):,} orders  |  depot=({depot_lat:.4f}, {depot_lon:.4f})")

    # Load model
    model_dir = os.path.join(BASE_DIR, "models")
    best_path  = os.path.join(model_dir, "best_model.zip")
    final_path = os.path.join(model_dir, "ppo_delivery.zip")
    vnorm_path = os.path.join(model_dir, "vecnorm.pkl")
    model_path = best_path if os.path.exists(best_path) else final_path

    if not os.path.exists(model_path):
        print("ERROR: No trained model found. Train first with train_rl.py")
        return

    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    print(f"Model loaded: {model_path}\n")

    # ── Run A: RL policy (warm-start generator) ───────────────────────────────
    print(f"[1/3] Running RL policy — {N_EPISODES} episodes...")
    rl_results = run_rl_episodes(df, model, vnorm_path, N_EPISODES, BATCH_SIZE, SEEDS)

    # ── Run B: OR-Tools cold start ─────────────────────────────────────────────
    print(f"\n[2/3] Running OR-Tools cold-start — {N_EPISODES} episodes...")
    or_results = run_ortools_episodes(df, depot_lat, depot_lon, N_EPISODES, BATCH_SIZE, SEEDS)

    # ── Run C: Hybrid (use HybridRouter) ──────────────────────────────────────
    print(f"\n[3/3] Running Hybrid (RL warm-start + OR-Tools) — {N_EPISODES} episodes...")
    from hybrid_integration import HybridRouter

    hybrid_results = []
    for i, seed in enumerate(SEEDS[:N_EPISODES]):
        sample = df.sample(n=min(BATCH_SIZE, len(df)), random_state=seed)
        orders = sample.to_dict(orient="records")
        try:
            router = HybridRouter(
                depot_lat=depot_lat, depot_lon=depot_lon,
                n_vehicles=3, capacity=10,
                rl_model=model,
                vecnorm=vnorm_path if os.path.exists(vnorm_path) else None,
            )
            result = router.optimize(orders)
            hybrid_results.append({
                "episode": i,
                "seed": seed,
                "solve_time_s":  round(result.get("solve_time_s", 0), 4),
                "total_dist_km": round(result.get("total_dist_km", 0), 3),
                "total_fuel_L":  round(result.get("total_fuel_L", 0), 3),
                "on_time_pct":   result.get("on_time_pct", 0),
                "orders_served": result.get("orders_served", 0),
            })
            print(f"  Episode {i+1:2d}: dist={result.get('total_dist_km',0):.2f} km, "
                  f"time={result.get('solve_time_s',0):.3f}s, "
                  f"on_time={result.get('on_time_pct',0):.1f}%")
        except Exception as e:
            print(f"  Episode {i+1} failed: {e}")

    # ── Statistical comparisons ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STATISTICAL RESULTS")
    print("=" * 65)

    comparisons = []

    # Extract arrays
    rl_served    = [r["orders_served"] for r in rl_results]
    rl_ontime    = [r["on_time_pct"]   for r in rl_results]
    rl_reward    = [r["reward"]        for r in rl_results]

    hy_served    = [r["orders_served"] for r in hybrid_results]
    hy_ontime    = [r["on_time_pct"]   for r in hybrid_results]
    hy_dist      = [r["total_dist_km"] for r in hybrid_results]
    hy_fuel      = [r["total_fuel_L"]  for r in hybrid_results]
    hy_time      = [r["solve_time_s"]  for r in hybrid_results]

    or_dist      = [r["total_dist_km"] for r in or_results]
    or_fuel      = [r["total_fuel_L"]  for r in or_results]
    or_time      = [r["solve_time_s"]  for r in or_results]

    # Test 1: Hybrid vs OR-Tools — distance
    if hy_dist and or_dist:
        c = compare("Hybrid", hy_dist, "OR_Cold", or_dist, "total_dist_km", higher_is_better=False)
        comparisons.append(c)
        print(f"\n[Test 1] Distance — Hybrid vs OR-Tools cold-start")
        print(f"  Hybrid:   {c['mean_Hybrid']:.3f} ± {c['std_Hybrid']:.3f} km  CI95={c['ci95_Hybrid']}")
        print(f"  OR-Cold:  {c['mean_OR_Cold']:.3f} ± {c['std_OR_Cold']:.3f} km  CI95={c['ci95_OR_Cold']}")
        print(f"  {c['significance']}  |  Cohen's d={c['cohens_d']} ({c['effect_size']})")

    # Test 2: Hybrid vs OR-Tools — solve time
    if hy_time and or_time:
        c = compare("Hybrid", hy_time, "OR_Cold", or_time, "solve_time_s", higher_is_better=False)
        comparisons.append(c)
        print(f"\n[Test 2] Solve Time — Hybrid vs OR-Tools cold-start")
        print(f"  Hybrid:   {c['mean_Hybrid']:.4f} ± {c['std_Hybrid']:.4f} s")
        print(f"  OR-Cold:  {c['mean_OR_Cold']:.4f} ± {c['std_OR_Cold']:.4f} s")
        print(f"  {c['significance']}  |  Cohen's d={c['cohens_d']} ({c['effect_size']})")

    # Test 3: RL policy — orders served (consistency)
    if rl_served:
        print(f"\n[Test 3] RL Policy — Orders Served over {len(rl_served)} episodes")
        print(f"  Mean:   {np.mean(rl_served):.2f} ± {np.std(rl_served, ddof=1):.2f}")
        print(f"  CI95:   {bootstrap_ci(rl_served)}")
        print(f"  Min/Max: {min(rl_served)} / {max(rl_served)}")

    # Test 4: RL — on-time rate
    if rl_ontime:
        print(f"\n[Test 4] RL Policy — On-Time Rate over {len(rl_ontime)} episodes")
        print(f"  Mean:   {np.mean(rl_ontime):.2f}% ± {np.std(rl_ontime, ddof=1):.2f}%")
        print(f"  CI95:   {bootstrap_ci(rl_ontime)}")

    # Test 5: Hybrid — fuel savings
    if hy_fuel and or_fuel:
        c = compare("Hybrid", hy_fuel, "OR_Cold", or_fuel, "total_fuel_L", higher_is_better=False)
        comparisons.append(c)
        print(f"\n[Test 5] Fuel — Hybrid vs OR-Tools cold-start")
        print(f"  Hybrid:   {c['mean_Hybrid']:.3f} ± {c['std_Hybrid']:.3f} L")
        print(f"  OR-Cold:  {c['mean_OR_Cold']:.3f} ± {c['std_OR_Cold']:.3f} L")
        print(f"  {c['significance']}  |  Cohen's d={c['cohens_d']} ({c['effect_size']})")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "config": {"n_episodes": N_EPISODES, "batch_size": BATCH_SIZE},
        "rl_results":     rl_results,
        "or_results":     or_results,
        "hybrid_results": hybrid_results,
        "comparisons":    comparisons,
        "summary": {
            "rl_orders_served_mean": round(np.mean(rl_served), 3)   if rl_served else None,
            "rl_orders_served_std":  round(np.std(rl_served, ddof=1), 3) if rl_served else None,
            "rl_orders_served_ci95": bootstrap_ci(rl_served)        if rl_served else None,
            "rl_on_time_mean":       round(np.mean(rl_ontime), 3)   if rl_ontime else None,
            "rl_on_time_ci95":       bootstrap_ci(rl_ontime)        if rl_ontime else None,
            "hybrid_dist_mean":      round(np.mean(hy_dist), 3)     if hy_dist else None,
            "hybrid_dist_ci95":      bootstrap_ci(hy_dist)          if hy_dist else None,
            "or_cold_dist_mean":     round(np.mean(or_dist), 3)     if or_dist else None,
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*65}")
    print(f"Results saved → {RESULTS_FILE}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
