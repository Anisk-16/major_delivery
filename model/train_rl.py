"""
train_rl.py
===========
Trains a PPO Actor-Critic agent on the DeliveryEnv using Stable-Baselines3.

Usage:
    python train_rl.py                        # default 100k steps
    python train_rl.py --timesteps 200000     # longer training
    python train_rl.py --timesteps 50000 --quick   # quick smoke-test

Outputs:
    models/ppo_delivery.zip   — trained model
    models/vecnorm.pkl        — VecNormalize statistics
    logs/                     — TensorBoard logs
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "preprocessing", "orders_clean.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ── lazy import env ────────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from delivery_env import DeliveryEnv


def make_env(df, seed=0, max_orders=20):
    def _init():
        env = DeliveryEnv(df, max_orders=max_orders, seed=seed)
        env = Monitor(env)
        return env
    return _init


def train(timesteps: int = 100_000, max_orders: int = 20, quick: bool = False):
    print(f"[train_rl] Loading dataset …")
    df = pd.read_csv(DATA_PATH)
    print(f"[train_rl] {len(df):,} clean orders loaded")

    # split train / eval
    train_df = df.sample(frac=0.9, random_state=42).reset_index(drop=True)
    eval_df  = df.drop(train_df.index).reset_index(drop=True)

    # vectorised training env (4 parallel)
    n_envs = 1 if quick else 4
    train_env = DummyVecEnv([make_env(train_df, seed=i, max_orders=max_orders)
                             for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0)

    # eval env (no reward normalisation for honest metrics)
    eval_env = DummyVecEnv([make_env(eval_df, seed=99, max_orders=max_orders)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    # ── PPO model ──────────────────────────────────────────────────────────────
    model = PPO(
        policy         = "MlpPolicy",
        env            = train_env,
        n_steps        = 512,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        learning_rate  = 3e-4,
        verbose        = 1,
        tensorboard_log= LOG_DIR,
        policy_kwargs  = dict(net_arch=[256, 256]),
    )

    # ── callbacks ──────────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = max(1000, timesteps // 20),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq = max(5000, timesteps // 10),
        save_path = MODEL_DIR,
        name_prefix = "ppo_delivery",
    )

    # ── train ──────────────────────────────────────────────────────────────────
    print(f"[train_rl] Starting PPO training for {timesteps:,} steps …")
    model.learn(total_timesteps=timesteps, callback=[eval_cb, ckpt_cb])

    # ── save ───────────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "ppo_delivery")
    vecnorm_path = os.path.join(MODEL_DIR, "vecnorm.pkl")
    model.save(model_path)
    train_env.save(vecnorm_path)
    print(f"[train_rl] ✅ Model saved → {model_path}.zip")
    print(f"[train_rl] ✅ VecNorm  saved → {vecnorm_path}")

    return model, train_env


def evaluate(model, env, n_episodes=5):
    print("\n[train_rl] Evaluating …")
    rewards, infos_all = [], []
    for ep in range(n_episodes):
        obs = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            if done[0]:
                infos_all.append(info[0])
        rewards.append(ep_reward)

    print(f"  Mean episode reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    for k in ["total_dist_km", "total_fuel_L", "late_count", "orders_served"]:
        vals = [i.get("episode", {}).get(k, i.get(k, 0)) for i in infos_all]
        print(f"  {k:20s}: {np.mean(vals):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",  type=int,  default=100_000)
    parser.add_argument("--max_orders", type=int,  default=20)
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    ts = 5_000 if args.quick else args.timesteps
    model, env = train(timesteps=ts, max_orders=args.max_orders, quick=args.quick)
    evaluate(model, env)
