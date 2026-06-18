# -*- coding: utf-8 -*-
r"""
Regina_02_build_ARMA100_model.py

作用：
1. 读取第一份代码生成的 Regina_cleaned/Regina_hourly_cleaned_2001_2003.csv；
2. 计算 8760 个小时位置的 mu_t、sigma_t，并构造标准化序列 y_t；
3. 建立 Regina ARMA(4,3) 模型：
   - 默认使用论文给出的 Regina ARMA(4,3) 参数，用于更贴近 Fig.12 原文基准；
   - 同时可选用 statsmodels 对 2001-2003 数据重新拟合 ARMA(4,3)；
4. 模拟多年风速，生成 Regina_ARMA100_wind_model.csv，作为 Fig.12 的 100-step ARMA 基准模型。

运行：
python Regina_02_build_ARMA100_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_paths import *

# =========================================================
# 1. 路径与参数
# =========================================================

CLEAN_DIR = REGINA_CLEANED_DIR

RESULT_DIR = REGINA_ARMA_RESULT_DIR

CLEANED_PATH = (
    REGINA_CLEANED_DIR /
    "Regina_hourly_cleaned_2001_2003.csv"
)

ACTUAL_SUMMARY_PATH = (
    REGINA_ARMA_RESULT_DIR /
    "Regina_actual_mu_sigma_summary.csv"
)
# Fig.12 原文给 Regina 使用 ARMA(4,3)。
AR_ORDER = 4
MA_ORDER = 3

# 选择用于模拟 100-step ARMA 基准模型的参数来源：
# "paper"：使用论文给出的 Regina ARMA(4,3) 参数；
# "refit"：使用当前下载的 2001-2003 数据重新拟合参数。
# 严格复现 Fig.12 的逻辑，建议先用 "paper"。
ARMA_PARAMETER_MODE = "paper"

# 是否额外拟合一套 ARMA(4,3) 参数作为参考输出。
TRY_REFIT_FOR_REFERENCE = True

# 模拟年数。数值越大，概率分布越稳定，但运行时间越长。
SIMULATION_YEARS = 1000
RANDOM_SEED = 20260617
BURN_IN = 1000

# 100-step 风速模型设置
NB_STEPS = 100
Z_SPAN = 10.0


# =========================================================
# 2. Regina 论文 ARMA(4,3) 参数
# =========================================================
# 论文公式：
# y_t = 0.9336 y_{t-1} + 0.4506 y_{t-2} - 0.5545 y_{t-3} + 0.1110 y_{t-4}
#       + alpha_t - 0.2033 alpha_{t-1} - 0.4684 alpha_{t-2} + 0.2301 alpha_{t-3}
# alpha_t ∈ NID(0, 0.409423^2)
PAPER_AR = np.array([0.9336, 0.4506, -0.5545, 0.1110], dtype=float)
PAPER_MA_ADDITIVE = np.array([-0.2033, -0.4684, 0.2301], dtype=float)
PAPER_ALPHA_STD = 0.409423


# =========================================================
# 3. 数据读取与标准化
# =========================================================
def check_required_files():
    missing = []
    for p in [CLEANED_PATH, ACTUAL_SUMMARY_PATH]:
        if not p.exists():
            missing.append(p)
    if missing:
        print("\n========== 缺失文件 ==========")
        for p in missing:
            print(p)
        raise FileNotFoundError("请先运行 Regina_01_clean_hourly_data.py 完成数据清洗。")


def load_cleaned_data() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    required = ["datetime", "year", "month", "day", "hour", "wind_speed_kmh", "hour_index_8760"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{CLEANED_PATH.name} 缺少列：{missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["wind_speed_kmh"] = pd.to_numeric(df["wind_speed_kmh"], errors="coerce")
    df["hour_index_8760"] = pd.to_numeric(df["hour_index_8760"], errors="coerce").astype(int)

    df = df.dropna(subset=["datetime", "wind_speed_kmh", "hour_index_8760"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def read_actual_mu_sigma():
    summary = pd.read_csv(ACTUAL_SUMMARY_PATH, encoding="utf-8-sig")
    row = summary.iloc[0]
    return float(row["mean_wind_speed_kmh"]), float(row["std_wind_speed_kmh"])


def build_hourly_mu_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """按 0-8759 小时位置计算 mu_t、sigma_t。"""
    hourly = (
        df.groupby("hour_index_8760", as_index=False)
        .agg(
            mu_t_kmh=("wind_speed_kmh", "mean"),
            sigma_t_kmh=("wind_speed_kmh", "std"),
            sample_count=("wind_speed_kmh", "count"),
        )
        .sort_values("hour_index_8760")
        .reset_index(drop=True)
    )

    # 保证 8760 个小时位置齐全
    full = pd.DataFrame({"hour_index_8760": np.arange(8760)})
    hourly = full.merge(hourly, on="hour_index_8760", how="left")

    global_sigma = df["wind_speed_kmh"].std(ddof=1)
    hourly["mu_t_kmh"] = hourly["mu_t_kmh"].interpolate(limit_direction="both")
    hourly["sigma_t_kmh"] = hourly["sigma_t_kmh"].replace(0, np.nan)
    hourly["sigma_t_kmh"] = hourly["sigma_t_kmh"].interpolate(limit_direction="both")
    hourly["sigma_t_kmh"] = hourly["sigma_t_kmh"].fillna(global_sigma)
    hourly["sample_count"] = hourly["sample_count"].fillna(0).astype(int)

    # 避免极小标准差导致标准化爆炸
    sigma_floor = max(global_sigma * 0.05, 0.1)
    hourly.loc[hourly["sigma_t_kmh"] < sigma_floor, "sigma_t_kmh"] = sigma_floor

    out_path = RESULT_DIR / "Regina_hourly_mu_sigma_8760.csv"
    hourly.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"8760 小时 mu_t/sigma_t 已保存：{out_path}")

    return hourly


def build_standardized_y(df: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    temp = df.merge(hourly, on="hour_index_8760", how="left")
    temp["y_standardized"] = (temp["wind_speed_kmh"] - temp["mu_t_kmh"]) / temp["sigma_t_kmh"]

    out_cols = [
        "datetime", "year", "month", "day", "hour", "hour_index_8760",
        "wind_speed_kmh", "mu_t_kmh", "sigma_t_kmh", "y_standardized"
    ]
    out = temp[out_cols].copy()
    out_path = RESULT_DIR / "Regina_standardized_y_for_ARMA_2001_2003.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"标准化 y 序列已保存：{out_path}")

    return out


# =========================================================
# 4. ARMA 拟合与模拟
# =========================================================
def fit_arma_4_3_with_statsmodels(y: np.ndarray):
    """使用 statsmodels 对 y 序列拟合 ARMA(4,3)。"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:
        print("\n未安装 statsmodels，跳过自拟合 ARMA(4,3)。")
        print(f"导入错误：{e}")
        return None

    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    print("\n========== 开始自拟合 Regina ARMA(4,3)，这一步可能需要一些时间 ==========")
    model = ARIMA(
        y,
        order=(AR_ORDER, 0, MA_ORDER),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit()

    # statsmodels 中 MA 参数是 additive 形式，可直接用于本脚本 simulate_arma_custom。
    ar = np.asarray(result.arparams, dtype=float)
    ma = np.asarray(result.maparams, dtype=float)
    sigma2 = float(result.scale if hasattr(result, "scale") else result.params[-1])
    # 某些版本 ARIMA 的 scale 固定为 1，sigma2 更可靠地从 result.params 中找 sigma2。
    for name, value in zip(result.param_names, result.params):
        if str(name).lower() in ["sigma2", "sigma2.var", "variance"]:
            sigma2 = float(value)
            break
    alpha_std = float(np.sqrt(max(sigma2, 1e-12)))

    param_rows = []
    for i, val in enumerate(ar, start=1):
        param_rows.append({"parameter_type": "AR_phi", "lag": i, "value": val})
    for i, val in enumerate(ma, start=1):
        param_rows.append({"parameter_type": "MA_additive", "lag": i, "value": val})
    param_rows.append({"parameter_type": "alpha_std", "lag": 0, "value": alpha_std})
    params_df = pd.DataFrame(param_rows)

    params_path = RESULT_DIR / "Regina_refit_ARMA_4_3_parameters.csv"
    params_df.to_csv(params_path, index=False, encoding="utf-8-sig")

    summary_path = RESULT_DIR / "Regina_refit_ARMA_4_3_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(result.summary()))

    print(f"自拟合 ARMA 参数已保存：{params_path}")
    print(f"自拟合 ARMA 摘要已保存：{summary_path}")

    return {
        "ar": ar,
        "ma": ma,
        "alpha_std": alpha_std,
        "summary": str(result.summary()),
    }


def simulate_arma_custom(ar, ma, alpha_std, n_samples, seed=20260617, burn_in=1000):
    """按 y_t = AR + alpha_t + MA_previous_alpha 形式模拟 ARMA。"""
    rng = np.random.default_rng(seed)
    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)

    p = len(ar)
    q = len(ma)
    max_lag = max(p, q)
    total = n_samples + burn_in + max_lag + 5

    alpha = rng.normal(loc=0.0, scale=alpha_std, size=total)
    y = np.zeros(total, dtype=float)

    for t in range(max_lag, total):
        ar_part = 0.0
        for i in range(p):
            ar_part += ar[i] * y[t - i - 1]

        ma_part = 0.0
        for j in range(q):
            ma_part += ma[j] * alpha[t - j - 1]

        y[t] = ar_part + alpha[t] + ma_part

    start = burn_in + max_lag + 5
    return y[start:start + n_samples]


def choose_arma_parameters(refit_result):
    if ARMA_PARAMETER_MODE.lower() == "paper":
        mode = "paper_given_Regina_ARMA_4_3"
        ar = PAPER_AR
        ma = PAPER_MA_ADDITIVE
        alpha_std = PAPER_ALPHA_STD
    elif ARMA_PARAMETER_MODE.lower() == "refit":
        if refit_result is None:
            raise RuntimeError("当前选择 refit 模式，但 statsmodels 自拟合失败。请安装 statsmodels 或改用 ARMA_PARAMETER_MODE='paper'。")
        mode = "refit_from_Regina_2001_2003"
        ar = refit_result["ar"]
        ma = refit_result["ma"]
        alpha_std = refit_result["alpha_std"]
    else:
        raise ValueError("ARMA_PARAMETER_MODE 只能是 'paper' 或 'refit'")

    param_rows = []
    for i, val in enumerate(ar, start=1):
        param_rows.append({"mode_used_for_simulation": mode, "parameter_type": "AR_phi", "lag": i, "value": val})
    for i, val in enumerate(ma, start=1):
        param_rows.append({"mode_used_for_simulation": mode, "parameter_type": "MA_additive", "lag": i, "value": val})
    param_rows.append({"mode_used_for_simulation": mode, "parameter_type": "alpha_std", "lag": 0, "value": alpha_std})

    used_params = pd.DataFrame(param_rows)
    used_path = RESULT_DIR / "Regina_ARMA_4_3_parameters_used_for_100step_simulation.csv"
    used_params.to_csv(used_path, index=False, encoding="utf-8-sig")
    print(f"用于 100-step 模拟的 ARMA 参数已保存：{used_path}")

    return mode, ar, ma, alpha_std


# =========================================================
# 5. ARMA 风速模拟与 100-step 模型
# =========================================================
def reconstruct_wind_speed(y_sim: np.ndarray, hourly: pd.DataFrame) -> pd.DataFrame:
    mu_t = hourly["mu_t_kmh"].to_numpy(dtype=float)
    sigma_t = hourly["sigma_t_kmh"].to_numpy(dtype=float)

    n = len(y_sim)
    hour_idx = np.arange(n) % 8760

    wind_raw = mu_t[hour_idx] + sigma_t[hour_idx] * y_sim
    wind_zero = np.where(wind_raw < 0, 0.0, wind_raw)

    sim_df = pd.DataFrame({
        "simulation_hour": np.arange(n),
        "simulation_year": np.arange(n) // 8760 + 1,
        "hour_index_8760": hour_idx,
        "y_simulated": y_sim,
        "wind_speed_raw_kmh": wind_raw,
        "wind_speed_kmh": wind_zero,
    })

    return sim_df


def build_100step_wind_model(sim_df: pd.DataFrame, annual_mu: float, annual_sigma: float) -> pd.DataFrame:
    """将 ARMA 模拟风速转换为 100-step Regina site-specific wind model。"""
    step_width = Z_SPAN / NB_STEPS
    step_index = np.arange(1, NB_STEPS + 1)

    # 论文 100-step 中点：-4.9, -4.8, ..., 0, ..., 5.0
    z_midpoint = step_width * (step_index - 0.5 * NB_STEPS)

    # 边界用相邻中点的中间值，首尾扩展到无穷，保证概率和为 1。
    finite_edges = (z_midpoint[:-1] + z_midpoint[1:]) / 2
    edges = np.concatenate(([-np.inf], finite_edges, [np.inf]))

    z_raw = (sim_df["wind_speed_raw_kmh"].to_numpy(dtype=float) - annual_mu) / annual_sigma
    bin_id = np.digitize(z_raw, edges, right=False) - 1
    bin_id = np.clip(bin_id, 0, NB_STEPS - 1)
    counts = np.bincount(bin_id, minlength=NB_STEPS)
    probability = counts / counts.sum()

    wind_speed_raw = annual_mu + z_midpoint * annual_sigma
    wind_speed_zero = np.where(wind_speed_raw < 0, 0.0, wind_speed_raw)

    model = pd.DataFrame({
        "step_index": step_index,
        "z_midpoint": z_midpoint,
        "wind_speed_raw_kmh": wind_speed_raw,
        "wind_speed_kmh": wind_speed_zero,
        "probability": probability,
        "count": counts,
    })

    model["probability"] = model["probability"] / model["probability"].sum()

    out_path = RESULT_DIR / "Regina_ARMA100_wind_model.csv"
    model.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Regina ARMA 100-step 风速模型已保存：{out_path}")

    return model


def save_simulation_outputs(sim_df: pd.DataFrame, actual_mu: float, actual_sigma: float, mode: str):
    one_year = sim_df[sim_df["simulation_year"] == 1].copy()
    one_year_path = RESULT_DIR / "Regina_ARMA_simulated_one_year.csv"
    one_year.to_csv(one_year_path, index=False, encoding="utf-8-sig")

    yearly_stats = (
        sim_df.groupby("simulation_year", as_index=False)
        .agg(
            mean_wind_speed_kmh=("wind_speed_kmh", "mean"),
            std_wind_speed_kmh=("wind_speed_kmh", "std"),
            min_wind_speed_kmh=("wind_speed_kmh", "min"),
            max_wind_speed_kmh=("wind_speed_kmh", "max"),
        )
    )
    yearly_stats_path = RESULT_DIR / "Regina_ARMA_simulated_yearly_statistics.csv"
    yearly_stats.to_csv(yearly_stats_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {
            "data_type": "Actual 2001-2003 cleaned",
            "mean_wind_speed_kmh": actual_mu,
            "std_wind_speed_kmh": actual_sigma,
            "simulation_years": np.nan,
            "arma_parameter_mode": "actual_data",
        },
        {
            "data_type": "ARMA simulated zero-negative all years",
            "mean_wind_speed_kmh": sim_df["wind_speed_kmh"].mean(),
            "std_wind_speed_kmh": sim_df["wind_speed_kmh"].std(ddof=1),
            "simulation_years": SIMULATION_YEARS,
            "arma_parameter_mode": mode,
        },
        {
            "data_type": "ARMA simulated raw all years",
            "mean_wind_speed_kmh": sim_df["wind_speed_raw_kmh"].mean(),
            "std_wind_speed_kmh": sim_df["wind_speed_raw_kmh"].std(ddof=1),
            "simulation_years": SIMULATION_YEARS,
            "arma_parameter_mode": mode,
        },
    ])
    summary_path = RESULT_DIR / "Regina_actual_vs_ARMA_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"模拟第一年风速已保存：{one_year_path}")
    print(f"模拟逐年统计已保存：{yearly_stats_path}")
    print(f"实际 vs ARMA 模拟统计摘要已保存：{summary_path}")
    print(summary.round(6).to_string(index=False))


def plot_100step_model(model: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.bar(model["z_midpoint"], model["probability"], width=0.08)
    plt.xlabel("Standardized wind speed z = (SW - mu) / sigma")
    plt.ylabel("Probability")
    plt.title("Regina ARMA 100-step Wind Speed Model")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = RESULT_DIR / "Regina_ARMA100_wind_model_probability.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Regina ARMA 100-step 概率图已保存：{fig_path}")


# =========================================================
# 6. 主程序
# =========================================================
def main():
    print("========== Regina ARMA(4,3) 建模与 100-step 风速模型生成 ==========")
    print(f"清洗输入文件：{CLEANED_PATH}")
    print(f"输出目录：{RESULT_DIR}")
    print(f"ARMA_PARAMETER_MODE = {ARMA_PARAMETER_MODE}")
    print(f"SIMULATION_YEARS = {SIMULATION_YEARS}")

    check_required_files()

    df = load_cleaned_data()
    actual_mu, actual_sigma = read_actual_mu_sigma()

    print("\n========== Regina 实际 2001-2003 μ / σ ==========")
    print(f"mu    = {actual_mu:.6f} km/h")
    print(f"sigma = {actual_sigma:.6f} km/h")

    hourly = build_hourly_mu_sigma(df)
    y_df = build_standardized_y(df, hourly)

    y = y_df["y_standardized"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    refit_result = None
    if TRY_REFIT_FOR_REFERENCE or ARMA_PARAMETER_MODE.lower() == "refit":
        refit_result = fit_arma_4_3_with_statsmodels(y)

    mode, ar, ma, alpha_std = choose_arma_parameters(refit_result)

    print("\n========== 开始模拟 Regina ARMA 风速 ==========")
    n_samples = SIMULATION_YEARS * 8760
    y_sim = simulate_arma_custom(
        ar=ar,
        ma=ma,
        alpha_std=alpha_std,
        n_samples=n_samples,
        seed=RANDOM_SEED,
        burn_in=BURN_IN,
    )

    sim_df = reconstruct_wind_speed(y_sim, hourly)
    save_simulation_outputs(sim_df, actual_mu, actual_sigma, mode)

    model_100 = build_100step_wind_model(sim_df, actual_mu, actual_sigma)
    plot_100step_model(model_100)

    print("\n========== 完成 ==========")
    print("后续请运行 Regina_03_Fig12_LOLE_compare.py 计算 Fig.12 LOLE 对比。")


if __name__ == "__main__":
    main()
