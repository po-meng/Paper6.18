import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_paths import *


# =========================================================
# 1. 基本参数
# =========================================================

SITE_NAME = "SwiftCurrent"

DATA_DIR = SWIFTCURRENT_CLEANED_DIR
OUT_DIR = SWIFTCURRENT_REFIT_ARMA_RESULT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1989
END_YEAR = 2003

FIT_START_YEAR = 2001
FIT_END_YEAR = 2003

EXPECTED_YEARS = list(range(START_YEAR, END_YEAR + 1))
EXPECTED_MONTHS = list(range(1, 13))

AR_ORDER = 3
MA_ORDER = 2

N_SIM_YEARS = 1000
RANDOM_SEED = 2026

FILE_PATTERN = re.compile(
    r".*_(\d{2})-(\d{4})_P1H_cleaned\.csv$",
    re.IGNORECASE
)

KEEP_COLUMNS = [
    "Year",
    "Month",
    "Day",
    "Time (LST)",
    "Wind Spd (km/h)"
]


# =========================================================
# 2. 数据读取
# =========================================================

def parse_year_month_from_filename(file_path: Path):
    match = FILE_PATTERN.match(file_path.name)
    if not match:
        return None, None

    month = int(match.group(1))
    year = int(match.group(2))
    return year, month


def read_one_cleaned_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]

    missing_cols = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"{file_path.name} 缺少列：{missing_cols}\n"
            f"当前实际列名：{list(df.columns)}"
        )

    df = df[KEEP_COLUMNS].copy()

    for col in ["Year", "Month", "Day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Wind Spd (km/h)"] = pd.to_numeric(
        df["Wind Spd (km/h)"],
        errors="coerce"
    )

    df = df.dropna(subset=["Year", "Month", "Day"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Day"] = df["Day"].astype(int)

    return df


def build_datetime_column(df: pd.DataFrame) -> pd.Series:
    date_part = (
        df["Year"].astype(str).str.zfill(4) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Day"].astype(str).str.zfill(2)
    )

    time_part = df["Time (LST)"].astype(str).str.strip()
    datetime_str = date_part + " " + time_part

    return pd.to_datetime(datetime_str, errors="coerce")


def load_all_data():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"找不到数据文件夹：{DATA_DIR.resolve()}")

    csv_files = sorted(DATA_DIR.glob("*_cleaned.csv"))

    if not csv_files:
        raise FileNotFoundError(f"{DATA_DIR.resolve()} 下没有找到 *_cleaned.csv 文件")

    file_map = {}
    invalid_files = []

    for file_path in csv_files:
        year, month = parse_year_month_from_filename(file_path)

        if year is None or month is None:
            invalid_files.append(file_path.name)
            continue

        file_map.setdefault((year, month), []).append(file_path)

    expected_keys = {
        (year, month)
        for year in EXPECTED_YEARS
        for month in EXPECTED_MONTHS
    }

    actual_keys = set(file_map.keys())

    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    duplicate_keys = sorted([key for key, files in file_map.items() if len(files) > 1])

    print(f"\n========== {SITE_NAME} cleaned 文件检查 ==========")
    print(f"扫描到 cleaned CSV 文件数：{len(csv_files)}")
    print(f"理论应有文件数：{len(expected_keys)}")
    print(f"识别到年月文件数：{len(actual_keys)}")

    if invalid_files:
        print("\n[命名不符合规则]")
        for f in invalid_files:
            print("  -", f)

    if missing_keys:
        print("\n[缺失的年月]")
        for year, month in missing_keys:
            print(f"  - {year}-{month:02d}")

    if extra_keys:
        print("\n[年份范围外文件]")
        for year, month in extra_keys:
            print(f"  - {year}-{month:02d}")

    if duplicate_keys:
        print("\n[重复年月文件]")
        for year, month in duplicate_keys:
            print(f"  - {year}-{month:02d}")
            for f in file_map[(year, month)]:
                print("     ", f.name)

    if invalid_files or missing_keys or extra_keys or duplicate_keys:
        raise RuntimeError("cleaned 文件完整性检查未通过。")

    all_dfs = []

    for year in EXPECTED_YEARS:
        for month in EXPECTED_MONTHS:
            file_path = file_map[(year, month)][0]
            all_dfs.append(read_one_cleaned_csv(file_path))

    data = pd.concat(all_dfs, ignore_index=True)
    data["datetime"] = build_datetime_column(data)

    invalid_time_count = data["datetime"].isna().sum()
    if invalid_time_count > 0:
        raise ValueError(f"存在 {invalid_time_count} 行无法构造 datetime。")

    data = data[~((data["Month"] == 2) & (data["Day"] == 29))].copy()
    data = data.sort_values("datetime").reset_index(drop=True)

    annual_count = (
        data.groupby("Year")
        .size()
        .reset_index(name="hour_count")
    )
    annual_count["expected"] = 8760
    annual_count["ok"] = annual_count["hour_count"] == annual_count["expected"]

    print("\n========== 逐年小时数检查 ==========")
    print(annual_count.to_string(index=False))

    if not annual_count["ok"].all():
        raise RuntimeError("存在年份不是 8760 条数据，请检查清洗结果。")

    total_expected = 8760 * len(EXPECTED_YEARS)

    print("\n========== 总体数据检查 ==========")
    print(f"总数据量：{len(data)}")
    print(f"理论数据量：{total_expected}")
    print(f"风速缺失值数量：{data['Wind Spd (km/h)'].isna().sum()}")

    if len(data) != total_expected:
        raise RuntimeError("总数据量不等于 8760 × 15。")

    return data


# =========================================================
# 3. 计算 mu_t 和 sigma_t
# =========================================================

def compute_hourly_mu_sigma(data: pd.DataFrame):
    data = data.copy()
    data["Hour"] = data["datetime"].dt.hour

    clim = (
        data
        .groupby(["Month", "Day", "Hour"])["Wind Spd (km/h)"]
        .agg(
            mu_t="mean",
            sigma_t=lambda x: x.std(ddof=1),
            count="count"
        )
        .reset_index()
    )

    if len(clim) != 8760:
        raise RuntimeError(f"mu_t / sigma_t 数量不是 8760，而是 {len(clim)}。")

    zero_sigma_count = (clim["sigma_t"] == 0).sum()
    nan_sigma_count = clim["sigma_t"].isna().sum()

    if zero_sigma_count > 0 or nan_sigma_count > 0:
        print(f"\n警告：sigma_t 为 0 的小时数：{zero_sigma_count}")
        print(f"警告：sigma_t 为空的小时数：{nan_sigma_count}")
        clim["sigma_t"] = clim["sigma_t"].replace(0, np.nan)
        clim["sigma_t"] = clim["sigma_t"].interpolate(limit_direction="both")

    clim_path = OUT_DIR / f"{SITE_NAME}_hourly_mu_sigma_8760.csv"
    clim.to_csv(clim_path, index=False, encoding="utf-8-sig")

    print("\n========== mu_t / sigma_t 计算完成 ==========")
    print(f"已保存：{clim_path.resolve()}")

    return clim


def get_mu_sigma_vectors(clim: pd.DataFrame):
    ref_time = pd.date_range(
        start="2001-01-01 00:00:00",
        end="2001-12-31 23:00:00",
        freq="h"
    )

    ref_df = pd.DataFrame({
        "Month": ref_time.month,
        "Day": ref_time.day,
        "Hour": ref_time.hour
    })

    merged = ref_df.merge(
        clim,
        on=["Month", "Day", "Hour"],
        how="left"
    )

    if merged["mu_t"].isna().sum() > 0 or merged["sigma_t"].isna().sum() > 0:
        raise RuntimeError("生成 8760 小时 mu_t / sigma_t 向量时存在缺失。")

    return merged["mu_t"].to_numpy(), merged["sigma_t"].to_numpy()


# =========================================================
# 4. 构造 y_t 并拟合 ARMA
# =========================================================

def build_standardized_y_series(data: pd.DataFrame, clim: pd.DataFrame):
    data = data.copy()
    data["Hour"] = data["datetime"].dt.hour

    fit_data = data[
        (data["Year"] >= FIT_START_YEAR)
        & (data["Year"] <= FIT_END_YEAR)
    ].copy()

    merged = fit_data.merge(
        clim[["Month", "Day", "Hour", "mu_t", "sigma_t"]],
        on=["Month", "Day", "Hour"],
        how="left"
    )

    merged["y_t"] = (
        (merged["Wind Spd (km/h)"] - merged["mu_t"])
        / merged["sigma_t"]
    )

    before_drop = len(merged)

    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["y_t"]).copy()

    print("\n========== 标准化 y_t 序列构造完成 ==========")
    print(f"拟合数据年份：{FIT_START_YEAR}-{FIT_END_YEAR}")
    print(f"理论小时数：{8760 * (FIT_END_YEAR - FIT_START_YEAR + 1)}")
    print(f"实际用于拟合的 y_t 数量：{len(merged)}")
    print(f"删除行数：{before_drop - len(merged)}")
    print(f"y_t 均值：{merged['y_t'].mean():.6f}")
    print(f"y_t 标准差：{merged['y_t'].std(ddof=1):.6f}")

    y_path = OUT_DIR / f"{SITE_NAME}_standardized_y_for_ARMA_fit_2001_2003.csv"
    merged[[
        "datetime",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Wind Spd (km/h)",
        "mu_t",
        "sigma_t",
        "y_t"
    ]].to_csv(y_path, index=False, encoding="utf-8-sig")

    print(f"已保存 y_t 拟合序列：{y_path.resolve()}")

    return merged["y_t"].to_numpy()


def fit_arma_model(y: np.ndarray):
    print("\n========== 开始拟合 ARMA(3,2) ==========")

    warnings.filterwarnings("ignore")

    model = ARIMA(
        y,
        order=(AR_ORDER, 0, MA_ORDER),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit()

    print(result.summary())

    summary_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_3_2_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(result.summary()))

    print(f"\n已保存 ARMA 拟合 summary：{summary_path.resolve()}")

    return result


def extract_arma_params(result):
    param_dict = dict(zip(result.param_names, result.params))

    ar_coefs = [float(param_dict.get(f"ar.L{i}", 0.0)) for i in range(1, AR_ORDER + 1)]
    ma_coefs = [float(param_dict.get(f"ma.L{i}", 0.0)) for i in range(1, MA_ORDER + 1)]

    sigma2 = float(param_dict.get("sigma2", np.var(result.resid, ddof=1)))
    sigma_alpha = float(np.sqrt(sigma2))

    param_table = pd.DataFrame({
        "parameter": (
            [f"phi_{i}" for i in range(1, AR_ORDER + 1)]
            + [f"theta_{i}" for i in range(1, MA_ORDER + 1)]
            + ["sigma_alpha", "sigma_alpha_squared", "AIC", "BIC"]
        ),
        "value": (
            ar_coefs
            + ma_coefs
            + [sigma_alpha, sigma2, result.aic, result.bic]
        )
    })

    param_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_3_2_parameters.csv"
    param_table.to_csv(param_path, index=False, encoding="utf-8-sig")

    print(f"\n========== {SITE_NAME} 重新拟合的 ARMA 参数 ==========")
    print(
        f"y_t = {ar_coefs[0]:+.6f} y_(t-1) "
        f"{ar_coefs[1]:+.6f} y_(t-2) "
        f"{ar_coefs[2]:+.6f} y_(t-3)"
    )
    print(
        f"      + alpha_t "
        f"{ma_coefs[0]:+.6f} alpha_(t-1) "
        f"{ma_coefs[1]:+.6f} alpha_(t-2)"
    )
    print(f"alpha_t ~ NID(0, {sigma_alpha:.6f}^2)")
    print(f"AIC = {result.aic:.4f}")
    print(f"BIC = {result.bic:.4f}")
    print(f"已保存参数表：{param_path.resolve()}")

    return ar_coefs, ma_coefs, sigma_alpha


# =========================================================
# 5. 模拟与绘图
# =========================================================

def simulate_y_by_fitted_arma(n_hours, ar_coefs, ma_coefs, sigma_alpha, seed=RANDOM_SEED, burn_in=1000):
    rng = np.random.default_rng(seed)

    p = len(ar_coefs)
    q = len(ma_coefs)
    max_lag = max(p, q)

    total_len = n_hours + burn_in

    alpha = rng.normal(0.0, sigma_alpha, size=total_len)
    y = np.zeros(total_len)

    for t in range(max_lag, total_len):
        ar_part = sum(phi * y[t - i - 1] for i, phi in enumerate(ar_coefs))
        ma_part = sum(theta * alpha[t - j - 1] for j, theta in enumerate(ma_coefs))
        y[t] = ar_part + alpha[t] + ma_part

    return y[burn_in:]


def simulate_wind_speed(mu_vec, sigma_vec, ar_coefs, ma_coefs, sigma_alpha, n_years):
    n_hours = 8760 * n_years

    y_sim = simulate_y_by_fitted_arma(
        n_hours=n_hours,
        ar_coefs=ar_coefs,
        ma_coefs=ma_coefs,
        sigma_alpha=sigma_alpha,
        seed=RANDOM_SEED,
        burn_in=1000
    )

    mu_all = np.tile(mu_vec, n_years)
    sigma_all = np.tile(sigma_vec, n_years)

    sw_raw = mu_all + sigma_all * y_sim
    sw_zero = np.maximum(sw_raw, 0)

    return y_sim, sw_raw, sw_zero


def calculate_probability_distribution(wind_speed, x_min=-20, x_max=70, bin_width=1.0):
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    counts, edges = np.histogram(wind_speed, bins=bins)
    probs = counts / len(wind_speed)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, probs


def plot_probability_distribution(x, p, title, save_path, label, x_min=-20, x_max=70):
    plt.figure(figsize=(8, 5))
    plt.plot(x, p, linewidth=1.8, label=label)
    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"已保存图像：{save_path.resolve()}")


def save_statistics_and_outputs(data, mu_vec, sigma_vec, y_sim, sw_raw, sw_zero, n_years):
    actual_mean = data["Wind Spd (km/h)"].mean()
    actual_std = data["Wind Spd (km/h)"].std(ddof=1)

    raw_mean = sw_raw.mean()
    raw_std = sw_raw.std(ddof=1)

    zero_mean = sw_zero.mean()
    zero_std = sw_zero.std(ddof=1)

    print("\n========== 统计特征对比 ==========")
    print(f"实际风速均值：{actual_mean:.4f} km/h")
    print(f"实际风速标准差：{actual_std:.4f} km/h")
    print(f"重新拟合 ARMA 模拟风速均值，未处理负值：{raw_mean:.4f} km/h")
    print(f"重新拟合 ARMA 模拟风速标准差，未处理负值：{raw_std:.4f} km/h")
    print(f"重新拟合 ARMA 模拟风速均值，负值置0：{zero_mean:.4f} km/h")
    print(f"重新拟合 ARMA 模拟风速标准差，负值置0：{zero_std:.4f} km/h")
    print(f"模拟风速中负值比例：{np.mean(sw_raw < 0):.6f}")

    one_year_result = pd.DataFrame({
        "hour": np.arange(1, 8761),
        "mu_t": mu_vec,
        "sigma_t": sigma_vec,
        "y_t_simulated": y_sim[:8760],
        "SW_raw_kmh": sw_raw[:8760],
        "SW_negative_set_zero_kmh": sw_zero[:8760]
    })

    one_year_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_simulated_one_year.csv"
    one_year_result.to_csv(one_year_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存模拟前一年数据：{one_year_path.resolve()}")

    x_raw, p_raw = calculate_probability_distribution(sw_raw, -20, 70, 1.0)

    prob_raw_df = pd.DataFrame({
        "wind_speed_midpoint_kmh": x_raw,
        "probability": p_raw
    })

    prob_raw_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_probability_distribution_raw.csv"
    prob_raw_df.to_csv(prob_raw_path, index=False, encoding="utf-8-sig")

    fig_raw_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_probability_distribution_raw.png"

    plot_probability_distribution(
        x_raw,
        p_raw,
        f"Probability Distribution of Simulated Wind Data - {SITE_NAME} Refit ARMA",
        fig_raw_path,
        f"{SITE_NAME} Refit ARMA",
        -20,
        70
    )

    x_zero, p_zero = calculate_probability_distribution(sw_zero, 0, 70, 1.0)

    prob_zero_df = pd.DataFrame({
        "wind_speed_midpoint_kmh": x_zero,
        "probability": p_zero
    })

    prob_zero_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_probability_distribution_negative_set_zero.csv"
    prob_zero_df.to_csv(prob_zero_path, index=False, encoding="utf-8-sig")

    fig_zero_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_probability_distribution_negative_set_zero.png"

    plot_probability_distribution(
        x_zero,
        p_zero,
        f"Probability Distribution after Negative Wind Speeds Set to Zero - {SITE_NAME} Refit ARMA",
        fig_zero_path,
        f"{SITE_NAME} Refit ARMA",
        0,
        70
    )

    actual_yearly_stats = (
        data
        .groupby("Year")["Wind Spd (km/h)"]
        .agg(
            actual_mean="mean",
            actual_std=lambda x: x.std(ddof=1),
            actual_count="count"
        )
        .reset_index()
    )

    actual_yearly_path = OUT_DIR / f"{SITE_NAME}_actual_yearly_statistics_1989_2003.csv"
    actual_yearly_stats.to_csv(actual_yearly_path, index=False, encoding="utf-8-sig")

    sim_year_index = np.repeat(np.arange(1, n_years + 1), 8760)

    sim_df = pd.DataFrame({
        "simulation_year": sim_year_index,
        "SW_raw_kmh": sw_raw,
        "SW_negative_set_zero_kmh": sw_zero
    })

    sim_yearly_stats = (
        sim_df
        .groupby("simulation_year")
        .agg(
            simulated_raw_mean=("SW_raw_kmh", "mean"),
            simulated_raw_std=("SW_raw_kmh", lambda x: x.std(ddof=1)),
            simulated_zero_mean=("SW_negative_set_zero_kmh", "mean"),
            simulated_zero_std=("SW_negative_set_zero_kmh", lambda x: x.std(ddof=1)),
            count=("SW_raw_kmh", "count")
        )
        .reset_index()
    )

    sim_yearly_path = OUT_DIR / f"{SITE_NAME}_refit_ARMA_simulated_yearly_statistics.csv"
    sim_yearly_stats.to_csv(sim_yearly_path, index=False, encoding="utf-8-sig")

    summary_compare = pd.DataFrame({
        "data_type": [
            "Actual 1989-2003",
            "Refit ARMA simulated raw",
            "Refit ARMA simulated negative set zero"
        ],
        "mean_wind_speed_kmh": [
            actual_mean,
            raw_mean,
            zero_mean
        ],
        "std_wind_speed_kmh": [
            actual_std,
            raw_std,
            zero_std
        ],
        "sample_hours": [
            len(data),
            len(sw_raw),
            len(sw_zero)
        ]
    })

    summary_path = OUT_DIR / f"{SITE_NAME}_actual_vs_refit_ARMA_summary.csv"
    summary_compare.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n已保存真实逐年统计：{actual_yearly_path.resolve()}")
    print(f"已保存模拟逐年统计：{sim_yearly_path.resolve()}")
    print(f"已保存统计汇总：{summary_path.resolve()}")


# =========================================================
# 6. 主程序
# =========================================================

def main():
    data = load_all_data()

    clim = compute_hourly_mu_sigma(data)

    y_fit = build_standardized_y_series(data, clim)

    result = fit_arma_model(y_fit)

    ar_coefs, ma_coefs, sigma_alpha = extract_arma_params(result)

    mu_vec, sigma_vec = get_mu_sigma_vectors(clim)

    print(f"\n========== 开始用重新拟合 ARMA 模型模拟 {SITE_NAME} 风速 ==========")
    print(f"模拟年份数：{N_SIM_YEARS}")
    print(f"模拟小时数：{8760 * N_SIM_YEARS}")

    y_sim, sw_raw, sw_zero = simulate_wind_speed(
        mu_vec=mu_vec,
        sigma_vec=sigma_vec,
        ar_coefs=ar_coefs,
        ma_coefs=ma_coefs,
        sigma_alpha=sigma_alpha,
        n_years=N_SIM_YEARS
    )

    save_statistics_and_outputs(
        data=data,
        mu_vec=mu_vec,
        sigma_vec=sigma_vec,
        y_sim=y_sim,
        sw_raw=sw_raw,
        sw_zero=sw_zero,
        n_years=N_SIM_YEARS
    )

    print(f"\n========== {SITE_NAME} 自拟合 ARMA 模型与概率分布建立完成 ==========")


if __name__ == "__main__":
    main()