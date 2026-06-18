import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 路径与基本参数
# =========================================================

# 清洗后的 SwiftCurrent 文件夹
DATA_DIR = Path("SwiftCurrent_cleaned")

# 输出结果文件夹
OUT_DIR = Path("SwiftCurrent_ARMA_Result")
OUT_DIR.mkdir(exist_ok=True)

START_YEAR = 1989
END_YEAR = 2003

EXPECTED_YEARS = list(range(START_YEAR, END_YEAR + 1))
EXPECTED_MONTHS = list(range(1, 13))

# 清洗后文件名示例：
# en_climate_hourly_SK_4028040_01-1989_P1H_cleaned.csv
FILE_PATTERN = re.compile(
    r".*_(\d{2})-(\d{4})_P1H_cleaned\.csv$",
    re.IGNORECASE
)

# 需要读取的列
KEEP_COLUMNS = [
    "Year",
    "Month",
    "Day",
    "Time (LST)",
    "Wind Spd (km/h)"
]


# =========================================================
# 2. 论文 Swift Current 的 ARMA 参数
# =========================================================

# 论文公式：
# y_t = 0.8782 y_{t-1} - 0.0066 y_{t-2} + 0.0265 y_{t-3}
#       + alpha_t - 0.2162 alpha_{t-1} + 0.0091 alpha_{t-2}
#
# alpha_t ~ NID(0, 0.5579^2)

AR_COEFS = [0.8782, -0.0066, 0.0265]
MA_COEFS = [-0.2162, 0.0091]
SIGMA_ALPHA = 0.5579


# =========================================================
# 3. 文件读取与校验
# =========================================================

def parse_year_month_from_filename(file_path: Path):
    """
    从清洗后的文件名中解析 年份 和 月份。
    """
    match = FILE_PATTERN.match(file_path.name)

    if not match:
        return None, None

    month = int(match.group(1))
    year = int(match.group(2))

    return year, month


def read_one_cleaned_csv(file_path: Path) -> pd.DataFrame:
    """
    读取单个清洗后的 CSV。
    """
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
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["Wind Spd (km/h)"] = pd.to_numeric(
        df["Wind Spd (km/h)"],
        errors="coerce"
    )

    return df


def build_datetime_column(df: pd.DataFrame) -> pd.Series:
    """
    根据 Year, Month, Day, Time (LST) 构造 datetime。
    """
    date_part = (
        df["Year"].astype(str).str.zfill(4) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Day"].astype(str).str.zfill(2)
    )

    time_part = df["Time (LST)"].astype(str).str.strip()

    datetime_str = date_part + " " + time_part

    return pd.to_datetime(datetime_str, errors="coerce")


def load_all_swiftcurrent_data():
    """
    读取 SwiftCurrent_cleaned 文件夹中的所有月度文件，并合并到内存中。
    注意：这里是建模阶段需要合并，和之前清洗阶段不合并不冲突。
    """
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
    duplicate_keys = sorted([
        key for key, files in file_map.items()
        if len(files) > 1
    ])

    print("\n========== 文件完整性检查 ==========")
    print(f"扫描到 cleaned CSV 文件数：{len(csv_files)}")
    print(f"理论应有文件数：{len(expected_keys)}")
    print(f"识别到年月文件数：{len(actual_keys)}")

    if invalid_files:
        print("\n[命名不符合规则的文件]")
        for f in invalid_files:
            print("  -", f)

    if missing_keys:
        print("\n[缺失的年月]")
        for year, month in missing_keys:
            print(f"  - {year}-{month:02d}")

    if extra_keys:
        print("\n[年份范围之外的文件]")
        for year, month in extra_keys:
            print(f"  - {year}-{month:02d}")

    if duplicate_keys:
        print("\n[重复的年月文件]")
        for year, month in duplicate_keys:
            print(f"  - {year}-{month:02d}")
            for f in file_map[(year, month)]:
                print("     ", f.name)

    if invalid_files or missing_keys or extra_keys or duplicate_keys:
        raise RuntimeError("文件完整性检查未通过，请先处理上述问题。")

    all_dfs = []

    for year in EXPECTED_YEARS:
        for month in EXPECTED_MONTHS:
            file_path = file_map[(year, month)][0]
            df = read_one_cleaned_csv(file_path)
            all_dfs.append(df)

    data = pd.concat(all_dfs, ignore_index=True)

    data["datetime"] = build_datetime_column(data)

    invalid_time_count = data["datetime"].isna().sum()

    if invalid_time_count > 0:
        raise ValueError(f"存在 {invalid_time_count} 行无法构造 datetime，请检查 Time (LST) 格式。")

    # 再次过滤 1989-2003
    data = data[
        (data["Year"] >= START_YEAR)
        & (data["Year"] <= END_YEAR)
    ].copy()

    # 再次删除 2月29日，防止遗漏
    data = data[
        ~((data["Month"] == 2) & (data["Day"] == 29))
    ].copy()

    data = data.sort_values("datetime").reset_index(drop=True)

    # 检查重复时间
    duplicated_count = data.duplicated(subset=["datetime"], keep=False).sum()

    if duplicated_count > 0:
        print(f"\n警告：存在 {duplicated_count} 行重复时间记录。")

    # 检查逐年小时数
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
        raise RuntimeError("存在年份不是 8760 条数据，请先检查清洗结果。")

    total_expected = 8760 * len(EXPECTED_YEARS)

    print("\n========== 总体数据检查 ==========")
    print(f"总数据量：{len(data)}")
    print(f"理论数据量：{total_expected}")
    print(f"风速缺失值数量：{data['Wind Spd (km/h)'].isna().sum()}")

    if len(data) != total_expected:
        raise RuntimeError("总数据量不等于 8760 × 15。")

    return data


# =========================================================
# 4. 计算每个小时的 mu_t 和 sigma_t
# =========================================================

def compute_hourly_mu_sigma(data: pd.DataFrame):
    """
    用 1989-2003 年数据计算每一个 月-日-小时 的历史平均风速 mu_t 和标准差 sigma_t。
    最终应该得到 8760 个 mu_t 和 8760 个 sigma_t。
    """
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
        raise RuntimeError(f"mu_t / sigma_t 的小时数量不是 8760，而是 {len(clim)}。")

    # 防止某些小时标准差为0
    zero_sigma_count = (clim["sigma_t"] == 0).sum()
    nan_sigma_count = clim["sigma_t"].isna().sum()

    if zero_sigma_count > 0 or nan_sigma_count > 0:
        print(f"\n警告：sigma_t 为0的小时数：{zero_sigma_count}")
        print(f"警告：sigma_t 为空的小时数：{nan_sigma_count}")

        clim["sigma_t"] = clim["sigma_t"].replace(0, np.nan)
        clim["sigma_t"] = clim["sigma_t"].interpolate(limit_direction="both")

    # 保存 mu_t 和 sigma_t
    clim_path = OUT_DIR / "SwiftCurrent_hourly_mu_sigma_8760.csv"
    clim.to_csv(clim_path, index=False, encoding="utf-8-sig")

    print("\n========== mu_t / sigma_t 计算完成 ==========")
    print(f"已保存：{clim_path.resolve()}")

    return clim


def get_mu_sigma_vectors(clim: pd.DataFrame):
    """
    按一年 8760 小时顺序生成 mu_t 和 sigma_t 向量。
    """
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

    mu_vec = merged["mu_t"].to_numpy()
    sigma_vec = merged["sigma_t"].to_numpy()

    return mu_vec, sigma_vec


# =========================================================
# 5. 按论文 ARMA 公式模拟 y_t
# =========================================================

def simulate_y_by_paper_arma(
    n_hours: int,
    ar_coefs,
    ma_coefs,
    sigma_alpha: float,
    seed: int = 2026,
    burn_in: int = 1000
):
    """
    根据论文 Swift Current 的 ARMA 公式模拟 y_t。

    y_t = 0.8782 y_{t-1} - 0.0066 y_{t-2} + 0.0265 y_{t-3}
          + alpha_t - 0.2162 alpha_{t-1} + 0.0091 alpha_{t-2}

    alpha_t ~ N(0, 0.5579^2)
    """
    rng = np.random.default_rng(seed)

    p = len(ar_coefs)
    q = len(ma_coefs)
    max_lag = max(p, q)

    total_len = n_hours + burn_in

    alpha = rng.normal(
        loc=0.0,
        scale=sigma_alpha,
        size=total_len
    )

    y = np.zeros(total_len)

    for t in range(max_lag, total_len):
        ar_part = 0.0
        for i, phi in enumerate(ar_coefs):
            ar_part += phi * y[t - i - 1]

        ma_part = 0.0
        for j, theta in enumerate(ma_coefs):
            ma_part += theta * alpha[t - j - 1]

        y[t] = ar_part + alpha[t] + ma_part

    return y[burn_in:]


def simulate_wind_speed(mu_vec, sigma_vec, n_years: int = 1000):
    """
    根据 SW_t = mu_t + sigma_t * y_t 模拟风速。
    """
    n_hours = 8760 * n_years

    y_sim = simulate_y_by_paper_arma(
        n_hours=n_hours,
        ar_coefs=AR_COEFS,
        ma_coefs=MA_COEFS,
        sigma_alpha=SIGMA_ALPHA,
        seed=2026,
        burn_in=1000
    )

    mu_all = np.tile(mu_vec, n_years)
    sigma_all = np.tile(sigma_vec, n_years)

    sw_raw = mu_all + sigma_all * y_sim

    # 论文中提到负风速没有物理意义，后续可置为0
    sw_zero = np.maximum(sw_raw, 0)

    return y_sim, sw_raw, sw_zero


# =========================================================
# 6. 概率分布计算与绘图
# =========================================================

def calculate_probability_distribution(
    wind_speed,
    x_min=-20,
    x_max=70,
    bin_width=1.0
):
    """
    计算风速概率分布。
    每 1 km/h 为一个区间，概率 = 该区间样本数 / 总样本数。
    """
    bins = np.arange(x_min, x_max + bin_width, bin_width)

    counts, edges = np.histogram(wind_speed, bins=bins)

    probs = counts / len(wind_speed)

    centers = (edges[:-1] + edges[1:]) / 2

    return centers, probs


def plot_probability_distribution(x, p, title, save_path):
    """
    绘制概率分布图。
    """
    plt.figure(figsize=(8, 5))

    plt.plot(x, p, linewidth=1.8, label="Swift Current")

    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Probability")
    plt.title(title)

    plt.xlim(-20, 70)
    plt.ylim(bottom=0)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"已保存图像：{save_path.resolve()}")


# =========================================================
# 7. 主程序
# =========================================================

def main():
    # 1. 读取清洗后的 Swift Current 数据
    data = load_all_swiftcurrent_data()

    # 2. 计算 1989-2003 年每个小时的 mu_t 和 sigma_t
    clim = compute_hourly_mu_sigma(data)

    # 3. 生成 8760 小时顺序的 mu_t 和 sigma_t
    mu_vec, sigma_vec = get_mu_sigma_vectors(clim)

    # 4. 按论文 ARMA 公式模拟风速
    n_years = 1000

    print("\n========== 开始模拟 Swift Current ARMA 风速 ==========")
    print(f"模拟年份数：{n_years}")
    print(f"模拟小时数：{8760 * n_years}")

    y_sim, sw_raw, sw_zero = simulate_wind_speed(
        mu_vec=mu_vec,
        sigma_vec=sigma_vec,
        n_years=n_years
    )

    # 5. 输出统计特征
    actual_mean = data["Wind Spd (km/h)"].mean()
    actual_std = data["Wind Spd (km/h)"].std(ddof=1)

    raw_mean = sw_raw.mean()
    raw_std = sw_raw.std(ddof=1)

    zero_mean = sw_zero.mean()
    zero_std = sw_zero.std(ddof=1)

    print("\n========== 统计特征对比 ==========")
    print(f"实际风速均值：{actual_mean:.4f} km/h")
    print(f"实际风速标准差：{actual_std:.4f} km/h")
    print(f"ARMA模拟风速均值，未处理负值：{raw_mean:.4f} km/h")
    print(f"ARMA模拟风速标准差，未处理负值：{raw_std:.4f} km/h")
    print(f"ARMA模拟风速均值，负值置0：{zero_mean:.4f} km/h")
    print(f"ARMA模拟风速标准差，负值置0：{zero_std:.4f} km/h")
    print(f"模拟风速中负值比例：{np.mean(sw_raw < 0):.6f}")

    # 6. 保存模拟序列的前一年结果，避免文件过大
    one_year_result = pd.DataFrame({
        "hour": np.arange(1, 8761),
        "mu_t": mu_vec,
        "sigma_t": sigma_vec,
        "y_t_simulated": y_sim[:8760],
        "SW_raw_kmh": sw_raw[:8760],
        "SW_negative_set_zero_kmh": sw_zero[:8760]
    })

    one_year_path = OUT_DIR / "SwiftCurrent_ARMA_simulated_one_year.csv"
    one_year_result.to_csv(one_year_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存模拟前一年数据：{one_year_path.resolve()}")

    # 7. 绘制未处理负风速的概率分布图
    # 这个图更接近论文 Fig.1，因为论文图中可以看到负风速区域
    x_raw, p_raw = calculate_probability_distribution(
        wind_speed=sw_raw,
        x_min=-20,
        x_max=70,
        bin_width=1.0
    )

    prob_raw_df = pd.DataFrame({
        "wind_speed_midpoint_kmh": x_raw,
        "probability": p_raw
    })

    prob_raw_path = OUT_DIR / "SwiftCurrent_probability_distribution_raw.csv"
    prob_raw_df.to_csv(prob_raw_path, index=False, encoding="utf-8-sig")

    fig_raw_path = OUT_DIR / "SwiftCurrent_probability_distribution_raw.png"

    plot_probability_distribution(
        x=x_raw,
        p=p_raw,
        title="Probability Distribution of Simulated Wind Data - Swift Current",
        save_path=fig_raw_path
    )

    # 8. 绘制负风速置0后的概率分布图
    # 这个图用于后续物理风速模型和风机功率模型
    x_zero, p_zero = calculate_probability_distribution(
        wind_speed=sw_zero,
        x_min=0,
        x_max=70,
        bin_width=1.0
    )

    prob_zero_df = pd.DataFrame({
        "wind_speed_midpoint_kmh": x_zero,
        "probability": p_zero
    })

    prob_zero_path = OUT_DIR / "SwiftCurrent_probability_distribution_negative_set_zero.csv"
    prob_zero_df.to_csv(prob_zero_path, index=False, encoding="utf-8-sig")

    fig_zero_path = OUT_DIR / "SwiftCurrent_probability_distribution_negative_set_zero.png"

    plt.figure(figsize=(8, 5))
    plt.plot(x_zero, p_zero, linewidth=1.8, label="Swift Current")
    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Probability")
    plt.title("Probability Distribution after Negative Wind Speeds Set to Zero")
    plt.xlim(0, 70)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_zero_path, dpi=300)
    plt.show()

    print(f"已保存图像：{fig_zero_path.resolve()}")

    # 9. 保存统计报告
    report = pd.DataFrame({
        "item": [
            "actual_mean",
            "actual_std",
            "simulated_raw_mean",
            "simulated_raw_std",
            "simulated_negative_set_zero_mean",
            "simulated_negative_set_zero_std",
            "negative_wind_speed_ratio",
            "simulation_years",
            "simulation_hours"
        ],
        "value": [
            actual_mean,
            actual_std,
            raw_mean,
            raw_std,
            zero_mean,
            zero_std,
            np.mean(sw_raw < 0),
            n_years,
            8760 * n_years
        ]
    })

    report_path = OUT_DIR / "SwiftCurrent_ARMA_statistical_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"\n已保存统计报告：{report_path.resolve()}")

    print("\n========== Swift Current ARMA 模型与概率分布建立完成 ==========")


if __name__ == "__main__":
    main()