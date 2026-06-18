import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_paths import *


# =========================================================
# 1. 路径设置
# =========================================================

OUT_DIR = COMMON_WIND_SPEED_MODEL_RESULT_DIR

# 三个地点的自拟合结果文件夹
SITES = {
    "North Battleford": {
        "site_key": "NorthBattleford",
        "result_dir": NORTH_BATTLEFORD_REFIT_ARMA_RESULT_DIR,
    },
    "Swift Current": {
        "site_key": "SwiftCurrent",
        "result_dir": SWIFTCURRENT_REFIT_ARMA_RESULT_DIR,
    },
    "Toronto Island A": {
        "site_key": "TorontoIslandA",
        "result_dir": TORONTO_ISLAND_A_REFIT_ARMA_RESULT_DIR,
    },
}

# 模拟年份数
N_SIM_YEARS = 1000
RANDOM_SEED = 2026

# Fig.1 风速概率分布参数
FIG1_X_MIN = -20
FIG1_X_MAX = 70
FIG1_BIN_WIDTH = 1.0

# Fig.2 / Fig.3 标准化概率模型参数
NB = 100


# =========================================================
# 2. 读取自拟合 ARMA 参数
# =========================================================

def read_refit_arma_params(site_key: str, result_dir: Path):
    """
    读取自拟合 ARMA(3,2) 参数。
    需要文件：
    *_refit_ARMA_3_2_parameters.csv
    """
    param_path = result_dir / f"{site_key}_refit_ARMA_3_2_parameters.csv"

    if not param_path.exists():
        raise FileNotFoundError(f"找不到 ARMA 参数文件：{param_path}")

    df = pd.read_csv(param_path, encoding="utf-8-sig")

    param_dict = dict(zip(df["parameter"], df["value"]))

    ar_coefs = [
        float(param_dict["phi_1"]),
        float(param_dict["phi_2"]),
        float(param_dict["phi_3"]),
    ]

    ma_coefs = [
        float(param_dict["theta_1"]),
        float(param_dict["theta_2"]),
    ]

    sigma_alpha = float(param_dict["sigma_alpha"])

    return ar_coefs, ma_coefs, sigma_alpha


# =========================================================
# 3. 读取 8760 个 mu_t 和 sigma_t
# =========================================================

def read_mu_sigma_vectors(site_key: str, result_dir: Path):
    """
    读取每个地点的 8760 个 mu_t 和 sigma_t。
    需要文件：
    *_hourly_mu_sigma_8760.csv
    """
    clim_path = result_dir / f"{site_key}_hourly_mu_sigma_8760.csv"

    if not clim_path.exists():
        raise FileNotFoundError(f"找不到 mu_t / sigma_t 文件：{clim_path}")

    clim = pd.read_csv(clim_path, encoding="utf-8-sig")

    required_cols = ["Month", "Day", "Hour", "mu_t", "sigma_t"]
    missing_cols = [c for c in required_cols if c not in clim.columns]

    if missing_cols:
        raise KeyError(f"{clim_path.name} 缺少列：{missing_cols}")

    # 按一年 8760 小时顺序排列
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
        clim[["Month", "Day", "Hour", "mu_t", "sigma_t"]],
        on=["Month", "Day", "Hour"],
        how="left"
    )

    if merged["mu_t"].isna().sum() > 0 or merged["sigma_t"].isna().sum() > 0:
        raise RuntimeError(f"{site_key} 的 8760 小时 mu_t / sigma_t 存在缺失。")

    mu_vec = merged["mu_t"].to_numpy()
    sigma_vec = merged["sigma_t"].to_numpy()

    return mu_vec, sigma_vec


# =========================================================
# 4. 根据 ARMA 参数模拟 y_t 和 SW_t
# =========================================================

def simulate_y_by_arma(
    n_hours: int,
    ar_coefs,
    ma_coefs,
    sigma_alpha: float,
    seed: int,
    burn_in: int = 1000
):
    """
    使用自拟合 ARMA(3,2) 参数模拟标准化扰动序列 y_t。

    y_t = phi1*y_{t-1} + phi2*y_{t-2} + phi3*y_{t-3}
          + alpha_t + theta1*alpha_{t-1} + theta2*alpha_{t-2}
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
        ar_part = sum(
            phi * y[t - i - 1]
            for i, phi in enumerate(ar_coefs)
        )

        ma_part = sum(
            theta * alpha[t - j - 1]
            for j, theta in enumerate(ma_coefs)
        )

        y[t] = ar_part + alpha[t] + ma_part

    return y[burn_in:]


def simulate_site_wind(site_name: str, site_key: str, result_dir: Path, seed_offset: int = 0):
    """
    对单个地点进行 ARMA 风速模拟。
    """
    ar_coefs, ma_coefs, sigma_alpha = read_refit_arma_params(site_key, result_dir)
    mu_vec, sigma_vec = read_mu_sigma_vectors(site_key, result_dir)

    n_hours = 8760 * N_SIM_YEARS

    y_sim = simulate_y_by_arma(
        n_hours=n_hours,
        ar_coefs=ar_coefs,
        ma_coefs=ma_coefs,
        sigma_alpha=sigma_alpha,
        seed=RANDOM_SEED + seed_offset,
        burn_in=1000
    )

    mu_all = np.tile(mu_vec, N_SIM_YEARS)
    sigma_all = np.tile(sigma_vec, N_SIM_YEARS)

    sw_raw = mu_all + sigma_all * y_sim
    sw_zero = np.maximum(sw_raw, 0)

    annual_mu = sw_raw.mean()
    annual_sigma = sw_raw.std(ddof=1)

    print("\n==========", site_name, "==========")
    print(f"ARMA 参数：")
    print(f"  AR = {ar_coefs}")
    print(f"  MA = {ma_coefs}")
    print(f"  sigma_alpha = {sigma_alpha:.6f}")
    print(f"模拟风速均值 raw：{annual_mu:.4f} km/h")
    print(f"模拟风速标准差 raw：{annual_sigma:.4f} km/h")
    print(f"负风速比例：{np.mean(sw_raw < 0):.6f}")

    return {
        "site_name": site_name,
        "site_key": site_key,
        "result_dir": result_dir,
        "ar_coefs": ar_coefs,
        "ma_coefs": ma_coefs,
        "sigma_alpha": sigma_alpha,
        "mu_vec": mu_vec,
        "sigma_vec": sigma_vec,
        "y_sim": y_sim,
        "sw_raw": sw_raw,
        "sw_zero": sw_zero,
        "annual_mu": annual_mu,
        "annual_sigma": annual_sigma,
    }


# =========================================================
# 5. Fig.1：三地点模拟风速概率分布叠加图
# =========================================================

def calculate_raw_probability_distribution(
    wind_speed,
    x_min=FIG1_X_MIN,
    x_max=FIG1_X_MAX,
    bin_width=FIG1_BIN_WIDTH
):
    bins = np.arange(x_min, x_max + bin_width, bin_width)

    counts, edges = np.histogram(wind_speed, bins=bins)

    probability = counts / len(wind_speed)

    centers = (edges[:-1] + edges[1:]) / 2

    return centers, probability


def plot_fig1(site_results: dict):
    plt.figure(figsize=(8.5, 5.2))

    fig1_table = None

    for site_name, res in site_results.items():
        x, p = calculate_raw_probability_distribution(res["sw_raw"])

        plt.plot(x, p, linewidth=1.6, label=site_name)

        temp = pd.DataFrame({
            "wind_speed_midpoint_kmh": x,
            site_name: p
        })

        if fig1_table is None:
            fig1_table = temp
        else:
            fig1_table = fig1_table.merge(
                temp,
                on="wind_speed_midpoint_kmh",
                how="outer"
            )

    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Probability")
    plt.title("Fig.1 Probability Distributions of Simulated Wind Data")
    plt.xlim(FIG1_X_MIN, FIG1_X_MAX)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig_path = OUT_DIR / "Fig1_three_sites_probability_distributions.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    csv_path = OUT_DIR / "Fig1_three_sites_probability_distributions.csv"
    fig1_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nFig.1 图像已保存：{fig_path}")
    print(f"Fig.1 数据已保存：{csv_path}")


# =========================================================
# 6. Fig.2 / Fig.3：标准化概率分布与通用风速模型
# =========================================================

def standardized_probability_distribution(sw_raw, annual_mu, annual_sigma, nb=NB):
    """
    按论文式 (7)(8) 生成标准化概率分布。

    对于 Nb = 100：
    横坐标中心点为：
    mu - 4.9 sigma, mu - 4.8 sigma, ..., mu, ..., mu + 5 sigma

    这里实际计算时使用：
    z = (SW - mu) / sigma
    然后统计 z 在 100 个标准化区间内的概率。
    """
    z = (sw_raw - annual_mu) / annual_sigma

    step = 10 / nb

    # i = 1, 2, ..., Nb
    # z_i = step * (i - 0.5 * Nb)
    z_centers = np.array([
        step * (i - 0.5 * nb)
        for i in range(1, nb + 1)
    ])

    # 区间边界：中心点左右各 0.05
    z_edges = np.concatenate([
        [z_centers[0] - step / 2],
        z_centers + step / 2
    ])

    counts, _ = np.histogram(z, bins=z_edges)

    probs = counts / len(z)

    outside_ratio = 1.0 - probs.sum()

    return z_centers, probs, outside_ratio


def plot_fig2_and_fig3(site_results: dict):
    plt.figure(figsize=(8.5, 5.2))

    std_prob_list = []
    fig2_table = None

    for site_name, res in site_results.items():
        z_centers, p_std, outside_ratio = standardized_probability_distribution(
            sw_raw=res["sw_raw"],
            annual_mu=res["annual_mu"],
            annual_sigma=res["annual_sigma"],
            nb=NB
        )

        std_prob_list.append(p_std)

        plt.plot(
            z_centers,
            p_std,
            linewidth=1.4,
            label=f"{site_name} Model"
        )

        temp = pd.DataFrame({
            "z_midpoint": z_centers,
            site_name: p_std
        })

        if fig2_table is None:
            fig2_table = temp
        else:
            fig2_table = fig2_table.merge(
                temp,
                on="z_midpoint",
                how="outer"
            )

        print(f"{site_name} 标准化区间外概率：{outside_ratio:.8f}")

    std_prob_array = np.vstack(std_prob_list)

    common_prob = std_prob_array.mean(axis=0)

    fig2_table["Common Wind Speed Model"] = common_prob

    plt.plot(
        z_centers,
        common_prob,
        "k--",
        linewidth=2.0,
        label="Common Wind Speed Model"
    )

    plt.xlabel("Simulated Wind Speed by Mean Value (μ) and Standard Deviation (σ)")
    plt.ylabel("Probability")
    plt.title("Fig.2 Combining Wind Speed Models for Different Sites")
    plt.xlim(-5, 5)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig2_path = OUT_DIR / "Fig2_combining_wind_speed_models.png"
    plt.savefig(fig2_path, dpi=300)
    plt.show()

    fig2_csv_path = OUT_DIR / "Fig2_combining_wind_speed_models.csv"
    fig2_table.to_csv(fig2_csv_path, index=False, encoding="utf-8-sig")

    print(f"\nFig.2 图像已保存：{fig2_path}")
    print(f"Fig.2 数据已保存：{fig2_csv_path}")

    # ==============================
    # Fig.3：单独画通用风速模型
    # ==============================

    common_model = pd.DataFrame({
        "step": np.arange(1, NB + 1),
        "z_midpoint": z_centers,
        "wind_speed_expression": [
            f"mu {z:+.1f} sigma".replace("+", "+ ").replace("-", "- ")
            for z in z_centers
        ],
        "probability": common_prob
    })

    common_csv_path = OUT_DIR / "CommonWindSpeedModel_100step.csv"
    common_model.to_csv(common_csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        z_centers,
        common_prob,
        linewidth=2.0,
        label="Common Wind Speed Model"
    )

    plt.xlabel("Standardized Wind Speed: (SW - μ) / σ")
    plt.ylabel("Probability")
    plt.title("Fig.3 Common Wind Speed Model")
    plt.xlim(-5, 5)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig3_path = OUT_DIR / "Fig3_common_wind_speed_model.png"
    plt.savefig(fig3_path, dpi=300)
    plt.show()

    print(f"\nFig.3 图像已保存：{fig3_path}")
    print(f"通用风速模型 100 步数据已保存：{common_csv_path}")

    print("\n========== Common Wind Speed Model 概率检查 ==========")
    print(f"Common model probability sum = {common_prob.sum():.8f}")
    print("如果略小于 1，是因为标准化区间只统计了 μ-4.95σ 到 μ+5.05σ 内的样本。")


# =========================================================
# 7. 保存三地点统计对比表
# =========================================================

def save_site_summary(site_results: dict):
    rows = []

    for site_name, res in site_results.items():
        rows.append({
            "site": site_name,
            "annual_mu_from_sim_raw": res["annual_mu"],
            "annual_sigma_from_sim_raw": res["annual_sigma"],
            "negative_wind_ratio": np.mean(res["sw_raw"] < 0),
            "ar_1": res["ar_coefs"][0],
            "ar_2": res["ar_coefs"][1],
            "ar_3": res["ar_coefs"][2],
            "ma_1": res["ma_coefs"][0],
            "ma_2": res["ma_coefs"][1],
            "sigma_alpha": res["sigma_alpha"],
        })

    summary = pd.DataFrame(rows)

    summary_path = OUT_DIR / "three_sites_refit_ARMA_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n三地点 ARMA 与统计汇总已保存：{summary_path}")


# =========================================================
# 8. 主程序
# =========================================================

def main():
    print("========== 读取三个地点自拟合 ARMA 结果并重新模拟 ==========")
    print(f"模拟年份数：{N_SIM_YEARS}")
    print(f"每个地点模拟小时数：{8760 * N_SIM_YEARS}")

    site_results = {}

    for idx, (site_name, cfg) in enumerate(SITES.items()):
        res = simulate_site_wind(
            site_name=site_name,
            site_key=cfg["site_key"],
            result_dir=cfg["result_dir"],
            seed_offset=idx * 100
        )

        site_results[site_name] = res

    save_site_summary(site_results)

    print("\n========== 开始绘制 Fig.1 ==========")
    plot_fig1(site_results)

    print("\n========== 开始绘制 Fig.2 和 Fig.3 ==========")
    plot_fig2_and_fig3(site_results)

    print("\n========== 三地点概率分布与通用风速模型生成完成 ==========")
    print(f"所有结果已保存到：{OUT_DIR}")


if __name__ == "__main__":
    main()