import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_paths import *

# =========================================================
# 1. 路径设置
# =========================================================


COMMON_MODEL_PATH = COMMON_WIND_MODEL_100STEP

OUT_DIR = APPLY_COMMON_WIND_SPEED_MODEL_RESULT_DIR


# =========================================================
# 2. 读取通用风速模型
# =========================================================

def load_common_wind_speed_model():
    """
    读取已经建立好的通用风速模型。
    需要包含：
    z_midpoint, probability
    """
    if not COMMON_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到通用风速模型文件：{COMMON_MODEL_PATH}")

    model = pd.read_csv(COMMON_MODEL_PATH, encoding="utf-8-sig")

    required_cols = ["z_midpoint", "probability"]
    missing_cols = [col for col in required_cols if col not in model.columns]

    if missing_cols:
        raise KeyError(
            f"{COMMON_MODEL_PATH.name} 缺少列：{missing_cols}\n"
            f"当前实际列名：{list(model.columns)}"
        )

    model = model[required_cols].copy()

    model["z_midpoint"] = pd.to_numeric(model["z_midpoint"], errors="coerce")
    model["probability"] = pd.to_numeric(model["probability"], errors="coerce")

    model = model.dropna(subset=["z_midpoint", "probability"]).copy()
    model = model.sort_values("z_midpoint").reset_index(drop=True)

    print("\n========== 通用风速模型读取完成 ==========")
    print(f"模型步数：{len(model)}")
    print(f"概率和：{model['probability'].sum():.8f}")

    return model


# =========================================================
# 3. 根据 μ 和 σ 转换为具体地点风速分布
# =========================================================

def apply_common_model(model: pd.DataFrame, annual_mu: float, annual_sigma: float,
                       set_negative_to_zero: bool = True):
    """
    根据论文中的思想：
        SW_i = μ + z_i * σ

    其中：
        z_i 是通用风速模型中的标准化横坐标
        μ 是该地区年平均风速
        σ 是该地区年风速标准差

    如果 set_negative_to_zero=True，则把负风速转换为 0。
    """
    result = model.copy()

    result["annual_mu"] = annual_mu
    result["annual_sigma"] = annual_sigma

    # 由标准化风速转换为具体风速
    result["wind_speed_raw_kmh"] = (
        annual_mu + result["z_midpoint"] * annual_sigma
    )

    result["probability_raw"] = result["probability"]

    if set_negative_to_zero:
        result["wind_speed_kmh"] = result["wind_speed_raw_kmh"].clip(lower=0)

        # 如果多个负风速都被压到 0，需要把它们的概率合并
        grouped = (
            result
            .groupby("wind_speed_kmh", as_index=False)["probability"]
            .sum()
            .sort_values("wind_speed_kmh")
            .reset_index(drop=True)
        )

        return result, grouped

    else:
        result["wind_speed_kmh"] = result["wind_speed_raw_kmh"]

        grouped = result[["wind_speed_kmh", "probability"]].copy()
        grouped = grouped.sort_values("wind_speed_kmh").reset_index(drop=True)

        return result, grouped


# =========================================================
# 4. 绘图
# =========================================================

def plot_site_probability_distribution(site_distribution: pd.DataFrame,
                                       site_name: str,
                                       annual_mu: float,
                                       annual_sigma: float,
                                       set_negative_to_zero: bool):
    """
    绘制某地区基于通用风速模型得到的风速概率分布图。
    """
    plt.figure(figsize=(8.5, 5.2))

    plt.plot(
        site_distribution["wind_speed_kmh"],
        site_distribution["probability"],
        linewidth=2.0,
        marker="o",
        markersize=3,
        label=site_name
    )

    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Probability")

    title = (
        f"Wind Speed Probability Distribution Based on Common Wind Speed Model\n"
        f"{site_name}: μ={annual_mu:.2f} km/h, σ={annual_sigma:.2f} km/h"
    )

    plt.title(title)

    plt.xlim(left=0 if set_negative_to_zero else site_distribution["wind_speed_kmh"].min())
    plt.ylim(bottom=0)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    safe_site_name = site_name.replace(" ", "_").replace("/", "_")

    fig_path = OUT_DIR / f"{safe_site_name}_common_model_probability_distribution.png"

    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"概率分布图已保存：{fig_path.resolve()}")


# =========================================================
# 5. 保存结果
# =========================================================

def save_results(full_result: pd.DataFrame,
                 grouped_result: pd.DataFrame,
                 site_name: str,
                 annual_mu: float,
                 annual_sigma: float,
                 set_negative_to_zero: bool):
    """
    保存转换过程表和最终概率分布表。
    """
    safe_site_name = site_name.replace(" ", "_").replace("/", "_")

    full_path = OUT_DIR / f"{safe_site_name}_common_model_full_100step.csv"
    grouped_path = OUT_DIR / f"{safe_site_name}_common_model_probability_distribution.csv"

    full_result.to_csv(full_path, index=False, encoding="utf-8-sig")
    grouped_result.to_csv(grouped_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame({
        "item": [
            "site_name",
            "annual_mu_kmh",
            "annual_sigma_kmh",
            "set_negative_to_zero",
            "probability_sum",
            "expected_mean_from_distribution",
            "expected_std_from_distribution"
        ],
        "value": [
            site_name,
            annual_mu,
            annual_sigma,
            set_negative_to_zero,
            grouped_result["probability"].sum(),
            np.sum(grouped_result["wind_speed_kmh"] * grouped_result["probability"]),
            np.sqrt(
                np.sum(
                    ((grouped_result["wind_speed_kmh"]
                      - np.sum(grouped_result["wind_speed_kmh"] * grouped_result["probability"])) ** 2)
                    * grouped_result["probability"]
                )
            )
        ]
    })

    summary_path = OUT_DIR / f"{safe_site_name}_common_model_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"完整 100 步转换表已保存：{full_path.resolve()}")
    print(f"最终风速概率分布表已保存：{grouped_path.resolve()}")
    print(f"统计摘要已保存：{summary_path.resolve()}")


# =========================================================
# 6. 主程序
# =========================================================

def main():
    print("========== 根据通用风速模型生成某地区风速概率分布 ==========")

    site_name = input("请输入地区名称，例如 Regina 或 NewSite：").strip()

    annual_mu = float(input("请输入该地区年平均风速 μ，单位 km/h：").strip())
    annual_sigma = float(input("请输入该地区年风速标准差 σ，单位 km/h：").strip())

    choice = input("是否将负风速转换为 0？输入 y/n，建议 y：").strip().lower()

    set_negative_to_zero = choice != "n"

    common_model = load_common_wind_speed_model()

    full_result, site_distribution = apply_common_model(
        model=common_model,
        annual_mu=annual_mu,
        annual_sigma=annual_sigma,
        set_negative_to_zero=set_negative_to_zero
    )

    print("\n========== 生成结果检查 ==========")
    print(f"地区：{site_name}")
    print(f"μ = {annual_mu:.4f} km/h")
    print(f"σ = {annual_sigma:.4f} km/h")
    print(f"是否负风速置 0：{set_negative_to_zero}")
    print(f"最终概率和：{site_distribution['probability'].sum():.8f}")

    print("\n前 10 行风速概率分布：")
    print(site_distribution.head(10).to_string(index=False))

    save_results(
        full_result=full_result,
        grouped_result=site_distribution,
        site_name=site_name,
        annual_mu=annual_mu,
        annual_sigma=annual_sigma,
        set_negative_to_zero=set_negative_to_zero
    )

    plot_site_probability_distribution(
        site_distribution=site_distribution,
        site_name=site_name,
        annual_mu=annual_mu,
        annual_sigma=annual_sigma,
        set_negative_to_zero=set_negative_to_zero
    )

    print("\n========== 完成 ==========")


if __name__ == "__main__":
    main()