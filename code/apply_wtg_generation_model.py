import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# 1. 路径设置
# =========================================================

BASE_DIR = Path(r"D:\tools\za\paper")

# 这里读取你刚刚用通用风速模型生成的示例地点风速概率分布
INPUT_DIR = BASE_DIR / "Apply_Common_Wind_Speed_Model_Result"

# 如果你的文件名不是 example_site，可以在这里改
SITE_DISTRIBUTION_PATH = INPUT_DIR / "example_site_common_model_probability_distribution.csv"

OUT_DIR = BASE_DIR / "WTG_Power_Generation_Model_Result"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. WTG 参数：论文 Regina 示例
# =========================================================

# 额定功率，单位 MW
PR_MW = 1.5

# 切入风速、额定风速、切出风速，单位 km/h
VCI = 14.4
VR = 45.0
VCO = 90.0

# 功率输出保留小数位数
# Table III 中一般展示 4 位左右比较合适
POWER_ROUND_DECIMALS = 4


# =========================================================
# 3. 计算论文式(9)中的 A、B、C
# =========================================================

def calculate_abc(vci: float, vr: float):
    """
    计算论文式(9)中二次功率曲线的 A、B、C 常数。

    P = Pr * (A + B * SW + C * SW^2), Vci <= SW < Vr
    """
    k = (vci + vr) / (2 * vr)

    a = (
        vci * (vci + vr)
        - 4 * vci * vr * (k ** 3)
    ) / ((vci - vr) ** 2)

    b = (
        4 * (vci + vr) * (k ** 3)
        - (3 * vci + vr)
    ) / ((vci - vr) ** 2)

    c = (
        2
        - 4 * (k ** 3)
    ) / ((vci - vr) ** 2)

    return a, b, c


A, B, C = calculate_abc(VCI, VR)


# =========================================================
# 4. 读取示例地点风速概率分布
# =========================================================

def load_site_wind_distribution():
    """
    读取通用风速模型转换后的具体地点风速概率分布。

    需要至少包含：
    wind_speed_kmh, probability

    如果找不到指定文件，会自动在目录下寻找
    *_common_model_probability_distribution.csv
    """
    path = SITE_DISTRIBUTION_PATH

    if not path.exists():
        candidates = list(INPUT_DIR.glob("*_common_model_probability_distribution.csv"))

        if not candidates:
            raise FileNotFoundError(
                f"找不到风速概率分布文件：{SITE_DISTRIBUTION_PATH}\n"
                f"并且 {INPUT_DIR} 下也没有 *_common_model_probability_distribution.csv"
            )

        path = candidates[0]
        print(f"未找到指定文件，自动使用：{path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    if "wind_speed_kmh" not in df.columns:
        raise KeyError(
            f"{path.name} 中找不到 wind_speed_kmh 列。\n"
            f"当前列名为：{list(df.columns)}"
        )

    if "probability" not in df.columns:
        raise KeyError(
            f"{path.name} 中找不到 probability 列。\n"
            f"当前列名为：{list(df.columns)}"
        )

    df = df[["wind_speed_kmh", "probability"]].copy()

    df["wind_speed_kmh"] = pd.to_numeric(df["wind_speed_kmh"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    df = df.dropna(subset=["wind_speed_kmh", "probability"]).copy()

    # 确保负风速已经置 0
    df["wind_speed_kmh"] = df["wind_speed_kmh"].clip(lower=0)

    # 如果多个负风速已经被压到 0，这里再次合并概率，防止重复
    df = (
        df
        .groupby("wind_speed_kmh", as_index=False)["probability"]
        .sum()
        .sort_values("wind_speed_kmh")
        .reset_index(drop=True)
    )

    print("\n========== 示例地点风速概率分布读取完成 ==========")
    print(f"读取文件：{path}")
    print(f"风速状态数：{len(df)}")
    print(f"概率和：{df['probability'].sum():.8f}")

    return df, path


# =========================================================
# 5. WTG 功率曲线
# =========================================================

def wtg_power_output(sw: float) -> float:
    """
    根据论文式(9)计算某一风速对应的 WTG 输出功率，单位 MW。

    Pi = 0, 0 <= SW < Vci
       = Pr(A + B*SW + C*SW^2), Vci <= SW < Vr
       = Pr, Vr <= SW <= Vco
       = 0, Vco < SW
    """
    if 0 <= sw < VCI:
        return 0.0

    elif VCI <= sw < VR:
        p = PR_MW * (A + B * sw + C * sw ** 2)

        # 防止数值误差导致略小于0或略大于Pr
        p = max(0.0, min(PR_MW, p))

        return p

    elif VR <= sw <= VCO:
        return PR_MW

    else:
        return 0.0


def build_power_generation_model(wind_df: pd.DataFrame):
    """
    将风速概率模型转换为 WTG 功率概率模型，并按照相同功率输出聚合概率。
    """
    expanded = wind_df.copy()

    expanded["power_output_MW_raw"] = expanded["wind_speed_kmh"].apply(wtg_power_output)

    # 为了得到类似 Table III 的有限状态模型，对功率进行四舍五入后聚合
    expanded["power_output_MW"] = expanded["power_output_MW_raw"].round(POWER_ROUND_DECIMALS)

    # 标记功率区间类型
    def classify_region(sw):
        if sw < VCI:
            return "zero_below_cut_in"
        elif VCI <= sw < VR:
            return "partial_power"
        elif VR <= sw <= VCO:
            return "rated_power"
        else:
            return "zero_above_cut_out"

    expanded["region"] = expanded["wind_speed_kmh"].apply(classify_region)

    # 聚合同一功率输出的概率
    table3 = (
        expanded
        .groupby("power_output_MW", as_index=False)
        .agg(
            probability=("probability", "sum"),
            wind_speed_min_kmh=("wind_speed_kmh", "min"),
            wind_speed_max_kmh=("wind_speed_kmh", "max"),
            wind_speed_step_count=("wind_speed_kmh", "count"),
            regions=("region", lambda x: ",".join(sorted(set(x))))
        )
        .sort_values("power_output_MW")
        .reset_index(drop=True)
    )

    return expanded, table3


# =========================================================
# 6. 绘图：WTG 功率曲线
# =========================================================

def plot_wtg_power_curve():
    sw_values = np.linspace(0, 100, 1000)
    p_values = np.array([wtg_power_output(sw) for sw in sw_values])

    plt.figure(figsize=(8, 5))

    plt.plot(sw_values, p_values, linewidth=2)

    plt.axvline(VCI, linestyle="--", linewidth=1, label=f"Vci={VCI} km/h")
    plt.axvline(VR, linestyle="--", linewidth=1, label=f"Vr={VR} km/h")
    plt.axvline(VCO, linestyle="--", linewidth=1, label=f"Vco={VCO} km/h")
    plt.axhline(PR_MW, linestyle="--", linewidth=1, label=f"Pr={PR_MW} MW")

    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Power Output (MW)")
    plt.title("WTG Power Curve")
    plt.xlim(0, 100)
    plt.ylim(0, PR_MW * 1.15)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig_path = OUT_DIR / "WTG_power_curve.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"WTG功率曲线图已保存：{fig_path}")


# =========================================================
# 7. 绘图：功率概率分布
# =========================================================

def plot_power_probability_distribution(table3: pd.DataFrame):
    plt.figure(figsize=(8, 5))

    plt.bar(
        table3["power_output_MW"],
        table3["probability"],
        width=0.025,
        align="center"
    )

    plt.xlabel("Power Output (MW)")
    plt.ylabel("Probability")
    plt.title("WTG Power Generation Model")
    plt.xlim(0, PR_MW * 1.05)
    plt.ylim(bottom=0)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = OUT_DIR / "WTG_power_generation_probability_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"WTG功率概率分布图已保存：{fig_path}")


# =========================================================
# 8. 保存结果
# =========================================================

def save_results(expanded: pd.DataFrame, table3: pd.DataFrame, input_path: Path):
    expanded_path = OUT_DIR / "WTG_power_generation_expanded_100step.csv"
    table3_path = OUT_DIR / "WTG_power_generation_model_TableIII_like.csv"

    expanded.to_csv(expanded_path, index=False, encoding="utf-8-sig")
    table3.to_csv(table3_path, index=False, encoding="utf-8-sig")

    # 计算 P0 和 Pr
    zero_probability = table3.loc[
        table3["power_output_MW"] == 0,
        "probability"
    ].sum()

    rated_probability = table3.loc[
        table3["power_output_MW"] == round(PR_MW, POWER_ROUND_DECIMALS),
        "probability"
    ].sum()

    summary = pd.DataFrame({
        "item": [
            "input_wind_distribution_file",
            "rated_power_Pr_MW",
            "cut_in_speed_Vci_kmh",
            "rated_speed_Vr_kmh",
            "cut_out_speed_Vco_kmh",
            "A",
            "B",
            "C",
            "wind_state_probability_sum",
            "power_state_probability_sum",
            "number_of_wind_speed_states",
            "number_of_power_output_states",
            "zero_output_probability_P0",
            "rated_output_probability_Pr_state"
        ],
        "value": [
            str(input_path),
            PR_MW,
            VCI,
            VR,
            VCO,
            A,
            B,
            C,
            expanded["probability"].sum(),
            table3["probability"].sum(),
            len(expanded),
            len(table3),
            zero_probability,
            rated_probability
        ]
    })

    summary_path = OUT_DIR / "WTG_power_generation_model_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== 文件保存完成 ==========")
    print(f"100步风速-功率展开表：{expanded_path}")
    print(f"Table III 类似功率概率表：{table3_path}")
    print(f"统计摘要：{summary_path}")

    print("\n========== 关键结果 ==========")
    print(f"A = {A:.8f}")
    print(f"B = {B:.8f}")
    print(f"C = {C:.8f}")
    print(f"零出力概率 P0 = {zero_probability:.4f}")
    print(f"额定出力概率 = {rated_probability:.4f}")
    print(f"功率状态数 = {len(table3)}")

    print("\nTable III 类似结果预览：")
    print(table3.to_string(index=False))


# =========================================================
# 9. 主程序
# =========================================================

def main():
    print("========== 通用风速模型 + WTG功率曲线 ==========")
    print(f"Pr = {PR_MW} MW")
    print(f"Vci = {VCI} km/h")
    print(f"Vr  = {VR} km/h")
    print(f"Vco = {VCO} km/h")

    wind_df, input_path = load_site_wind_distribution()

    expanded, table3 = build_power_generation_model(wind_df)

    save_results(
        expanded=expanded,
        table3=table3,
        input_path=input_path
    )

    plot_wtg_power_curve()

    plot_power_probability_distribution(table3)

    print("\n========== WTG功率生成模型建立完成 ==========")


if __name__ == "__main__":
    main()