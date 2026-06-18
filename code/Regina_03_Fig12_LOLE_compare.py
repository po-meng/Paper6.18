# -*- coding: utf-8 -*-
r"""
Regina_03_Fig12_LOLE_compare.py

作用：
1. 读取第二份代码生成的 Regina_ARMA100_wind_model.csv，作为 Regina 100-step ARMA 基准模型；
2. 读取 CommonWindSpeedModel_100step.csv，并聚合生成 Regina 6-step simplified common model；
3. 注意：6-step common model 使用 Regina 2001-2003 实际计算得到的 mu/sigma，
   不再使用论文中的 mu=19.53、sigma=10.06；
4. 将两种风速模型转换为 18 台 WTG 风电场功率模型；
5. 与 RBTS 常规机组 COPT 卷积，计算不同峰值负荷下 LOLE，并绘制 Fig.12 严格复现/数据版对比图。

运行：
python Regina_03_Fig12_LOLE_compare.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 路径设置
# =========================================================
BASE_DIR = Path(r"D:\tools\za\paper")
COMMON_RESULT_DIR = BASE_DIR / "Common_Wind_Speed_Model_Result"
REGINA_RESULT_DIR = BASE_DIR / "Regina_ARMA_Result"
OUT_DIR = BASE_DIR / "Regina_Fig12_LOLE_Result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_MODEL_100STEP_PATH = COMMON_RESULT_DIR / "CommonWindSpeedModel_100step.csv"
REGINA_ARMA100_PATH = REGINA_RESULT_DIR / "Regina_ARMA100_wind_model.csv"
REGINA_MU_SIGMA_PATH = REGINA_RESULT_DIR / "Regina_actual_mu_sigma_summary.csv"


# =========================================================
# 2. WTG 与 RBTS 参数
# =========================================================
WTG_COUNT = 18
PR_SINGLE_MW = 1.5
VCI = 14.4
VR = 45.0
VCO = 90.0
POWER_ROUND_DECIMALS = 4

RBTS_PEAK_LOAD_MW = 185.0
HOURS_PER_YEAR = 8760
PEAK_LOAD_PU_LIST = [0.92, 0.95, 0.98, 1.00, 1.02, 1.05]
LDC_METHOD = "step"   # "step" 或 "linear"


# =========================================================
# 3. 输入检查与读取
# =========================================================
def check_required_files():
    required = [
        COMMON_MODEL_100STEP_PATH,
        REGINA_ARMA100_PATH,
        REGINA_MU_SIGMA_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n========== 缺失文件 ==========")
        for p in missing:
            print(p)
        raise FileNotFoundError(
            "请确认已经先运行：\n"
            "1. Regina_01_clean_hourly_data.py\n"
            "2. Regina_02_build_ARMA100_model.py\n"
            "并且 Common_Wind_Speed_Model_Result/CommonWindSpeedModel_100step.csv 已存在。"
        )

    print("\n========== 输入文件检查通过 ==========")
    print(f"Common 100-step 通用风速模型：{COMMON_MODEL_100STEP_PATH}")
    print(f"Regina ARMA 100-step 模型：{REGINA_ARMA100_PATH}")
    print(f"Regina 实际 μ/σ：{REGINA_MU_SIGMA_PATH}")


def read_regina_mu_sigma():
    df = pd.read_csv(REGINA_MU_SIGMA_PATH, encoding="utf-8-sig")
    row = df.iloc[0]
    mu = float(row["mean_wind_speed_kmh"])
    sigma = float(row["std_wind_speed_kmh"])
    return mu, sigma


def read_common_100step():
    df = pd.read_csv(COMMON_MODEL_100STEP_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    required = ["z_midpoint", "probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{COMMON_MODEL_100STEP_PATH.name} 缺少列：{missing}")
    df = df[required].copy()
    df["z_midpoint"] = pd.to_numeric(df["z_midpoint"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["probability"] = df["probability"] / df["probability"].sum()
    return df


def read_regina_arma100_wind_model():
    df = pd.read_csv(REGINA_ARMA100_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    required = ["z_midpoint", "wind_speed_kmh", "probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{REGINA_ARMA100_PATH.name} 缺少列：{missing}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["probability"] = df["probability"] / df["probability"].sum()
    return df


# =========================================================
# 4. 六步通用风速模型构造
# =========================================================
def build_six_step_common_standardized(common_100: pd.DataFrame) -> pd.DataFrame:
    """
    按论文 six-step 中点公式构造 z_midpoint：
    z_i = (i - 3) * 5/3, i=1,...,6
    概率由已有 100-step Common Wind Speed Model 聚合得到。
    """
    step_index = np.arange(1, 7)
    z_mid = (step_index - 3) * (5.0 / 3.0)

    # 用相邻中点的中间值作为分界，首尾用无穷扩展，保证概率和为 1。
    finite_edges = (z_mid[:-1] + z_mid[1:]) / 2.0
    edges = np.concatenate(([-np.inf], finite_edges, [np.inf]))

    z = common_100["z_midpoint"].to_numpy(dtype=float)
    p = common_100["probability"].to_numpy(dtype=float)

    bin_id = np.digitize(z, edges, right=False) - 1
    bin_id = np.clip(bin_id, 0, 5)

    prob = np.zeros(6)
    source_count = np.zeros(6, dtype=int)
    source_z_min = np.full(6, np.nan)
    source_z_max = np.full(6, np.nan)

    for k in range(6):
        mask = bin_id == k
        prob[k] = p[mask].sum()
        source_count[k] = int(mask.sum())
        if mask.any():
            source_z_min[k] = float(z[mask].min())
            source_z_max[k] = float(z[mask].max())

    prob = prob / prob.sum()

    six = pd.DataFrame({
        "step_index": step_index,
        "z_midpoint": z_mid,
        "probability": prob,
        "source_100step_count": source_count,
        "source_z_min": source_z_min,
        "source_z_max": source_z_max,
    })

    out_path = OUT_DIR / "Regina_Fig12_6step_common_standardized_model.csv"
    six.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"6-step 标准化通用模型已保存：{out_path}")

    return six


def apply_six_step_to_regina(six: pd.DataFrame, mu: float, sigma: float) -> pd.DataFrame:
    df = six.copy()
    df["wind_speed_raw_kmh"] = mu + df["z_midpoint"] * sigma
    df["wind_speed_kmh"] = df["wind_speed_raw_kmh"].clip(lower=0.0)
    df = df[["step_index", "z_midpoint", "wind_speed_raw_kmh", "wind_speed_kmh", "probability"]].copy()
    df["probability"] = df["probability"] / df["probability"].sum()

    out_path = OUT_DIR / "Regina_Fig12_6step_simplified_common_wind_model_actual_mu_sigma.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Regina 6-step simplified wind model 已保存：{out_path}")

    return df


# =========================================================
# 5. WTG 功率曲线与功率模型
# =========================================================
def calculate_abc(vci: float, vr: float):
    """计算论文式(9)二次功率曲线中的 A、B、C。"""
    k = (vci + vr) / (2 * vr)

    a = (vci * (vci + vr) - 4 * vci * vr * (k ** 3)) / ((vci - vr) ** 2)
    b = (4 * (vci + vr) * (k ** 3) - (3 * vci + vr)) / ((vci - vr) ** 2)
    c = (2 - 4 * (k ** 3)) / ((vci - vr) ** 2)

    return a, b, c


A, B, C = calculate_abc(VCI, VR)


def single_wtg_power(sw_kmh: float) -> float:
    if sw_kmh < VCI:
        return 0.0
    elif VCI <= sw_kmh < VR:
        p = PR_SINGLE_MW * (A + B * sw_kmh + C * sw_kmh ** 2)
        return max(0.0, min(PR_SINGLE_MW, p))
    elif VR <= sw_kmh <= VCO:
        return PR_SINGLE_MW
    else:
        return 0.0


def wind_farm_power(sw_kmh: float, wtg_count: int = WTG_COUNT) -> float:
    return wtg_count * single_wtg_power(sw_kmh)


def wind_model_to_power_model(wind_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    expanded = wind_df.copy()
    expanded["single_wtg_power_MW"] = expanded["wind_speed_kmh"].apply(single_wtg_power)
    expanded["wind_farm_power_MW_raw"] = expanded["wind_speed_kmh"].apply(lambda x: wind_farm_power(x, WTG_COUNT))
    expanded["wind_farm_power_MW"] = expanded["wind_farm_power_MW_raw"].round(POWER_ROUND_DECIMALS)

    power_model = (
        expanded
        .groupby("wind_farm_power_MW", as_index=False)
        .agg(
            probability=("probability", "sum"),
            wind_state_count=("wind_speed_kmh", "count"),
            z_min=("z_midpoint", "min"),
            z_max=("z_midpoint", "max"),
            wind_speed_min_kmh=("wind_speed_kmh", "min"),
            wind_speed_max_kmh=("wind_speed_kmh", "max"),
        )
        .sort_values("wind_farm_power_MW")
        .reset_index(drop=True)
    )

    power_model["probability"] = power_model["probability"] / power_model["probability"].sum()

    expanded_path = OUT_DIR / f"{model_name}_wind_to_power_expanded.csv"
    power_path = OUT_DIR / f"{model_name}_wind_farm_power_model.csv"
    expanded.to_csv(expanded_path, index=False, encoding="utf-8-sig")
    power_model.to_csv(power_path, index=False, encoding="utf-8-sig")

    print(f"\n========== {model_name} 风电场功率模型 ==========")
    print(f"风速状态数：{len(wind_df)}")
    print(f"功率状态数：{len(power_model)}")
    print(f"概率和：{power_model['probability'].sum():.8f}")
    print(f"功率模型保存：{power_path}")

    return power_model


# =========================================================
# 6. RBTS 常规机组 COPT 与负荷持续曲线
# =========================================================
def build_rbts_conventional_copt():
    unit_rows = [
        (5, 2, 0.010),
        (10, 1, 0.020),
        (20, 4, 0.015),
        (20, 1, 0.025),
        (40, 1, 0.020),
        (40, 2, 0.030),
    ]

    total_capacity = sum(size * num for size, num, _ in unit_rows)
    outage_states = {0.0: 1.0}

    for size, num, for_rate in unit_rows:
        for _ in range(num):
            new_states = {}
            for outage, prob in outage_states.items():
                new_states[outage] = new_states.get(outage, 0.0) + prob * (1 - for_rate)
                outage_new = outage + size
                new_states[outage_new] = new_states.get(outage_new, 0.0) + prob * for_rate
            outage_states = new_states

    copt = pd.DataFrame({
        "outage_capacity_MW": list(outage_states.keys()),
        "probability": list(outage_states.values()),
    })
    copt["available_capacity_MW"] = total_capacity - copt["outage_capacity_MW"]

    copt = (
        copt.groupby("available_capacity_MW", as_index=False)["probability"]
        .sum()
        .sort_values("available_capacity_MW")
        .reset_index(drop=True)
    )
    copt["probability"] = copt["probability"] / copt["probability"].sum()

    out_path = OUT_DIR / "RBTS_conventional_COPT.csv"
    copt.to_csv(out_path, index=False, encoding="utf-8-sig")
    return copt


def build_rbts_100point_ldc():
    data = [
        (1.0000, 0.0000),
        (0.9733, 0.0006),
        (0.9466, 0.0024),
        (0.9199, 0.0076),
        (0.8931, 0.0160),
        (0.8664, 0.0333),
        (0.8397, 0.0614),
        (0.8130, 0.1004),
        (0.7863, 0.1452),
        (0.7596, 0.1918),
        (0.7329, 0.2339),
        (0.7061, 0.2773),
        (0.6794, 0.3300),
        (0.6527, 0.3934),
        (0.6260, 0.4591),
        (0.5993, 0.5242),
        (0.5726, 0.5742),
        (0.5459, 0.6265),
        (0.5191, 0.6881),
        (0.4924, 0.7603),
        (0.4657, 0.8302),
        (0.4390, 0.8880),
        (0.4123, 0.9420),
        (0.3856, 0.9783),
        (0.3588, 0.9949),
        (0.9933, 0.0002),
        (0.9666, 0.0008),
        (0.9399, 0.0034),
        (0.9132, 0.0081),
        (0.8865, 0.0189),
        (0.8597, 0.0401),
        (0.8330, 0.0718),
        (0.8063, 0.1122),
        (0.7796, 0.1574),
        (0.7529, 0.2005),
        (0.7262, 0.2436),
        (0.6995, 0.2909),
        (0.6727, 0.3448),
        (0.6460, 0.4094),
        (0.6193, 0.4771),
        (0.5926, 0.5390),
        (0.5659, 0.5869),
        (0.5392, 0.6415),
        (0.5125, 0.7043),
        (0.4857, 0.7810),
        (0.4590, 0.8473),
        (0.4323, 0.9029),
        (0.4056, 0.9549),
        (0.3789, 0.9827),
        (0.3522, 0.9977),
        (0.9866, 0.0003),
        (0.9599, 0.0010),
        (0.9332, 0.0040),
        (0.9065, 0.0100),
        (0.8798, 0.0239),
        (0.8531, 0.0464),
        (0.8264, 0.0823),
        (0.7996, 0.1254),
        (0.7729, 0.1704),
        (0.7462, 0.2114),
        (0.7195, 0.2561),
        (0.6928, 0.3030),
        (0.6661, 0.3616),
        (0.6394, 0.4260),
        (0.6126, 0.4932),
        (0.5859, 0.5501),
        (0.5592, 0.5992),
        (0.5325, 0.6544),
        (0.5058, 0.7218),
        (0.4791, 0.7992),
        (0.4523, 0.8599),
        (0.4256, 0.9159),
        (0.3989, 0.9347),
        (0.3722, 0.9867),
        (0.3455, 0.9991),
        (0.9800, 0.0004),
        (0.9532, 0.0015),
        (0.9265, 0.0058),
        (0.8998, 0.0137),
        (0.8731, 0.0290),
        (0.8464, 0.0517),
        (0.8197, 0.0906),
        (0.7960, 0.1353),
        (0.7662, 0.1823),
        (0.7395, 0.2232),
        (0.7128, 0.2670),
        (0.6861, 0.3163),
        (0.6594, 0.3769),
        (0.6327, 0.4420),
        (0.6060, 0.5089),
        (0.5792, 0.5625),
        (0.5525, 0.6134),
        (0.5259, 0.6706),
        (0.4991, 0.7410),
        (0.4724, 0.8158),
        (0.4457, 0.8758),
        (0.4190, 0.9293),
        (0.3922, 0.9721),
        (0.3655, 0.9905),
        (0.3388, 1.0000),
    ]
    ldc = pd.DataFrame(data, columns=["peak_load_pu", "study_period_pu"])
    ldc = ldc.sort_values("peak_load_pu", ascending=False).reset_index(drop=True)
    out_path = OUT_DIR / "RBTS_100point_load_duration_curve.csv"
    ldc.to_csv(out_path, index=False, encoding="utf-8-sig")
    return ldc


# =========================================================
# 7. LOLE 计算
# =========================================================
def combine_conventional_and_wind(conv_df: pd.DataFrame, wind_power_df: pd.DataFrame):
    rows = []
    for _, c in conv_df.iterrows():
        for _, w in wind_power_df.iterrows():
            rows.append({
                "available_capacity_MW": c["available_capacity_MW"] + w["wind_farm_power_MW"],
                "probability": c["probability"] * w["probability"],
            })

    combined = pd.DataFrame(rows)
    combined["available_capacity_MW"] = combined["available_capacity_MW"].round(POWER_ROUND_DECIMALS)
    combined = (
        combined.groupby("available_capacity_MW", as_index=False)["probability"]
        .sum()
        .sort_values("available_capacity_MW")
        .reset_index(drop=True)
    )
    combined["probability"] = combined["probability"] / combined["probability"].sum()
    return combined


def loss_of_load_fraction(capacity_MW: float, peak_load_MW: float, ldc_df: pd.DataFrame):
    threshold_pu = capacity_MW / peak_load_MW

    load_desc = ldc_df["peak_load_pu"].to_numpy()
    if threshold_pu >= load_desc.max():
        return 0.0
    if threshold_pu <= load_desc.min():
        return 1.0

    if LDC_METHOD == "linear":
        load_asc = ldc_df["peak_load_pu"].to_numpy()[::-1]
        period_asc = ldc_df["study_period_pu"].to_numpy()[::-1]
        return float(np.interp(threshold_pu, load_asc, period_asc))

    if LDC_METHOD == "step":
        eligible = ldc_df[ldc_df["peak_load_pu"] >= threshold_pu]
        if eligible.empty:
            return 0.0
        return float(eligible["study_period_pu"].max())

    raise ValueError("LDC_METHOD 只能是 'step' 或 'linear'")


def calculate_lole(generation_model: pd.DataFrame, ldc_df: pd.DataFrame, peak_load_MW: float):
    lole = 0.0
    for _, row in generation_model.iterrows():
        exceed_fraction = loss_of_load_fraction(
            capacity_MW=row["available_capacity_MW"],
            peak_load_MW=peak_load_MW,
            ldc_df=ldc_df,
        )
        lole += row["probability"] * exceed_fraction * HOURS_PER_YEAR
    return lole


# =========================================================
# 8. 绘图与主程序
# =========================================================
def plot_fig12(results: pd.DataFrame):
    x = np.arange(len(results))
    width = 0.36

    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2,
        results["Regina_ARMA100_LOLE"],
        width,
        label="Regina ARMA 100-step Model",
    )
    plt.bar(
        x + width / 2,
        results["Regina_6step_simplified_LOLE"],
        width,
        label="Regina 6-step Simplified Model",
    )

    plt.xticks(x, [str(v) for v in results["peak_load_pu"]])
    plt.xlabel("Peak Load (in p.u. of RBTS peak load)")
    plt.ylabel("LOLE (hours/year)")
    plt.title("Fig.12 Regina 6-step Simplified Model vs ARMA 100-step Model")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig_path = OUT_DIR / "Fig12_Regina_6step_simplified_vs_ARMA100_LOLE.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Fig.12 对比图已保存：{fig_path}")


def main():
    print("========== Regina Fig.12：6-step simplified model vs ARMA 100-step model ==========")
    print(f"输出目录：{OUT_DIR}")

    check_required_files()

    mu, sigma = read_regina_mu_sigma()
    print("\n========== Regina 实际 2001-2003 μ / σ ==========")
    print(f"mu    = {mu:.6f} km/h")
    print(f"sigma = {sigma:.6f} km/h")
    print("说明：本代码中的 6-step Common Model 使用以上实际 μ/σ，不使用论文 19.53/10.06。")

    common_100 = read_common_100step()
    regina_arma100_wind = read_regina_arma100_wind_model()

    # 1. 构造 Regina 6-step simplified wind model
    six_standard = build_six_step_common_standardized(common_100)
    regina_six_wind = apply_six_step_to_regina(six_standard, mu, sigma)

    # 保存 ARMA100 风速模型副本到 Fig12 输出目录
    arma_wind_path = OUT_DIR / "Regina_Fig12_ARMA100_wind_model.csv"
    regina_arma100_wind.to_csv(arma_wind_path, index=False, encoding="utf-8-sig")
    print(f"Regina ARMA100 风速模型副本已保存：{arma_wind_path}")

    # 2. 转换为风电场功率模型
    arma_power = wind_model_to_power_model(
        regina_arma100_wind,
        model_name="Regina_Fig12_ARMA100_18WTG",
    )
    six_power = wind_model_to_power_model(
        regina_six_wind,
        model_name="Regina_Fig12_6step_simplified_18WTG",
    )

    # 3. RBTS COPT 与 LDC
    conv_copt = build_rbts_conventional_copt()
    ldc_df = build_rbts_100point_ldc()

    # 4. 常规机组 + 风电卷积
    gen_arma = combine_conventional_and_wind(conv_copt, arma_power)
    gen_six = combine_conventional_and_wind(conv_copt, six_power)

    gen_arma_path = OUT_DIR / "Regina_Fig12_system_generation_ARMA100.csv"
    gen_six_path = OUT_DIR / "Regina_Fig12_system_generation_6step_simplified.csv"
    gen_arma.to_csv(gen_arma_path, index=False, encoding="utf-8-sig")
    gen_six.to_csv(gen_six_path, index=False, encoding="utf-8-sig")

    # 5. 计算 LOLE
    rows = []
    for pu in PEAK_LOAD_PU_LIST:
        peak_load = RBTS_PEAK_LOAD_MW * pu
        lole_arma = calculate_lole(gen_arma, ldc_df, peak_load)
        lole_six = calculate_lole(gen_six, ldc_df, peak_load)
        rows.append({
            "site": "Regina",
            "wtg_count": WTG_COUNT,
            "wind_capacity_MW": WTG_COUNT * PR_SINGLE_MW,
            "mu_used_for_six_step_kmh": mu,
            "sigma_used_for_six_step_kmh": sigma,
            "peak_load_pu": pu,
            "Regina_ARMA100_LOLE": lole_arma,
            "Regina_6step_simplified_LOLE": lole_six,
            "absolute_error_6step_minus_ARMA": lole_six - lole_arma,
            "relative_error_percent": (lole_six - lole_arma) / lole_arma * 100 if lole_arma != 0 else np.nan,
        })

    results = pd.DataFrame(rows)
    results_path = OUT_DIR / "Fig12_Regina_6step_simplified_vs_ARMA100_LOLE.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    print("\n========== Fig.12 Regina LOLE 对比结果 ==========")
    print(results.round(6).to_string(index=False))
    print(f"Fig.12 LOLE 数据已保存：{results_path}")

    plot_fig12(results)

    print("\n========== 完成 ==========")
    print(f"全部输出保存在：{OUT_DIR}")


if __name__ == "__main__":
    main()
