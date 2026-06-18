import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# 1. 基本路径设置
# =========================================================

BASE_DIR = Path(r"D:\tools\za\paper")

OUT_DIR = BASE_DIR / "RBTS_TableV_Fig6_From_Existing_Models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你已经建立好的通用风速模型结果文件夹
COMMON_RESULT_DIR = BASE_DIR / "Common_Wind_Speed_Model_Result"

# 直接引用你前面生成的 Fig.2 数据
# 里面包含：
# z_midpoint,
# North Battleford,
# Swift Current,
# Toronto Island A,
# Common Wind Speed Model
FIG2_SITE_MODELS_PATH = COMMON_RESULT_DIR / "Fig2_combining_wind_speed_models.csv"

# 直接引用你前面生成的 Fig.3 通用风速模型
COMMON_MODEL_PATH = COMMON_RESULT_DIR / "CommonWindSpeedModel_100step.csv"


# =========================================================
# 2. 三个地点配置
# =========================================================

SITES = {
    "North Battleford": {
        "site_key": "NorthBattleford",
        "fig2_column": "North Battleford",
        "result_dir": BASE_DIR / "North Battleford_Refit_ARMA_Result",
    },
    "Toronto": {
        "site_key": "TorontoIslandA",
        "fig2_column": "Toronto Island A",
        "result_dir": BASE_DIR / "Toronto Island A_Refit_ARMA_Result",
    },
    "Swift Current": {
        "site_key": "SwiftCurrent",
        "fig2_column": "Swift Current",
        "result_dir": BASE_DIR / "SwiftCurrent_Refit_ARMA_Result",
    },
}


# =========================================================
# 3. WTG 参数
# =========================================================

# 论文中使用 18 台 WTG
WTG_COUNT = 18

# 单台 WTG 额定功率
PR_SINGLE_MW = 1.5

# 风速参数，单位 km/h
VCI = 14.4
VR = 45.0
VCO = 90.0

# 风电场总额定功率
PR_WIND_FARM_MW = WTG_COUNT * PR_SINGLE_MW

# 功率状态聚合时保留的小数位
POWER_ROUND_DECIMALS = 4


# =========================================================
# 4. RBTS 参数
# =========================================================

RBTS_PEAK_LOAD_MW = 185.0
HOURS_PER_YEAR = 8760

# Fig.6 的峰值负荷水平
PEAK_LOAD_PU_LIST = [0.92, 0.95, 0.98, 1.00, 1.02, 1.05]

# LOLE 负荷持续曲线计算方式
# "step" 更接近论文中“离散阶梯负荷模型”的处理方式
# "linear" 使用线性插值
LDC_METHOD = "step"


# =========================================================
# 5. 检查关键输入文件
# =========================================================

def check_required_files():
    required_files = [
        FIG2_SITE_MODELS_PATH,
        COMMON_MODEL_PATH,
    ]

    for site_name, cfg in SITES.items():
        site_key = cfg["site_key"]
        result_dir = cfg["result_dir"]

        required_files.append(
            result_dir / f"{site_key}_actual_vs_refit_ARMA_summary.csv"
        )

    missing = [p for p in required_files if not p.exists()]

    if missing:
        print("\n========== 缺失文件 ==========")
        for p in missing:
            print(p)

        raise FileNotFoundError("上面这些文件不存在，请先确认前面 ARMA 和通用风速模型是否已经生成。")

    print("\n========== 已找到所有需要引用的前序结果文件 ==========")
    print(f"Fig.2 三地点标准化模型文件：{FIG2_SITE_MODELS_PATH}")
    print(f"Fig.3 通用风速模型文件：{COMMON_MODEL_PATH}")

    for site_name, cfg in SITES.items():
        site_key = cfg["site_key"]
        print(
            f"{site_name} 统计摘要："
            f"{cfg['result_dir'] / f'{site_key}_actual_vs_refit_ARMA_summary.csv'}"
        )


# =========================================================
# 6. WTG 功率曲线
# =========================================================

def calculate_abc(vci: float, vr: float):
    """
    计算论文式(9)二次功率曲线中的 A、B、C。
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


def single_wtg_power(sw_kmh: float) -> float:
    """
    单台 WTG 功率输出，单位 MW。
    """
    if sw_kmh < VCI:
        return 0.0

    elif VCI <= sw_kmh < VR:
        p = PR_SINGLE_MW * (A + B * sw_kmh + C * sw_kmh ** 2)
        return max(0.0, min(PR_SINGLE_MW, p))

    elif VR <= sw_kmh <= VCO:
        return PR_SINGLE_MW

    else:
        return 0.0


def wind_farm_power(sw_kmh: float) -> float:
    """
    18 台 WTG 风电场功率。
    论文中风电场内机组处于同一风速状态，所以直接 18 倍。
    """
    return WTG_COUNT * single_wtg_power(sw_kmh)


# =========================================================
# 7. RBTS 常规机组 COPT
# =========================================================

def build_rbts_conventional_copt():
    """
    根据论文 Table IV 建立 RBTS 常规机组容量概率模型。
    """
    unit_rows = [
        # unit_size_MW, number_of_units, FOR
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
                # 正常
                new_states[outage] = new_states.get(outage, 0.0) + prob * (1 - for_rate)

                # 强迫停运
                outage_new = outage + size
                new_states[outage_new] = new_states.get(outage_new, 0.0) + prob * for_rate

            outage_states = new_states

    copt = pd.DataFrame({
        "outage_capacity_MW": list(outage_states.keys()),
        "probability": list(outage_states.values())
    })

    copt["available_capacity_MW"] = total_capacity - copt["outage_capacity_MW"]

    copt = (
        copt
        .groupby("available_capacity_MW", as_index=False)["probability"]
        .sum()
        .sort_values("available_capacity_MW")
        .reset_index(drop=True)
    )

    copt["probability"] = copt["probability"] / copt["probability"].sum()

    out_path = OUT_DIR / "RBTS_conventional_COPT.csv"
    copt.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n========== RBTS 常规机组 COPT ==========")
    print(f"总装机容量：{total_capacity:.1f} MW")
    print(f"状态数：{len(copt)}")
    print(f"概率和：{copt['probability'].sum():.8f}")
    print(f"已保存：{out_path}")

    return copt


# =========================================================
# 8. RBTS 100 点负荷持续曲线
# =========================================================

def build_rbts_100point_ldc():
    """
    根据你提供的 RBTS 100 点负荷持续曲线表录入数据。
    peak_load_pu：负荷水平，按 RBTS 峰值负荷归一化。
    study_period_pu：负荷超过该水平的持续时间比例。
    """
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

    ldc = (
        ldc
        .sort_values("peak_load_pu", ascending=False)
        .reset_index(drop=True)
    )

    out_path = OUT_DIR / "RBTS_100point_load_duration_curve.csv"
    ldc.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n========== RBTS 100 点负荷持续曲线 ==========")
    print(f"点数：{len(ldc)}")
    print(f"最高负荷：{ldc['peak_load_pu'].max():.4f} p.u.")
    print(f"最低负荷：{ldc['peak_load_pu'].min():.4f} p.u.")
    print(f"已保存：{out_path}")

    return ldc


# =========================================================
# 9. 读取三地点 μ、σ
# =========================================================

def read_actual_mu_sigma(site_key: str, result_dir: Path):
    """
    从前面自拟合 ARMA 代码生成的 summary 中读取实际 1989-2003 均值和标准差。
    """
    summary_path = result_dir / f"{site_key}_actual_vs_refit_ARMA_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"找不到 {site_key} 的统计摘要文件：{summary_path}")

    df = pd.read_csv(summary_path, encoding="utf-8-sig")

    row = df[df["data_type"] == "Actual 1989-2003"]

    if row.empty:
        raise RuntimeError(f"{summary_path.name} 中找不到 data_type = Actual 1989-2003 的行")

    mu = float(row["mean_wind_speed_kmh"].iloc[0])
    sigma = float(row["std_wind_speed_kmh"].iloc[0])

    return mu, sigma


# =========================================================
# 10. 读取 site-specific ARMA 和 common wind model
# =========================================================

def load_fig2_site_specific_models():
    """
    读取 Fig.2 数据，其中每个地点列就是 site-specific ARMA 标准化概率分布。
    """
    df = pd.read_csv(FIG2_SITE_MODELS_PATH, encoding="utf-8-sig")

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    if "z_midpoint" not in df.columns:
        raise KeyError(f"{FIG2_SITE_MODELS_PATH.name} 缺少 z_midpoint 列")

    return df


def load_common_wind_model():
    """
    读取 Fig.3 的通用风速模型。
    """
    df = pd.read_csv(COMMON_MODEL_PATH, encoding="utf-8-sig")

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    required_cols = ["z_midpoint", "probability"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise KeyError(f"{COMMON_MODEL_PATH.name} 缺少列：{missing_cols}")

    df = df[["z_midpoint", "probability"]].copy()

    df["z_midpoint"] = pd.to_numeric(df["z_midpoint"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    df = df.dropna(subset=["z_midpoint", "probability"]).copy()

    df["probability"] = df["probability"] / df["probability"].sum()

    return df


def build_site_specific_wind_model_from_fig2(fig2_df: pd.DataFrame,
                                             fig2_column: str,
                                             mu: float,
                                             sigma: float):
    """
    路线 A：
    使用 Fig.2 中该地点自己的 ARMA 标准化概率分布。
    然后根据 SW = μ + zσ 转成该地点风速。
    """
    if fig2_column not in fig2_df.columns:
        raise KeyError(
            f"Fig.2 文件中找不到列：{fig2_column}\n"
            f"当前列名为：{list(fig2_df.columns)}"
        )

    df = fig2_df[["z_midpoint", fig2_column]].copy()

    df = df.rename(columns={fig2_column: "probability"})

    df["z_midpoint"] = pd.to_numeric(df["z_midpoint"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    df = df.dropna(subset=["z_midpoint", "probability"]).copy()

    df["wind_speed_kmh"] = mu + df["z_midpoint"] * sigma

    df = df[["z_midpoint", "wind_speed_kmh", "probability"]].copy()

    df["probability"] = df["probability"] / df["probability"].sum()

    return df


def build_common_wind_model_for_site(common_model: pd.DataFrame,
                                     mu: float,
                                     sigma: float):
    """
    路线 B：
    使用 Fig.3 通用风速模型。
    然后根据 SW = μ + zσ 转成该地点风速。
    """
    df = common_model.copy()

    df["wind_speed_kmh"] = mu + df["z_midpoint"] * sigma

    df = df[["z_midpoint", "wind_speed_kmh", "probability"]].copy()

    df["probability"] = df["probability"] / df["probability"].sum()

    return df


# =========================================================
# 11. 风速概率模型 -> 风电场功率模型
# =========================================================

def wind_model_to_wind_farm_power_model(wind_df: pd.DataFrame,
                                        model_name: str):
    """
    将风速概率模型转换成 18 台 WTG 风电场功率概率模型。
    """
    expanded = wind_df.copy()

    expanded["wind_farm_power_MW_raw"] = expanded["wind_speed_kmh"].apply(wind_farm_power)

    expanded["wind_farm_power_MW"] = (
        expanded["wind_farm_power_MW_raw"]
        .round(POWER_ROUND_DECIMALS)
    )

    power_model = (
        expanded
        .groupby("wind_farm_power_MW", as_index=False)
        .agg(
            probability=("probability", "sum"),
            z_min=("z_midpoint", "min"),
            z_max=("z_midpoint", "max"),
            wind_speed_min_kmh=("wind_speed_kmh", "min"),
            wind_speed_max_kmh=("wind_speed_kmh", "max"),
            wind_state_count=("wind_speed_kmh", "count")
        )
        .sort_values("wind_farm_power_MW")
        .reset_index(drop=True)
    )

    power_model["probability"] = power_model["probability"] / power_model["probability"].sum()

    expanded_path = OUT_DIR / f"{model_name}_wind_to_power_expanded.csv"
    power_path = OUT_DIR / f"{model_name}_wind_farm_power_model.csv"

    expanded.to_csv(expanded_path, index=False, encoding="utf-8-sig")
    power_model.to_csv(power_path, index=False, encoding="utf-8-sig")

    zero_prob = power_model.loc[
        power_model["wind_farm_power_MW"] == 0,
        "probability"
    ].sum()

    rated_prob = power_model.loc[
        power_model["wind_farm_power_MW"] == round(PR_WIND_FARM_MW, POWER_ROUND_DECIMALS),
        "probability"
    ].sum()

    print(f"\n========== {model_name} 风电场功率模型 ==========")
    print(f"风速状态数：{len(wind_df)}")
    print(f"功率状态数：{len(power_model)}")
    print(f"零出力概率：{zero_prob:.6f}")
    print(f"额定出力概率：{rated_prob:.6f}")
    print(f"概率和：{power_model['probability'].sum():.8f}")
    print(f"已保存：{power_path}")

    return power_model


# =========================================================
# 12. 常规机组 + 风电场卷积
# =========================================================

def combine_conventional_and_wind(conv_df: pd.DataFrame,
                                  wind_power_df: pd.DataFrame):
    """
    常规机组容量概率模型与风电场功率概率模型卷积。
    """
    rows = []

    for _, c in conv_df.iterrows():
        for _, w in wind_power_df.iterrows():
            rows.append({
                "available_capacity_MW": c["available_capacity_MW"] + w["wind_farm_power_MW"],
                "probability": c["probability"] * w["probability"]
            })

    combined = pd.DataFrame(rows)

    combined["available_capacity_MW"] = combined["available_capacity_MW"].round(POWER_ROUND_DECIMALS)

    combined = (
        combined
        .groupby("available_capacity_MW", as_index=False)["probability"]
        .sum()
        .sort_values("available_capacity_MW")
        .reset_index(drop=True)
    )

    combined["probability"] = combined["probability"] / combined["probability"].sum()

    return combined


# =========================================================
# 13. 用 RBTS 100 点负荷持续曲线计算 LOLE
# =========================================================

def loss_of_load_fraction(capacity_MW: float,
                          peak_load_MW: float,
                          ldc_df: pd.DataFrame):
    """
    给定可用容量 C，计算负荷超过 C 的时间比例。
    """
    threshold_pu = capacity_MW / peak_load_MW

    load_desc = ldc_df["peak_load_pu"].to_numpy()
    period_desc = ldc_df["study_period_pu"].to_numpy()

    if threshold_pu >= load_desc.max():
        return 0.0

    if threshold_pu <= load_desc.min():
        return 1.0

    if LDC_METHOD == "linear":
        load_asc = load_desc[::-1]
        period_asc = period_desc[::-1]

        return float(np.interp(threshold_pu, load_asc, period_asc))

    elif LDC_METHOD == "step":
        # 离散阶梯法：取所有负荷水平大于等于阈值的最大持续时间
        eligible = ldc_df[ldc_df["peak_load_pu"] >= threshold_pu]

        if eligible.empty:
            return 0.0

        return float(eligible["study_period_pu"].max())

    else:
        raise ValueError("LDC_METHOD 只能是 'step' 或 'linear'")


def calculate_lole(generation_model: pd.DataFrame,
                   ldc_df: pd.DataFrame,
                   peak_load_MW: float):
    """
    LOLE = 8760 × Σ P(C_i) × T(load > C_i)
    """
    lole = 0.0

    for _, row in generation_model.iterrows():
        capacity = row["available_capacity_MW"]
        probability = row["probability"]

        exceed_fraction = loss_of_load_fraction(
            capacity_MW=capacity,
            peak_load_MW=peak_load_MW,
            ldc_df=ldc_df
        )

        lole += probability * exceed_fraction * HOURS_PER_YEAR

    return lole


# =========================================================
# 14. 建立所有风电场模型
# =========================================================

def build_all_wind_models():
    fig2_df = load_fig2_site_specific_models()
    common_model = load_common_wind_model()

    all_power_models = {}
    site_mu_sigma_rows = []

    for site_name, cfg in SITES.items():
        site_key = cfg["site_key"]
        fig2_column = cfg["fig2_column"]
        result_dir = cfg["result_dir"]

        mu, sigma = read_actual_mu_sigma(site_key, result_dir)

        site_mu_sigma_rows.append({
            "site": site_name,
            "site_key": site_key,
            "mu_kmh": mu,
            "sigma_kmh": sigma
        })

        print(f"\n========== {site_name} μ / σ ==========")
        print(f"μ = {mu:.4f} km/h")
        print(f"σ = {sigma:.4f} km/h")

        # 路线 A：该地点自己的 ARMA 标准化概率模型
        site_specific_wind = build_site_specific_wind_model_from_fig2(
            fig2_df=fig2_df,
            fig2_column=fig2_column,
            mu=mu,
            sigma=sigma
        )

        site_specific_path = OUT_DIR / f"{site_key}_SiteSpecificARMA_wind_model_from_Fig2.csv"
        site_specific_wind.to_csv(site_specific_path, index=False, encoding="utf-8-sig")

        site_specific_power = wind_model_to_wind_farm_power_model(
            wind_df=site_specific_wind,
            model_name=f"{site_key}_SiteSpecificARMA"
        )

        # 路线 B：通用风速模型 + 该地点 μ / σ
        common_site_wind = build_common_wind_model_for_site(
            common_model=common_model,
            mu=mu,
            sigma=sigma
        )

        common_site_path = OUT_DIR / f"{site_key}_CommonWindModel_wind_model_from_Fig3.csv"
        common_site_wind.to_csv(common_site_path, index=False, encoding="utf-8-sig")

        common_power = wind_model_to_wind_farm_power_model(
            wind_df=common_site_wind,
            model_name=f"{site_key}_CommonWindModel"
        )

        all_power_models[(site_name, "SiteSpecificARMA")] = site_specific_power
        all_power_models[(site_name, "CommonWindModel")] = common_power

    site_mu_sigma = pd.DataFrame(site_mu_sigma_rows)

    site_mu_sigma_path = OUT_DIR / "three_sites_mu_sigma_used.csv"
    site_mu_sigma.to_csv(site_mu_sigma_path, index=False, encoding="utf-8-sig")

    print(f"\n三地点 μ / σ 已保存：{site_mu_sigma_path}")

    return all_power_models


# =========================================================
# 15. 复现 Table V
# =========================================================

def reproduce_table_v(conv_copt: pd.DataFrame,
                      ldc_df: pd.DataFrame,
                      all_power_models: dict):
    before_lole = calculate_lole(
        generation_model=conv_copt,
        ldc_df=ldc_df,
        peak_load_MW=RBTS_PEAK_LOAD_MW
    )

    row_before = {
        "Wind Model Used": "Before Adding WECS",
        "Before Adding WECS": before_lole,
        "North Battleford": np.nan,
        "Toronto": np.nan,
        "Swift Current": np.nan,
    }

    row_specific = {
        "Wind Model Used": "Site-specific ARMA model",
        "Before Adding WECS": np.nan,
    }

    row_common = {
        "Wind Model Used": "Common Wind Model",
        "Before Adding WECS": np.nan,
    }

    for site_name in SITES.keys():
        # ARMA 路线
        power_specific = all_power_models[(site_name, "SiteSpecificARMA")]

        gen_specific = combine_conventional_and_wind(
            conv_df=conv_copt,
            wind_power_df=power_specific
        )

        lole_specific = calculate_lole(
            generation_model=gen_specific,
            ldc_df=ldc_df,
            peak_load_MW=RBTS_PEAK_LOAD_MW
        )

        row_specific[site_name] = lole_specific

        # Common route
        power_common = all_power_models[(site_name, "CommonWindModel")]

        gen_common = combine_conventional_and_wind(
            conv_df=conv_copt,
            wind_power_df=power_common
        )

        lole_common = calculate_lole(
            generation_model=gen_common,
            ldc_df=ldc_df,
            peak_load_MW=RBTS_PEAK_LOAD_MW
        )

        row_common[site_name] = lole_common

        # 保存卷积后的系统容量模型
        site_key = SITES[site_name]["site_key"]

        gen_specific_path = OUT_DIR / f"{site_key}_system_generation_model_SiteSpecificARMA.csv"
        gen_common_path = OUT_DIR / f"{site_key}_system_generation_model_CommonWindModel.csv"

        gen_specific.to_csv(gen_specific_path, index=False, encoding="utf-8-sig")
        gen_common.to_csv(gen_common_path, index=False, encoding="utf-8-sig")

    table_v = pd.DataFrame([
        row_before,
        row_specific,
        row_common
    ])

    table_v_rounded = table_v.copy()

    for col in ["Before Adding WECS", "North Battleford", "Toronto", "Swift Current"]:
        table_v_rounded[col] = table_v_rounded[col].round(4)

    table_path = OUT_DIR / "TableV_LOLE_comparison.csv"
    table_rounded_path = OUT_DIR / "TableV_LOLE_comparison_rounded.csv"

    table_v.to_csv(table_path, index=False, encoding="utf-8-sig")
    table_v_rounded.to_csv(table_rounded_path, index=False, encoding="utf-8-sig")

    print("\n========== Table V 未四舍五入结果 ==========")
    print(table_v.to_string(index=False))

    print("\n========== Table V 四舍五入结果 ==========")
    print(table_v_rounded.to_string(index=False))

    print(f"\nTable V 已保存：{table_path}")
    print(f"Table V 四舍五入版已保存：{table_rounded_path}")

    return table_v


# =========================================================
# 16. 复现 Fig.6
# =========================================================

def reproduce_fig6(conv_copt: pd.DataFrame,
                   ldc_df: pd.DataFrame,
                   all_power_models: dict):
    site_name = "North Battleford"

    power_specific = all_power_models[(site_name, "SiteSpecificARMA")]
    power_common = all_power_models[(site_name, "CommonWindModel")]

    gen_specific = combine_conventional_and_wind(
        conv_df=conv_copt,
        wind_power_df=power_specific
    )

    gen_common = combine_conventional_and_wind(
        conv_df=conv_copt,
        wind_power_df=power_common
    )

    rows = []

    for peak_load_pu in PEAK_LOAD_PU_LIST:
        current_peak_load_MW = RBTS_PEAK_LOAD_MW * peak_load_pu

        lole_specific = calculate_lole(
            generation_model=gen_specific,
            ldc_df=ldc_df,
            peak_load_MW=current_peak_load_MW
        )

        lole_common = calculate_lole(
            generation_model=gen_common,
            ldc_df=ldc_df,
            peak_load_MW=current_peak_load_MW
        )

        rows.append({
            "peak_load_pu": peak_load_pu,
            "North_Battleford_ARMA_Model": lole_specific,
            "Common_Model": lole_common
        })

    fig6_df = pd.DataFrame(rows)

    fig6_csv_path = OUT_DIR / "Fig6_NorthBattleford_LOLE_vs_peak_load.csv"
    fig6_df.to_csv(fig6_csv_path, index=False, encoding="utf-8-sig")

    print("\n========== Fig.6 数据 ==========")
    print(fig6_df.round(4).to_string(index=False))
    print(f"Fig.6 数据已保存：{fig6_csv_path}")

    x = np.arange(len(fig6_df))
    width = 0.36

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        fig6_df["North_Battleford_ARMA_Model"],
        width,
        label="North Battleford ARMA Model"
    )

    plt.bar(
        x + width / 2,
        fig6_df["Common_Model"],
        width,
        label="Common Model"
    )

    plt.xticks(
        x,
        [str(v) for v in fig6_df["peak_load_pu"]]
    )

    plt.xlabel("Peak Load (in p.u. of RBTS peak load)")
    plt.ylabel("LOLE (hours/year)")
    plt.title("Fig.6 LOLE comparison for North Battleford")

    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig_path = OUT_DIR / "Fig6_NorthBattleford_LOLE_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"Fig.6 图像已保存：{fig_path}")


# =========================================================
# 17. 主程序
# =========================================================

def main():
    print("========== 复现 Table V 和 Fig.6：使用已有 ARMA 与通用风速模型文件 ==========")

    print("\n当前代码会明确读取这些前序文件：")
    print(f"1. 三地点标准化 ARMA 概率模型：{FIG2_SITE_MODELS_PATH}")
    print(f"2. 通用风速模型：{COMMON_MODEL_PATH}")
    print("3. 三地点 actual_vs_refit_ARMA_summary.csv 中的 μ、σ")

    check_required_files()

    print("\n========== WTG 参数 ==========")
    print(f"单台 Pr = {PR_SINGLE_MW} MW")
    print(f"WTG 数量 = {WTG_COUNT}")
    print(f"风电场 Pr = {PR_WIND_FARM_MW} MW")
    print(f"Vci = {VCI} km/h")
    print(f"Vr  = {VR} km/h")
    print(f"Vco = {VCO} km/h")
    print(f"A = {A:.8f}")
    print(f"B = {B:.8f}")
    print(f"C = {C:.8f}")

    print("\n========== LOLE 设置 ==========")
    print(f"RBTS 峰值负荷 = {RBTS_PEAK_LOAD_MW} MW")
    print(f"负荷持续曲线处理方式 LDC_METHOD = {LDC_METHOD}")

    # 1. 常规机组 COPT
    conv_copt = build_rbts_conventional_copt()

    # 2. RBTS 100 点负荷持续曲线
    ldc_df = build_rbts_100point_ldc()

    # 3. 建立三地点两种路线的风电场功率模型
    all_power_models = build_all_wind_models()

    # 4. 复现 Table V
    reproduce_table_v(
        conv_copt=conv_copt,
        ldc_df=ldc_df,
        all_power_models=all_power_models
    )

    # 5. 复现 Fig.6
    reproduce_fig6(
        conv_copt=conv_copt,
        ldc_df=ldc_df,
        all_power_models=all_power_models
    )

    print("\n========== 完成 ==========")
    print(f"所有输出结果保存在：{OUT_DIR}")


if __name__ == "__main__":
    main()