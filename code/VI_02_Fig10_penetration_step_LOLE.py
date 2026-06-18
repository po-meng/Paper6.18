# -*- coding: utf-8 -*-
"""
第 VI 部分：Simplified Multistate WTG Model 复现通用函数
说明：本脚本为完整独立脚本的一部分，不依赖 LOLE_comparison.py。
你只需要保证 BASE_DIR 指向你的工程根目录，并且前序文件已经生成：
1) Common_Wind_Speed_Model_Result/Fig2_combining_wind_speed_models.csv
2) Common_Wind_Speed_Model_Result/CommonWindSpeedModel_100step.csv
3) 三个地点 *_actual_vs_refit_ARMA_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from project_paths import *


# =========================================================
# 1. 基本路径设置
# =========================================================

OUT_DIR = SIMPLIFIED_MULTISTATE_WTG_MODEL_RESULT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_RESULT_DIR = COMMON_WIND_SPEED_MODEL_RESULT_DIR
FIG2_SITE_MODELS_PATH = COMMON_WIND_MODEL_FIG2
COMMON_MODEL_PATH = COMMON_WIND_MODEL_100STEP

SITES = {
    "North Battleford": {
        "site_key": "NorthBattleford",
        "fig2_column": "North Battleford",
        "result_dir": NORTH_BATTLEFORD_REFIT_ARMA_RESULT_DIR,
    },
    "Toronto": {
        "site_key": "TorontoIslandA",
        "fig2_column": "Toronto Island A",
        "result_dir": TORONTO_ISLAND_A_REFIT_ARMA_RESULT_DIR,
    },
    "Swift Current": {
        "site_key": "SwiftCurrent",
        "fig2_column": "Swift Current",
        "result_dir": SWIFTCURRENT_REFIT_ARMA_RESULT_DIR,
    },
}

# Regina 示例参数，来自论文 Table VI 前的说明
REGINA_MU_KMH = 19.53
REGINA_SIGMA_KMH = 10.06

# =========================================================
# 2. WTG / RBTS 参数
# =========================================================

PR_SINGLE_MW = 1.5
VCI = 14.4
VR = 45.0
VCO = 90.0
POWER_ROUND_DECIMALS = 4

RBTS_PEAK_LOAD_MW = 185.0
HOURS_PER_YEAR = 8760
PEAK_LOAD_PU_LIST = [0.92, 0.95, 0.98, 1.00, 1.02, 1.05]
LDC_METHOD = "step"   # 可改为 "linear"

STEP_LIST = [100, 10, 6, 5, 2]


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("-", "_")


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. 文件检查与读取
# =========================================================

def check_required_files(require_fig2=True, require_common=True, require_summary=True):
    required_files = []
    if require_fig2:
        required_files.append(FIG2_SITE_MODELS_PATH)
    if require_common:
        required_files.append(COMMON_MODEL_PATH)
    if require_summary:
        for _, cfg in SITES.items():
            site_key = cfg["site_key"]
            required_files.append(cfg["result_dir"] / f"{site_key}_actual_vs_refit_ARMA_summary.csv")

    missing = [p for p in required_files if not p.exists()]
    if missing:
        print("\n========== 缺失文件 ==========")
        for p in missing:
            print(p)
        raise FileNotFoundError("上面这些文件不存在，请先运行前面的 ARMA、Common Wind Model 和 Table V/Fig.6 代码。")

    print("\n========== 输入文件检查通过 ==========")
    if require_fig2:
        print(f"Fig.2 三地点标准化模型：{FIG2_SITE_MODELS_PATH}")
    if require_common:
        print(f"Fig.3 100-step 通用风速模型：{COMMON_MODEL_PATH}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def load_common_wind_model_100step() -> pd.DataFrame:
    df = pd.read_csv(COMMON_MODEL_PATH, encoding="utf-8-sig")
    df = normalize_columns(df)
    required_cols = ["z_midpoint", "probability"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{COMMON_MODEL_PATH.name} 缺少列：{missing}")
    df = df[required_cols].copy()
    df["z_midpoint"] = pd.to_numeric(df["z_midpoint"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.dropna(subset=["z_midpoint", "probability"]).sort_values("z_midpoint").reset_index(drop=True)
    df["probability"] = df["probability"] / df["probability"].sum()
    return df


def load_fig2_site_specific_models() -> pd.DataFrame:
    df = pd.read_csv(FIG2_SITE_MODELS_PATH, encoding="utf-8-sig")
    df = normalize_columns(df)
    if "z_midpoint" not in df.columns:
        raise KeyError(f"{FIG2_SITE_MODELS_PATH.name} 缺少 z_midpoint 列")
    return df


def read_actual_mu_sigma(site_name: str):
    cfg = SITES[site_name]
    site_key = cfg["site_key"]
    summary_path = cfg["result_dir"] / f"{site_key}_actual_vs_refit_ARMA_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到统计摘要文件：{summary_path}")
    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    df = normalize_columns(df)
    row = df[df["data_type"] == "Actual 1989-2003"]
    if row.empty:
        raise RuntimeError(f"{summary_path.name} 中找不到 data_type = Actual 1989-2003 的行")
    mu = float(row["mean_wind_speed_kmh"].iloc[0])
    sigma = float(row["std_wind_speed_kmh"].iloc[0])
    return mu, sigma


# =========================================================
# 4. 通用风速模型离散化
# =========================================================

def target_z_midpoints(nb: int) -> np.ndarray:
    """
    论文式(7) / 式(12)的标准化中点。
    even Nb: z_i = (10/Nb) * (i - 0.5*Nb), i=1..Nb
    odd  Nb: z_i = (10/Nb) * (i - 0.5*(Nb+1)), i=1..Nb
    当 Nb=6 时，得到 [-10/3, -5/3, 0, 5/3, 10/3, 5]，与论文式(12)一致。
    """
    i = np.arange(1, nb + 1, dtype=float)
    if nb % 2 == 0:
        return (10.0 / nb) * (i - 0.5 * nb)
    return (10.0 / nb) * (i - 0.5 * (nb + 1))


def rebin_probability_model(base_df: pd.DataFrame, nb: int, prob_col: str = "probability") -> pd.DataFrame:
    """
    将已有 100-step 概率模型聚合为 nb-step。
    由于你当前工程已经有 100-step 概率表，这里采用“最近中点归并”的方式，
    可以保证所有尾部概率都被保留，不会因为区间边界造成概率损失。
    """
    df = base_df[["z_midpoint", prob_col]].copy()
    df = df.rename(columns={prob_col: "probability"})
    df["z_midpoint"] = pd.to_numeric(df["z_midpoint"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.dropna(subset=["z_midpoint", "probability"]).copy()
    df["probability"] = df["probability"] / df["probability"].sum()

    if nb == len(df):
        out = df.sort_values("z_midpoint").reset_index(drop=True)
        out.insert(0, "step_index", np.arange(1, len(out) + 1))
        out["probability"] = out["probability"] / out["probability"].sum()
        return out[["step_index", "z_midpoint", "probability"]]

    target = target_z_midpoints(nb)
    z = df["z_midpoint"].to_numpy()
    p = df["probability"].to_numpy()
    nearest = np.abs(z[:, None] - target[None, :]).argmin(axis=1)

    prob = np.zeros(nb)
    source_state_count = np.zeros(nb, dtype=int)
    z_min = np.full(nb, np.nan)
    z_max = np.full(nb, np.nan)

    for k in range(nb):
        mask = nearest == k
        prob[k] = p[mask].sum()
        source_state_count[k] = int(mask.sum())
        if mask.any():
            z_min[k] = float(z[mask].min())
            z_max[k] = float(z[mask].max())

    if prob.sum() <= 0:
        raise RuntimeError(f"{nb}-step 模型聚合后概率和为 0，请检查输入概率表。")
    prob = prob / prob.sum()

    return pd.DataFrame({
        "step_index": np.arange(1, nb + 1),
        "z_midpoint": target,
        "probability": prob,
        "source_100step_count": source_state_count,
        "source_z_min": z_min,
        "source_z_max": z_max,
    })


def build_common_step_models(out_dir: Path = OUT_DIR) -> dict:
    base = load_common_wind_model_100step()
    models = {}
    rows = []
    for nb in STEP_LIST:
        model = rebin_probability_model(base, nb, prob_col="probability")
        model["probability"] = model["probability"] / model["probability"].sum()
        models[nb] = model
        path = out_dir / f"CommonWindSpeedModel_{nb}step.csv"
        model.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append({
            "step_number": nb,
            "state_count": len(model),
            "probability_sum": model["probability"].sum(),
            "csv_path": str(path),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "CommonWindSpeedModel_step_summary.csv", index=False, encoding="utf-8-sig")
    return models


def build_common_wind_model_for_site(step_model: pd.DataFrame, mu: float, sigma: float) -> pd.DataFrame:
    df = step_model.copy()
    df["wind_speed_raw_kmh"] = mu + df["z_midpoint"] * sigma
    df["wind_speed_kmh"] = df["wind_speed_raw_kmh"].clip(lower=0.0)
    df["probability"] = df["probability"] / df["probability"].sum()
    return df[["step_index", "z_midpoint", "wind_speed_raw_kmh", "wind_speed_kmh", "probability"]].copy()


def build_site_specific_wind_model_from_fig2(site_name: str) -> pd.DataFrame:
    fig2 = load_fig2_site_specific_models()
    cfg = SITES[site_name]
    col = cfg["fig2_column"]
    if col not in fig2.columns:
        raise KeyError(f"Fig.2 文件中找不到列：{col}；当前列为：{list(fig2.columns)}")
    mu, sigma = read_actual_mu_sigma(site_name)
    df = fig2[["z_midpoint", col]].copy().rename(columns={col: "probability"})
    df["z_midpoint"] = pd.to_numeric(df["z_midpoint"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.dropna(subset=["z_midpoint", "probability"]).copy()
    df["probability"] = df["probability"] / df["probability"].sum()
    df.insert(0, "step_index", np.arange(1, len(df) + 1))
    df["wind_speed_raw_kmh"] = mu + df["z_midpoint"] * sigma
    df["wind_speed_kmh"] = df["wind_speed_raw_kmh"].clip(lower=0.0)
    return df[["step_index", "z_midpoint", "wind_speed_raw_kmh", "wind_speed_kmh", "probability"]].copy()


# =========================================================
# 5. WTG 功率曲线与风电场功率模型
# =========================================================

def calculate_abc(vci: float, vr: float):
    k = (vci + vr) / (2 * vr)
    a = (vci * (vci + vr) - 4 * vci * vr * (k ** 3)) / ((vci - vr) ** 2)
    b = (4 * (vci + vr) * (k ** 3) - (3 * vci + vr)) / ((vci - vr) ** 2)
    c = (2 - 4 * (k ** 3)) / ((vci - vr) ** 2)
    return a, b, c


A, B, C = calculate_abc(VCI, VR)


def single_wtg_power(sw_kmh: float) -> float:
    if sw_kmh < VCI:
        return 0.0
    if VCI <= sw_kmh < VR:
        p = PR_SINGLE_MW * (A + B * sw_kmh + C * sw_kmh ** 2)
        return max(0.0, min(PR_SINGLE_MW, p))
    if VR <= sw_kmh <= VCO:
        return PR_SINGLE_MW
    return 0.0


def wind_farm_power(sw_kmh: float, wtg_count: int) -> float:
    return wtg_count * single_wtg_power(sw_kmh)


def wind_model_to_power_model(wind_df: pd.DataFrame,
                              wtg_count: int,
                              model_name: str,
                              out_dir: Path = OUT_DIR,
                              save_detail: bool = True) -> pd.DataFrame:
    expanded = wind_df.copy()
    expanded["single_wtg_power_MW"] = expanded["wind_speed_kmh"].apply(single_wtg_power)
    expanded["wind_farm_power_MW_raw"] = expanded["single_wtg_power_MW"] * wtg_count
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

    if save_detail:
        expanded.to_csv(out_dir / f"{model_name}_wind_to_power_expanded.csv", index=False, encoding="utf-8-sig")
        power_model.to_csv(out_dir / f"{model_name}_wind_farm_power_model.csv", index=False, encoding="utf-8-sig")

    return power_model


# =========================================================
# 6. RBTS COPT、负荷持续曲线、LOLE
# =========================================================

def build_rbts_conventional_copt(out_dir: Path = OUT_DIR) -> pd.DataFrame:
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
        copt.groupby("available_capacity_MW", as_index=False)["probability"].sum()
        .sort_values("available_capacity_MW")
        .reset_index(drop=True)
    )
    copt["probability"] = copt["probability"] / copt["probability"].sum()
    copt.to_csv(out_dir / "RBTS_conventional_COPT.csv", index=False, encoding="utf-8-sig")
    return copt


def build_rbts_100point_ldc(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    data = [
        (1.0000, 0.0000), (0.9733, 0.0006), (0.9466, 0.0024), (0.9199, 0.0076), (0.8931, 0.0160),
        (0.8664, 0.0333), (0.8397, 0.0614), (0.8130, 0.1004), (0.7863, 0.1452), (0.7596, 0.1918),
        (0.7329, 0.2339), (0.7061, 0.2773), (0.6794, 0.3300), (0.6527, 0.3934), (0.6260, 0.4591),
        (0.5993, 0.5242), (0.5726, 0.5742), (0.5459, 0.6265), (0.5191, 0.6881), (0.4924, 0.7603),
        (0.4657, 0.8302), (0.4390, 0.8880), (0.4123, 0.9420), (0.3856, 0.9783), (0.3588, 0.9949),
        (0.9933, 0.0002), (0.9666, 0.0008), (0.9399, 0.0034), (0.9132, 0.0081), (0.8865, 0.0189),
        (0.8597, 0.0401), (0.8330, 0.0718), (0.8063, 0.1122), (0.7796, 0.1574), (0.7529, 0.2005),
        (0.7262, 0.2436), (0.6995, 0.2909), (0.6727, 0.3448), (0.6460, 0.4094), (0.6193, 0.4771),
        (0.5926, 0.5390), (0.5659, 0.5869), (0.5392, 0.6415), (0.5125, 0.7043), (0.4857, 0.7810),
        (0.4590, 0.8473), (0.4323, 0.9029), (0.4056, 0.9549), (0.3789, 0.9827), (0.3522, 0.9977),
        (0.9866, 0.0003), (0.9599, 0.0010), (0.9332, 0.0040), (0.9065, 0.0100), (0.8798, 0.0239),
        (0.8531, 0.0464), (0.8264, 0.0823), (0.7996, 0.1254), (0.7729, 0.1704), (0.7462, 0.2114),
        (0.7195, 0.2561), (0.6928, 0.3030), (0.6661, 0.3616), (0.6394, 0.4260), (0.6126, 0.4932),
        (0.5859, 0.5501), (0.5592, 0.5992), (0.5325, 0.6544), (0.5058, 0.7218), (0.4791, 0.7992),
        (0.4523, 0.8599), (0.4256, 0.9159), (0.3989, 0.9347), (0.3722, 0.9867), (0.3455, 0.9991),
        (0.9800, 0.0004), (0.9532, 0.0015), (0.9265, 0.0058), (0.8998, 0.0137), (0.8731, 0.0290),
        (0.8464, 0.0517), (0.8197, 0.0906), (0.7960, 0.1353), (0.7662, 0.1823), (0.7395, 0.2232),
        (0.7128, 0.2670), (0.6861, 0.3163), (0.6594, 0.3769), (0.6327, 0.4420), (0.6060, 0.5089),
        (0.5792, 0.5625), (0.5525, 0.6134), (0.5259, 0.6706), (0.4991, 0.7410), (0.4724, 0.8158),
        (0.4457, 0.8758), (0.4190, 0.9293), (0.3922, 0.9721), (0.3655, 0.9905), (0.3388, 1.0000),
    ]
    ldc = pd.DataFrame(data, columns=["peak_load_pu", "study_period_pu"])
    ldc = ldc.sort_values("peak_load_pu", ascending=False).reset_index(drop=True)
    ldc.to_csv(out_dir / "RBTS_100point_load_duration_curve.csv", index=False, encoding="utf-8-sig")
    return ldc


def combine_conventional_and_wind(conv_df: pd.DataFrame, wind_power_df: pd.DataFrame) -> pd.DataFrame:
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
        combined.groupby("available_capacity_MW", as_index=False)["probability"].sum()
        .sort_values("available_capacity_MW")
        .reset_index(drop=True)
    )
    combined["probability"] = combined["probability"] / combined["probability"].sum()
    return combined


def loss_of_load_fraction(capacity_MW: float, peak_load_MW: float, ldc_df: pd.DataFrame) -> float:
    threshold_pu = capacity_MW / peak_load_MW
    load_desc = ldc_df["peak_load_pu"].to_numpy()
    if threshold_pu >= load_desc.max():
        return 0.0
    if threshold_pu <= load_desc.min():
        return 1.0
    if LDC_METHOD == "linear":
        load_asc = load_desc[::-1]
        period_asc = ldc_df["study_period_pu"].to_numpy()[::-1]
        return float(np.interp(threshold_pu, load_asc, period_asc))
    eligible = ldc_df[ldc_df["peak_load_pu"] >= threshold_pu]
    if eligible.empty:
        return 0.0
    return float(eligible["study_period_pu"].max())


def calculate_lole(generation_model: pd.DataFrame, ldc_df: pd.DataFrame, peak_load_MW: float) -> float:
    lole = 0.0
    for _, row in generation_model.iterrows():
        exceed_fraction = loss_of_load_fraction(row["available_capacity_MW"], peak_load_MW, ldc_df)
        lole += row["probability"] * exceed_fraction * HOURS_PER_YEAR
    return float(lole)


def calculate_lole_for_wind_power_model(conv_copt: pd.DataFrame,
                                        ldc_df: pd.DataFrame,
                                        wind_power_model: pd.DataFrame,
                                        peak_load_pu: float) -> float:
    system_model = combine_conventional_and_wind(conv_copt, wind_power_model)
    return calculate_lole(system_model, ldc_df, RBTS_PEAK_LOAD_MW * peak_load_pu)


def plot_lines(df: pd.DataFrame,
               x_col: str,
               y_cols: list,
               title: str,
               xlabel: str,
               ylabel: str,
               fig_path: Path):
    plt.figure(figsize=(8.5, 5.2))
    for col in y_cols:
        plt.plot(df[x_col], df[col], marker="o", linewidth=1.8, label=col)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


# =========================================================
# 代码 2：复现 Fig.10，并补充渗透率敏感性
# 功能：Swift Current，15% 风电渗透率（28 台 WTG），比较 100/10/6/5/2 step
# 说明：论文 Fig.10 为 Swift Current + 28 WTG；额外输出 1%/10%/15% 汇总表，方便报告讨论。
# =========================================================


def run_penetration_case(site_name: str, wtg_count: int, penetration_label: str) -> pd.DataFrame:
    mu, sigma = read_actual_mu_sigma(site_name)
    print(f"\n========== {site_name}：{penetration_label} wind penetration，WTG = {wtg_count} ==========")
    print(f"mu = {mu:.6f} km/h, sigma = {sigma:.6f} km/h")

    conv_copt = build_rbts_conventional_copt()
    ldc_df = build_rbts_100point_ldc()
    step_models = build_common_step_models()

    rows = []
    for nb in STEP_LIST:
        wind_df = build_common_wind_model_for_site(step_models[nb], mu, sigma)
        wind_df.to_csv(
            OUT_DIR / f"Penetration_{safe_name(site_name)}_{penetration_label}_{nb}step_wind_model.csv",
            index=False,
            encoding="utf-8-sig",
        )

        power_model = wind_model_to_power_model(
            wind_df=wind_df,
            wtg_count=wtg_count,
            model_name=f"Penetration_{safe_name(site_name)}_{penetration_label}_{nb}step_{wtg_count}WTG",
            out_dir=OUT_DIR,
            save_detail=True,
        )

        for peak_pu in PEAK_LOAD_PU_LIST:
            rows.append({
                "site": site_name,
                "wtg_count": wtg_count,
                "wind_capacity_MW": wtg_count * PR_SINGLE_MW,
                "wind_penetration_label": penetration_label,
                "step_number": nb,
                "peak_load_pu": peak_pu,
                "LOLE_hours_per_year": calculate_lole_for_wind_power_model(
                    conv_copt=conv_copt,
                    ldc_df=ldc_df,
                    wind_power_model=power_model,
                    peak_load_pu=peak_pu,
                ),
            })

    result = pd.DataFrame(rows)
    result.to_csv(
        OUT_DIR / f"Penetration_{safe_name(site_name)}_{penetration_label}_step_LOLE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def plot_fig10(result: pd.DataFrame):
    pivot = result.pivot(index="peak_load_pu", columns="step_number", values="LOLE_hours_per_year").reset_index()
    pivot.columns = ["peak_load_pu"] + [f"{int(c)}-step" for c in pivot.columns[1:]]
    pivot_path = OUT_DIR / "Fig10_SwiftCurrent_15pct_28WTG_step_LOLE_pivot.csv"
    pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")

    fig_path = OUT_DIR / "Fig10_SwiftCurrent_15pct_28WTG_step_LOLE.png"
    plot_lines(
        df=pivot,
        x_col="peak_load_pu",
        y_cols=[f"{nb}-step" for nb in STEP_LIST],
        title="Fig.10: Swift Current wind site with 15% wind penetration (28 WTG)",
        xlabel="Peak Load (p.u. of RBTS peak load)",
        ylabel="LOLE (hours/year)",
        fig_path=fig_path,
    )
    print(f"Fig.10 数据已保存：{pivot_path}")
    print(f"Fig.10 图像已保存：{fig_path}")
    print(pivot.round(4).to_string(index=False))


def build_penetration_error_summary(all_df: pd.DataFrame):
    """
    以 100-step 为基准，计算不同 step 数相对误差。
    """
    rows = []
    for (label, peak), g in all_df.groupby(["wind_penetration_label", "peak_load_pu"]):
        base = g.loc[g["step_number"] == 100, "LOLE_hours_per_year"]
        if base.empty:
            continue
        base_val = float(base.iloc[0])
        for _, r in g.iterrows():
            err = r["LOLE_hours_per_year"] - base_val
            rel = np.nan if abs(base_val) < 1e-12 else err / base_val * 100.0
            rows.append({
                "wind_penetration_label": label,
                "peak_load_pu": peak,
                "step_number": int(r["step_number"]),
                "LOLE_hours_per_year": r["LOLE_hours_per_year"],
                "benchmark_100step_LOLE": base_val,
                "absolute_error_vs_100step": err,
                "relative_error_vs_100step_percent": rel,
            })
    summary = pd.DataFrame(rows)
    path = OUT_DIR / "Penetration_step_error_summary_vs_100step.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"步数误差汇总已保存：{path}")


def main():
    print("========== 代码2：复现 Fig.10：Swift Current 15% 风电渗透率 step 数敏感性 ==========")
    print(f"输出目录：{OUT_DIR}")
    check_required_files(require_fig2=False, require_common=True, require_summary=True)

    # 论文 Fig.10：Swift Current，28 台 WTG，约 15% penetration
    fig10_df = run_penetration_case("Swift Current", wtg_count=28, penetration_label="15pct")
    plot_fig10(fig10_df)

    # 论文文字中提到 1% penetration 基本不受 step 数影响，这里作为补充输出。
    extra_1pct = run_penetration_case("Swift Current", wtg_count=2, penetration_label="1pct_optional")
    extra_10pct = run_penetration_case("Swift Current", wtg_count=18, penetration_label="10pct_reference")

    all_df = pd.concat([extra_1pct, extra_10pct, fig10_df], ignore_index=True)
    all_path = OUT_DIR / "SwiftCurrent_penetration_1pct_10pct_15pct_step_LOLE_all.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    build_penetration_error_summary(all_df)

    print("\n========== 完成 ==========")
    print(f"渗透率汇总结果已保存：{all_path}")


if __name__ == "__main__":
    main()
