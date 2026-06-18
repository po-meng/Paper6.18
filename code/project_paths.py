# -*- coding: utf-8 -*-
"""
project_paths.py

统一管理整个 Paper6.18 项目的路径。

优点：
1. 不依赖电脑盘符
2. Git 同步后自动适配
3. 所有脚本统一引用
4. 后续新增模块无需重复写 BASE_DIR
"""

from pathlib import Path

# ==========================================================
# 项目根目录
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# 原始数据目录
# ==========================================================

REGINA_DIR = PROJECT_ROOT / "Regina"

SWIFTCURRENT_DIR = PROJECT_ROOT / "SwiftCurrent"

NORTH_BATTLEFORD_DIR = PROJECT_ROOT / "North Battleford"


# ==========================================================
# 清洗数据目录
# ==========================================================

REGINA_CLEANED_DIR = PROJECT_ROOT / "Regina_cleaned"

SWIFTCURRENT_CLEANED_DIR = PROJECT_ROOT / "SwiftCurrent_cleaned"

NORTH_BATTLEFORD_CLEANED_DIR = (
    PROJECT_ROOT / "North Battleford_cleaned"
)


# ==========================================================
# ARMA模型结果
# ==========================================================

REGINA_ARMA_RESULT_DIR = (
    PROJECT_ROOT / "Regina_ARMA_Result"
)

SWIFTCURRENT_ARMA_RESULT_DIR = (
    PROJECT_ROOT / "SwiftCurrent_ARMA_Result"
)

SWIFTCURRENT_REFIT_ARMA_RESULT_DIR = (
    PROJECT_ROOT / "SwiftCurrent_Refit_ARMA_Result"
)

NORTH_BATTLEFORD_ARMA_RESULT_DIR = (
    PROJECT_ROOT / "North Battleford_ARMA_Result"
)

NORTH_BATTLEFORD_REFIT_ARMA_RESULT_DIR = (
    PROJECT_ROOT / "North Battleford_Refit_ARMA_Result"
)





# ==========================================================
# TORONTO_ISLAND_A
# ==========================================================

TORONTO_ISLAND_A_DIR = PROJECT_ROOT / "Toronto Island A"

TORONTO_ISLAND_A_CLEANED_DIR = (
    PROJECT_ROOT / "Toronto Island A_cleaned"
)

TORONTO_ISLAND_A_ARMA_RESULT_DIR = (
    PROJECT_ROOT / "Toronto Island A_ARMA_Result"
)

TORONTO_ISLAND_A_REFIT_ARMA_RESULT_DIR = (
    PROJECT_ROOT / "Toronto Island A_Refit_ARMA_Result"
)
# ==========================================================
# Common Wind Speed Model
# ==========================================================

COMMON_WIND_SPEED_MODEL_RESULT_DIR = (
    PROJECT_ROOT / "Common_Wind_Speed_Model_Result"
)

APPLY_COMMON_WIND_SPEED_MODEL_RESULT_DIR = (
    PROJECT_ROOT / "Apply_Common_Wind_Speed_Model_Result"
)


# ==========================================================
# WTG功率模型
# ==========================================================

WTG_POWER_GENERATION_MODEL_RESULT_DIR = (
    PROJECT_ROOT / "WTG_Power_Generation_Model_Result"
)


# ==========================================================
# Table V & Fig.6
# ==========================================================

RBTS_TABLEV_FIG6_DIR = (
    PROJECT_ROOT / "RBTS_TableV_Fig6_From_Existing_Models"
)


# ==========================================================
# Fig.8 Fig.9 Fig.10 Fig.11
# ==========================================================

SIMPLIFIED_MULTISTATE_WTG_MODEL_RESULT_DIR = (
    PROJECT_ROOT / "Simplified_Multistate_WTG_Model_Result"
)


# ==========================================================
# Regina Fig.12
# ==========================================================

REGINA_FIG12_LOLE_RESULT_DIR = (
    PROJECT_ROOT / "Regina_Fig12_LOLE_Result"
)


# ==========================================================
# 常用文件
# ==========================================================

COMMON_WIND_MODEL_100STEP = (
    COMMON_WIND_SPEED_MODEL_RESULT_DIR
    / "CommonWindSpeedModel_100step.csv"
)

COMMON_WIND_MODEL_FIG2 = (
    COMMON_WIND_SPEED_MODEL_RESULT_DIR
    / "Fig2_combining_wind_speed_models.csv"
)

REGINA_MU_SIGMA_FILE = (
    REGINA_ARMA_RESULT_DIR
    / "Regina_actual_mu_sigma_summary.csv"
)

REGINA_ARMA100_MODEL_FILE = (
    REGINA_ARMA_RESULT_DIR
    / "Regina_ARMA100_wind_model.csv"
)


# ==========================================================
# 自动创建输出目录
# ==========================================================

OUTPUT_DIRS = [

    COMMON_WIND_SPEED_MODEL_RESULT_DIR,
    APPLY_COMMON_WIND_SPEED_MODEL_RESULT_DIR,

    WTG_POWER_GENERATION_MODEL_RESULT_DIR,

    REGINA_CLEANED_DIR,
    REGINA_ARMA_RESULT_DIR,

    SWIFTCURRENT_ARMA_RESULT_DIR,
    SWIFTCURRENT_REFIT_ARMA_RESULT_DIR,

    NORTH_BATTLEFORD_ARMA_RESULT_DIR,
    NORTH_BATTLEFORD_REFIT_ARMA_RESULT_DIR,

    TORONTO_ISLAND_A_ARMA_RESULT_DIR,
    TORONTO_ISLAND_A_REFIT_ARMA_RESULT_DIR,

    RBTS_TABLEV_FIG6_DIR,

    SIMPLIFIED_MULTISTATE_WTG_MODEL_RESULT_DIR,

    REGINA_FIG12_LOLE_RESULT_DIR,
]

for folder in OUTPUT_DIRS:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 调试
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print("\n关键目录检查：")

    print(
        "CommonWindSpeedModel_100step:",
        COMMON_WIND_MODEL_100STEP.exists()
    )

    print(
        "Regina mu sigma:",
        REGINA_MU_SIGMA_FILE.exists()
    )

    print(
        "Regina ARMA100:",
        REGINA_ARMA100_MODEL_FILE.exists()
    )

    print("=" * 60)