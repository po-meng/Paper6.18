# -*- coding: utf-8 -*-
r"""
Regina_01_clean_hourly_data.py

作用：
1. 读取 D:\tools\za\paper\Regina 下已经下载好的 Regina 2001-2003 每时风速 csv 文件；
2. 自动识别 Canadian Climate Data 常见列名；
3. 清洗 Wind Spd (km/h)，删除 2 月 29 日，补齐每年 8760 小时；
4. 输出 Regina_cleaned 文件夹与 Regina_ARMA_Result 中后续建模需要的统计文件。

运行：
python Regina_01_clean_hourly_data.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


from project_paths import *

# =========================================================
# 1. 路径设置
# =========================================================

RAW_DIR = REGINA_DIR

CLEAN_DIR = REGINA_CLEANED_DIR

RESULT_DIR = REGINA_ARMA_RESULT_DIR

YEAR_START = 2001
YEAR_END = 2003
YEARS = list(range(YEAR_START, YEAR_END + 1))


# =========================================================
# 2. 工具函数
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去除 BOM、首尾空格，统一列名。"""
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def find_first_existing_column(df: pd.DataFrame, candidates):
    """从候选列名中找第一个存在的列。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_csv_robust(path: Path) -> pd.DataFrame:
    """兼容 utf-8-sig / utf-8 / gbk 的 CSV 读取。"""
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败：{path}\n最后一次错误：{last_err}")


def parse_time_hour(value):
    """从 Time (LST) 中提取小时。兼容 00:00、0000、0 等格式。"""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan

    # 例如 00:00 / 01:00
    m = re.match(r"^(\d{1,2}):", s)
    if m:
        return int(m.group(1))

    # 例如 0000 / 0100 / 2300
    if s.isdigit():
        if len(s) >= 3:
            return int(s[:-2])
        return int(s)

    return np.nan


def make_datetime_from_columns(df: pd.DataFrame) -> pd.Series:
    """优先用 Date/Time (LST)，否则用 Year/Month/Day/Time (LST) 构造时间。"""
    date_col = find_first_existing_column(
        df,
        ["Date/Time (LST)", "Date/Time", "Date Time", "Date/Time (Local Standard Time)"]
    )
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.notna().sum() > 0:
            return dt

    required = ["Year", "Month", "Day"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"既没有可用 Date/Time 列，也缺少 {c} 列，无法构造时间。当前列：{list(df.columns)}")

    time_col = find_first_existing_column(df, ["Time (LST)", "Time", "Hour"])
    if time_col is not None:
        hour = df[time_col].apply(parse_time_hour)
    else:
        hour = 0

    dt = pd.to_datetime(
        dict(
            year=pd.to_numeric(df["Year"], errors="coerce"),
            month=pd.to_numeric(df["Month"], errors="coerce"),
            day=pd.to_numeric(df["Day"], errors="coerce"),
            hour=pd.to_numeric(hour, errors="coerce"),
        ),
        errors="coerce"
    )
    return dt


def extract_wind_column(df: pd.DataFrame) -> str:
    wind_col = find_first_existing_column(
        df,
        [
            "Wind Spd (km/h)",
            "Wind Speed (km/h)",
            "Wind Spd", 
            "Wind Speed",
            "Wind Spd km/h",
        ]
    )
    if wind_col is None:
        raise KeyError(
            "找不到风速列。请确认原始 CSV 中是否存在 Wind Spd (km/h)。\n"
            f"当前列：{list(df.columns)}"
        )
    return wind_col


def clean_one_file(path: Path) -> pd.DataFrame:
    df = read_csv_robust(path)
    df = normalize_columns(df)

    wind_col = extract_wind_column(df)
    dt = make_datetime_from_columns(df)

    out = pd.DataFrame({
        "datetime": dt,
        "wind_speed_kmh": pd.to_numeric(df[wind_col], errors="coerce"),
        "source_file": path.name,
    })

    out = out.dropna(subset=["datetime"])
    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["hour"] = out["datetime"].dt.hour

    out = out[(out["year"] >= YEAR_START) & (out["year"] <= YEAR_END)].copy()

    return out


def complete_one_year(df_year: pd.DataFrame, year: int) -> pd.DataFrame:
    """补齐某一年 8760 小时，并对缺失风速做时间插值。"""
    # 非闰年完整小时索引。2001-2003 本身没有闰日，这里仍统一处理。
    start = pd.Timestamp(year=year, month=1, day=1, hour=0)
    end = pd.Timestamp(year=year, month=12, day=31, hour=23)
    full_index = pd.date_range(start=start, end=end, freq="h")

    # 若遇到闰年，删除 2 月 29 日，确保 8760 小时。
    full_index = full_index[~((full_index.month == 2) & (full_index.day == 29))]

    temp = df_year.copy()
    temp = temp.drop_duplicates(subset=["datetime"], keep="first")
    temp = temp.set_index("datetime").sort_index()

    completed = pd.DataFrame(index=full_index)
    completed["wind_speed_kmh"] = temp["wind_speed_kmh"]
    completed["source_file"] = temp["source_file"]

    missing_before = int(completed["wind_speed_kmh"].isna().sum())

    completed["wind_speed_kmh"] = completed["wind_speed_kmh"].interpolate(method="time", limit_direction="both")
    completed["wind_speed_kmh"] = completed["wind_speed_kmh"].ffill().bfill()

    missing_after = int(completed["wind_speed_kmh"].isna().sum())

    completed = completed.reset_index().rename(columns={"index": "datetime"})
    completed["year"] = completed["datetime"].dt.year
    completed["month"] = completed["datetime"].dt.month
    completed["day"] = completed["datetime"].dt.day
    completed["hour"] = completed["datetime"].dt.hour
    completed["time_lst"] = completed["datetime"].dt.strftime("%H:%M")
    completed["is_interpolated_or_filled"] = completed["source_file"].isna()
    completed["source_file"] = completed["source_file"].fillna("filled_by_script")

    return completed, missing_before, missing_after


def add_hour_index_nonleap(df: pd.DataFrame) -> pd.DataFrame:
    """增加非闰年 0-8759 小时序号。"""
    df = df.copy()
    base = pd.to_datetime(dict(year=df["year"], month=1, day=1, hour=0))
    df["hour_index_8760"] = ((df["datetime"] - base).dt.total_seconds() // 3600).astype(int)
    return df


# =========================================================
# 3. 主程序
# =========================================================
def main():
    print("========== Regina 2001-2003 每时风速数据清洗 ==========")
    print(f"原始数据目录：{RAW_DIR}")
    print(f"清洗输出目录：{CLEAN_DIR}")
    print(f"建模结果目录：{RESULT_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"找不到原始目录：{RAW_DIR}")

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{RAW_DIR} 下没有 csv 文件。请确认 Regina 2001-2003 每时数据已经下载。")

    print(f"发现 CSV 文件数：{len(csv_files)}")

    frames = []
    file_rows = []
    for path in csv_files:
        try:
            one = clean_one_file(path)
            frames.append(one)
            file_rows.append({
                "file": path.name,
                "rows_after_basic_parse": len(one),
                "year_min": int(one["year"].min()) if len(one) else np.nan,
                "year_max": int(one["year"].max()) if len(one) else np.nan,
                "missing_wind_before_completion": int(one["wind_speed_kmh"].isna().sum()),
            })
            print(f"读取完成：{path.name}，有效行数：{len(one)}")
        except Exception as e:
            file_rows.append({
                "file": path.name,
                "rows_after_basic_parse": 0,
                "error": str(e),
            })
            print(f"读取失败：{path.name}，错误：{e}")

    if not frames:
        raise RuntimeError("没有任何文件成功读取。")

    raw_all = pd.concat(frames, ignore_index=True)

    # 删除闰日
    raw_all = raw_all[~((raw_all["month"] == 2) & (raw_all["day"] == 29))].copy()

    completed_years = []
    year_report_rows = []
    for year in YEARS:
        df_year = raw_all[raw_all["year"] == year].copy()
        completed, missing_before, missing_after = complete_one_year(df_year, year)
        completed_years.append(completed)
        year_report_rows.append({
            "year": year,
            "raw_rows_in_year": len(df_year),
            "completed_rows": len(completed),
            "missing_wind_before_interpolation": missing_before,
            "missing_wind_after_interpolation": missing_after,
            "mean_wind_speed_kmh": completed["wind_speed_kmh"].mean(),
            "std_wind_speed_kmh": completed["wind_speed_kmh"].std(ddof=1),
            "min_wind_speed_kmh": completed["wind_speed_kmh"].min(),
            "max_wind_speed_kmh": completed["wind_speed_kmh"].max(),
        })
        out_year_path = CLEAN_DIR / f"Regina_cleaned_hourly_{year}.csv"
        completed.to_csv(out_year_path, index=False, encoding="utf-8-sig")
        print(f"{year} 年完成：{len(completed)} 行，保存：{out_year_path}")

    clean_all = pd.concat(completed_years, ignore_index=True)
    clean_all = add_hour_index_nonleap(clean_all)

    # 保证排序
    clean_all = clean_all.sort_values("datetime").reset_index(drop=True)

    combined_path = CLEAN_DIR / "Regina_hourly_cleaned_2001_2003.csv"
    clean_all.to_csv(combined_path, index=False, encoding="utf-8-sig")

    yearly_stats = pd.DataFrame(year_report_rows)
    yearly_stats_path = RESULT_DIR / "Regina_actual_yearly_statistics_2001_2003.csv"
    yearly_stats.to_csv(yearly_stats_path, index=False, encoding="utf-8-sig")

    file_report = pd.DataFrame(file_rows)
    file_report_path = RESULT_DIR / "Regina_cleaning_file_report.csv"
    file_report.to_csv(file_report_path, index=False, encoding="utf-8-sig")

    # 实际 2001-2003 总体均值和标准差。后续 6-step Common Model 使用这里的实际值，不再使用论文 19.53/10.06。
    summary = pd.DataFrame([
        {
            "data_type": "Actual 2001-2003 cleaned",
            "year_start": YEAR_START,
            "year_end": YEAR_END,
            "hours_total": len(clean_all),
            "mean_wind_speed_kmh": clean_all["wind_speed_kmh"].mean(),
            "std_wind_speed_kmh": clean_all["wind_speed_kmh"].std(ddof=1),
            "min_wind_speed_kmh": clean_all["wind_speed_kmh"].min(),
            "max_wind_speed_kmh": clean_all["wind_speed_kmh"].max(),
        }
    ])
    summary_path = RESULT_DIR / "Regina_actual_mu_sigma_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== 清洗完成 ==========")
    print(f"合并清洗文件：{combined_path}")
    print(f"年度统计文件：{yearly_stats_path}")
    print(f"总体 μ/σ 文件：{summary_path}")
    print(summary.round(6).to_string(index=False))

    if len(clean_all) != 8760 * len(YEARS):
        print("\n警告：最终总行数不是 8760 × 年数，请检查数据。")
    else:
        print(f"\n行数检查通过：{len(clean_all)} = 8760 × {len(YEARS)}")


if __name__ == "__main__":
    main()
