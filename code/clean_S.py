import re
import calendar
from pathlib import Path

import pandas as pd


# =========================================================
# 1. 基本参数设置
# =========================================================

# 原始文件夹名称
# 如果你的代码和 SwiftCurrent 文件夹在同一级目录，保持这样即可
RAW_DIR = Path("SwiftCurrent")

# 清洗后的文件夹，不会覆盖原始数据
CLEAN_DIR = Path("SwiftCurrent_cleaned")
CLEAN_DIR.mkdir(exist_ok=True)

# 年份范围：1989-2003，共15年
START_YEAR = 1989
END_YEAR = 2003

EXPECTED_YEARS = list(range(START_YEAR, END_YEAR + 1))
EXPECTED_MONTHS = list(range(1, 13))

# 文件名格式示例：
# en_climate_hourly_SK_4028040_01-1989_P1H.csv
FILE_PATTERN = re.compile(
    r".*_(\d{2})-(\d{4})_P1H\.csv$",
    re.IGNORECASE
)

# 需要保留的列
KEEP_COLUMNS = [
    "Year",
    "Month",
    "Day",
    "Time (LST)",
    "Wind Spd (km/h)"
]


# =========================================================
# 2. 读取文件工具函数
# =========================================================

def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    尝试使用多种编码读取 CSV 文件。
    Environment Canada 的数据一般 utf-8-sig 或 latin1 可读。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp1252", "latin1"]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"文件无法读取：{file_path}，最后错误：{last_error}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理列名中的空格、BOM隐藏字符等。
    """
    df = df.copy()
    df.columns = [
        str(col).strip().replace("\ufeff", "")
        for col in df.columns
    ]
    return df


# =========================================================
# 3. 文件名解析与完整性检查
# =========================================================

def parse_year_month_from_filename(file_path: Path):
    """
    从文件名中解析月份和年份。

    示例：
    en_climate_hourly_SK_4028040_01-1989_P1H.csv
    解析结果：
    month = 1
    year = 1989
    """
    match = FILE_PATTERN.match(file_path.name)

    if not match:
        return None, None

    month = int(match.group(1))
    year = int(match.group(2))

    return year, month


def check_file_completeness(raw_dir: Path):
    """
    检查 SwiftCurrent 文件夹下的 CSV 文件是否完整：
    1. 是否存在文件夹；
    2. 是否有 CSV 文件；
    3. 文件名是否符合规则；
    4. 1989-2003 年每年 12 个月是否齐全；
    5. 是否有重复的 年-月 文件。
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"没有找到原始数据文件夹：{raw_dir.resolve()}")

    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"{raw_dir.resolve()} 文件夹下没有 CSV 文件")

    file_map = {}
    invalid_files = []

    for file_path in csv_files:
        year, month = parse_year_month_from_filename(file_path)

        if year is None or month is None:
            invalid_files.append(file_path.name)
            continue

        key = (year, month)

        if key not in file_map:
            file_map[key] = []

        file_map[key].append(file_path)

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
    print(f"扫描到 CSV 文件总数：{len(csv_files)}")
    print(f"理论应有文件数：{len(expected_keys)}")
    print(f"识别到有效年月文件数：{len(actual_keys)}")

    if invalid_files:
        print("\n[文件名不符合规则]")
        for name in invalid_files:
            print("  -", name)

    if missing_keys:
        print("\n[缺失的年月文件]")
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
                print("      ", f.name)

    has_error = bool(
        invalid_files
        or missing_keys
        or extra_keys
        or duplicate_keys
    )

    if has_error:
        raise RuntimeError("文件完整性检查未通过，请先处理上面列出的文件问题。")

    print("\n文件完整性检查通过：1989-2003 年每个月份均存在，且没有重复文件。")

    return file_map


# =========================================================
# 4. 时间与小时数检查函数
# =========================================================

def expected_hours_in_month(year: int, month: int) -> int:
    """
    返回清洗后该月应该有多少小时。
    闰年的 2 月 29 日会被删除，所以所有年份的 2 月都按 28 天计算。
    """
    if month == 2:
        days = 28
    else:
        days = calendar.monthrange(year, month)[1]

    return days * 24


def expected_datetime_range(year: int, month: int):
    """
    生成该月清洗后应有的逐小时时间序列。
    闰年 2 月只生成到 2 月 28 日 23:00。
    """
    start = pd.Timestamp(year=year, month=month, day=1, hour=0)

    if month == 2:
        end_day = 28
    else:
        end_day = calendar.monthrange(year, month)[1]

    end = pd.Timestamp(year=year, month=month, day=end_day, hour=23)

    return pd.date_range(start=start, end=end, freq="h")


def build_datetime_column(df: pd.DataFrame) -> pd.Series:
    """
    根据 Year, Month, Day, Time (LST) 构造临时 datetime。
    这个列只用于校验和排序，不会保存到清洗后的 CSV 中。
    """
    date_part = (
        df["Year"].astype(str).str.zfill(4) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Day"].astype(str).str.zfill(2)
    )

    time_part = df["Time (LST)"].astype(str).str.strip()

    datetime_str = date_part + " " + time_part

    return pd.to_datetime(datetime_str, errors="coerce")


# =========================================================
# 5. 清洗单个 CSV 文件
# =========================================================

def clean_one_file(file_path: Path, year_from_name: int, month_from_name: int) -> dict:
    """
    清洗单个 CSV 文件。

    清洗内容：
    1. 只保留 Year, Month, Day, Time (LST), Wind Spd (km/h)；
    2. Year, Month, Day 转为整数；
    3. Wind Spd (km/h) 转为数值；
    4. 删除闰年 2 月 29 日；
    5. 检查文件内部年月是否和文件名一致；
    6. 检查该月小时数是否正确；
    7. 检查重复时间、缺失小时、风速缺失；
    8. 单独保存清洗后的 CSV，不合并。
    """
    df = read_csv_safely(file_path)
    df = normalize_column_names(df)

    missing_columns = [
        col for col in KEEP_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{file_path.name} 缺少必要列：{missing_columns}\n"
            f"当前文件实际列名为：{list(df.columns)}"
        )

    # 只保留指定列
    cleaned = df[KEEP_COLUMNS].copy()

    original_rows = len(cleaned)

    # 数值转换
    for col in ["Year", "Month", "Day"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned["Wind Spd (km/h)"] = pd.to_numeric(
        cleaned["Wind Spd (km/h)"],
        errors="coerce"
    )

    # 删除年月日为空的无效行
    before_drop_date_na = len(cleaned)
    cleaned = cleaned.dropna(subset=["Year", "Month", "Day"]).copy()
    dropped_date_na_rows = before_drop_date_na - len(cleaned)

    # 转整数
    cleaned["Year"] = cleaned["Year"].astype(int)
    cleaned["Month"] = cleaned["Month"].astype(int)
    cleaned["Day"] = cleaned["Day"].astype(int)

    # 检查文件内部年月
    years_inside = sorted(cleaned["Year"].unique().tolist())
    months_inside = sorted(cleaned["Month"].unique().tolist())

    year_month_ok = (
        years_inside == [year_from_name]
        and months_inside == [month_from_name]
    )

    # 删除闰年 2 月 29 日
    before_drop_feb29 = len(cleaned)

    feb29_mask = (
        (cleaned["Month"] == 2)
        & (cleaned["Day"] == 29)
    )

    cleaned = cleaned[~feb29_mask].copy()

    dropped_feb29_rows = before_drop_feb29 - len(cleaned)

    # 构造临时 datetime，用于排序和校验
    cleaned["__datetime__"] = build_datetime_column(cleaned)

    invalid_datetime_count = cleaned["__datetime__"].isna().sum()

    # 检查重复的年月日时记录
    duplicated_time_rows = cleaned.duplicated(
        subset=["Year", "Month", "Day", "Time (LST)"],
        keep=False
    ).sum()

    # 检查缺失小时和额外小时
    expected_dt = expected_datetime_range(year_from_name, month_from_name)

    actual_dt = cleaned["__datetime__"].dropna()

    actual_dt_set = set(actual_dt)
    expected_dt_set = set(expected_dt)

    missing_hours = sorted(expected_dt_set - actual_dt_set)
    extra_hours = sorted(actual_dt_set - expected_dt_set)

    missing_hour_count = len(missing_hours)
    extra_hour_count = len(extra_hours)

    # 风速缺失和异常检查
    wind_missing_count = cleaned["Wind Spd (km/h)"].isna().sum()
    wind_negative_count = (cleaned["Wind Spd (km/h)"] < 0).sum()

    # 排序
    cleaned = cleaned.sort_values(
        by=["__datetime__"],
        na_position="last"
    ).reset_index(drop=True)

    # 删除临时校验列，最终文件只保留5列
    cleaned = cleaned[KEEP_COLUMNS]

    actual_hours = len(cleaned)
    expected_hours = expected_hours_in_month(year_from_name, month_from_name)

    hour_count_ok = actual_hours == expected_hours

    datetime_complete_ok = (
        invalid_datetime_count == 0
        and duplicated_time_rows == 0
        and missing_hour_count == 0
        and extra_hour_count == 0
    )

    file_ok = (
        year_month_ok
        and hour_count_ok
        and datetime_complete_ok
    )

    # 保存清洗后的单月文件
    output_name = file_path.stem + "_cleaned.csv"
    output_path = CLEAN_DIR / output_name

    cleaned.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 只在报告中记录前几个缺失小时，避免报告过长
    missing_hours_sample = "; ".join(
        str(x) for x in missing_hours[:10]
    )

    extra_hours_sample = "; ".join(
        str(x) for x in extra_hours[:10]
    )

    return {
        "file_name": file_path.name,
        "year": year_from_name,
        "month": month_from_name,
        "original_rows": original_rows,
        "dropped_date_na_rows": dropped_date_na_rows,
        "dropped_feb29_rows": dropped_feb29_rows,
        "actual_hours_after_cleaning": actual_hours,
        "expected_hours": expected_hours,
        "hour_count_ok": hour_count_ok,
        "years_inside": str(years_inside),
        "months_inside": str(months_inside),
        "year_month_ok": year_month_ok,
        "invalid_datetime_count": invalid_datetime_count,
        "duplicated_time_rows": duplicated_time_rows,
        "missing_hour_count": missing_hour_count,
        "extra_hour_count": extra_hour_count,
        "wind_missing_count": wind_missing_count,
        "wind_negative_count": wind_negative_count,
        "datetime_complete_ok": datetime_complete_ok,
        "file_ok": file_ok,
        "missing_hours_sample": missing_hours_sample,
        "extra_hours_sample": extra_hours_sample,
        "output_file": str(output_path)
    }


# =========================================================
# 6. 批量清洗所有 CSV，但不合并
# =========================================================

def clean_all_files(file_map: dict):
    """
    对所有月度 CSV 文件逐个清洗。
    每个文件单独保存，不进行合并。
    """
    reports = []

    print("\n========== 开始逐文件清洗 ==========")

    for year in EXPECTED_YEARS:
        for month in EXPECTED_MONTHS:
            file_path = file_map[(year, month)][0]

            try:
                report = clean_one_file(
                    file_path=file_path,
                    year_from_name=year,
                    month_from_name=month
                )

                reports.append(report)

                status = "通过" if report["file_ok"] else "异常"

                print(
                    f"{year}-{month:02d} | {status} | "
                    f"实际 {report['actual_hours_after_cleaning']} / "
                    f"应有 {report['expected_hours']} 小时 | "
                    f"风速缺失 {report['wind_missing_count']} 个 | "
                    f"重复时间 {report['duplicated_time_rows']} 行"
                )

            except Exception as e:
                reports.append({
                    "file_name": file_path.name,
                    "year": year,
                    "month": month,
                    "file_ok": False,
                    "error": str(e)
                })

                print(f"{year}-{month:02d} | 读取或清洗失败 | {e}")

    report_df = pd.DataFrame(reports)

    # 保存逐文件清洗报告
    report_path = CLEAN_DIR / "SwiftCurrent_cleaning_report.csv"

    report_df.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 生成逐年小时数报告
    valid_hour_report = report_df.dropna(
        subset=["actual_hours_after_cleaning"]
    ).copy()

    annual_report = (
        valid_hour_report
        .groupby("year", as_index=False)["actual_hours_after_cleaning"]
        .sum()
        .rename(columns={
            "actual_hours_after_cleaning": "annual_hours_after_cleaning"
        })
    )

    annual_report["expected_annual_hours"] = 8760

    annual_report["annual_hour_count_ok"] = (
        annual_report["annual_hours_after_cleaning"]
        == annual_report["expected_annual_hours"]
    )

    annual_report_path = CLEAN_DIR / "SwiftCurrent_annual_hour_count_report.csv"

    annual_report.to_csv(
        annual_report_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 清洗完成 ==========")
    print(f"清洗后的单月 CSV 文件夹：{CLEAN_DIR.resolve()}")
    print(f"逐文件清洗报告：{report_path.resolve()}")
    print(f"逐年小时数报告：{annual_report_path.resolve()}")

    # 输出异常文件
    abnormal_files = report_df[
        report_df["file_ok"] != True
    ]

    if not abnormal_files.empty:
        print("\n========== 异常文件汇总 ==========")
        display_cols = [
            "file_name",
            "year",
            "month",
            "actual_hours_after_cleaning",
            "expected_hours",
            "year_month_ok",
            "hour_count_ok",
            "invalid_datetime_count",
            "duplicated_time_rows",
            "missing_hour_count",
            "extra_hour_count",
            "wind_missing_count",
            "wind_negative_count"
        ]

        display_cols = [
            col for col in display_cols
            if col in abnormal_files.columns
        ]

        print(abnormal_files[display_cols].to_string(index=False))
    else:
        print("\n所有文件均通过校验。")

    print("\n========== 逐年 8760 小时检查 ==========")
    print(annual_report.to_string(index=False))


# =========================================================
# 7. 主程序入口
# =========================================================

if __name__ == "__main__":
    file_map = check_file_completeness(RAW_DIR)
    clean_all_files(file_map)