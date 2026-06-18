import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


# =========================================================
# 1. 下载参数设置
# =========================================================

# 保存文件夹
SAVE_DIR = Path(r"D:\tools\za\paper\North Battleford")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# North Battleford A
# 你截图里显示 Climate ID = 4045600
CLIMATE_ID = "4045600"

# 关键：Environment Canada 批量下载接口一般使用 stationID
# 但你给的网址是 climate_id=4045600。
# 下面代码会先尝试从网页中自动解析 stationID；
# 如果解析不到，再尝试直接使用 climate_id 下载。
START_PAGE_URL = (
    "https://climate.weather.gc.ca/climate_data/hourly_data_e.html"
    "?hlyRange=1953-01-01%7C2005-09-01"
    "&dlyRange=1942-03-01%7C2005-09-30"
    "&mlyRange=1942-01-01%7C2005-09-01"
    "&climate_id=4045600"
    "&Prov=SK"
    "&urlExtension=_e.html"
    "&searchType=stnName"
    "&optLimit=yearRange"
    "&StartYear=1989"
    "&EndYear=2003"
    "&selRowPerPage=25"
    "&Line=0"
    "&searchMethod=contains"
    "&Month=1"
    "&Day=1"
    "&txtStationName=North+Battleford"
    "&timeframe=1"
    "&Year=1989"
)

START_YEAR = 1989
END_YEAR = 2003

# 每次下载之间暂停，避免请求过快
SLEEP_SECONDS = 1.0

# 下载接口
BULK_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"


# =========================================================
# 2. 工具函数
# =========================================================

def fetch_text(url: str, timeout: int = 30) -> str:
    """
    读取网页文本。
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()

    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass

    return raw.decode("utf-8", errors="ignore")


def try_parse_station_id(html: str):
    """
    尝试从网页源码中解析 stationID。
    有些页面源码或下载链接中会出现 stationID=xxxxx。
    """
    import re

    patterns = [
        r"stationID=(\d+)",
        r"StationID=(\d+)",
        r"name=[\"']stationID[\"']\s+value=[\"'](\d+)[\"']",
        r"name=[\"']StationID[\"']\s+value=[\"'](\d+)[\"']",
    ]

    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)

    return None


def build_bulk_url_with_station_id(station_id: str, year: int, month: int) -> str:
    """
    使用 stationID 构造批量下载 URL。
    timeframe=1 表示小时数据。
    format=csv 表示 CSV。
    Day 对小时月度下载基本不影响，保留 Day=1 即可。
    """
    params = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data"
    }

    return BULK_URL + "?" + urllib.parse.urlencode(params)


def build_bulk_url_with_climate_id(year: int, month: int) -> str:
    """
    备用方案：使用 climate_id 构造下载 URL。
    如果 stationID 下载失败，则尝试这个。
    """
    params = {
        "format": "csv",
        "climate_id": CLIMATE_ID,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data"
    }

    return BULK_URL + "?" + urllib.parse.urlencode(params)


def build_hourly_page_csv_url(year: int, month: int) -> str:
    """
    备用方案：直接在 hourly_data_e 页面参数上加 format=csv。
    某些情况下网页按钮会走这种形式。
    """
    params = {
        "hlyRange": "1953-01-01|2005-09-01",
        "dlyRange": "1942-03-01|2005-09-30",
        "mlyRange": "1942-01-01|2005-09-01",
        "climate_id": CLIMATE_ID,
        "Prov": "SK",
        "urlExtension": "_e.html",
        "searchType": "stnName",
        "optLimit": "yearRange",
        "StartYear": START_YEAR,
        "EndYear": END_YEAR,
        "selRowPerPage": 25,
        "Line": 0,
        "searchMethod": "contains",
        "Month": month,
        "Day": 1,
        "txtStationName": "North Battleford",
        "timeframe": 1,
        "Year": year,
        "format": "csv"
    }

    url = "https://climate.weather.gc.ca/climate_data/hourly_data_e.html"
    return url + "?" + urllib.parse.urlencode(params)


def download_file(url: str, save_path: Path, timeout: int = 60):
    """
    下载单个 CSV 文件。
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()

    text_head = content[:500].decode("utf-8", errors="ignore").lower()

    # 简单判断是否真的是 CSV 数据，而不是错误网页
    if "year" not in text_head or "month" not in text_head:
        raise RuntimeError("下载内容不像 CSV，可能是网页错误页或参数不正确。")

    save_path.write_bytes(content)


def expected_file_name(year: int, month: int) -> str:
    """
    按 Environment Canada 常见命名方式保存文件。
    """
    return f"en_climate_hourly_SK_{CLIMATE_ID}_{month:02d}-{year}_P1H.csv"


# =========================================================
# 3. 主下载流程
# =========================================================

def main():
    print("========== North Battleford 批量下载 ==========")
    print(f"保存路径：{SAVE_DIR}")
    print(f"年份范围：{START_YEAR}-{END_YEAR}")
    print(f"Climate ID：{CLIMATE_ID}")

    # 第一步：尝试从页面中解析 stationID
    print("\n正在尝试解析 stationID ...")

    station_id = None

    try:
        html = fetch_text(START_PAGE_URL)
        station_id = try_parse_station_id(html)
    except Exception as e:
        print(f"解析页面失败，后续会使用备用方案：{e}")

    if station_id:
        print(f"解析到 stationID：{station_id}")
    else:
        print("未解析到 stationID，将使用 climate_id 备用下载方式。")

    total = 0
    success = 0
    skipped = 0
    failed = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            total += 1

            file_name = expected_file_name(year, month)
            save_path = SAVE_DIR / file_name

            if save_path.exists() and save_path.stat().st_size > 1000:
                print(f"[跳过] {year}-{month:02d} 已存在：{file_name}")
                skipped += 1
                continue

            print(f"\n[下载] {year}-{month:02d}")

            candidate_urls = []

            if station_id:
                candidate_urls.append(
                    ("stationID", build_bulk_url_with_station_id(station_id, year, month))
                )

            candidate_urls.append(
                ("climate_id", build_bulk_url_with_climate_id(year, month))
            )

            candidate_urls.append(
                ("hourly_page_csv", build_hourly_page_csv_url(year, month))
            )

            month_ok = False
            last_error = None

            for method_name, url in candidate_urls:
                try:
                    print(f"  尝试方式：{method_name}")
                    download_file(url, save_path)
                    print(f"  成功保存：{save_path}")
                    success += 1
                    month_ok = True
                    break

                except Exception as e:
                    last_error = e
                    print(f"  失败：{e}")

            if not month_ok:
                failed.append({
                    "year": year,
                    "month": month,
                    "error": str(last_error)
                })
                print(f"  [失败] {year}-{month:02d}")

            time.sleep(SLEEP_SECONDS)

    print("\n========== 下载完成 ==========")
    print(f"理论文件数：{total}")
    print(f"成功下载数：{success}")
    print(f"跳过已有数：{skipped}")
    print(f"失败数：{len(failed)}")

    if failed:
        fail_log = SAVE_DIR / "download_failed_log.txt"
        with open(fail_log, "w", encoding="utf-8") as f:
            for item in failed:
                f.write(
                    f"{item['year']}-{item['month']:02d} | {item['error']}\n"
                )

        print(f"\n失败记录已保存：{fail_log}")
        print("如果失败很多，通常是 stationID 没解析出来，需要手动补 stationID。")
    else:
        print("\n全部月份下载成功。")


if __name__ == "__main__":
    main()